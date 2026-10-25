from __future__ import annotations

import threading
import time
from typing import Any

import gymnasium as gym
import numpy as np
import rclpy
from geometry_msgs.msg import TwistStamped
from gymnasium import spaces
from rclpy.executors import SingleThreadedExecutor
from ros_gz_interfaces.srv import ControlWorld
from sensor_msgs.msg import LaserScan

from ppo_ros2.observations import scan_to_observation


class Ros2PpoEnv(gym.Env[np.ndarray, np.ndarray]):
    """Continuous-control obstacle-avoidance environment for PPO."""

    metadata: dict[str, list[str]] = {"render_modes": []}

    def __init__(
        self,
        scan_topic: str = "/scan",
        cmd_topic: str = "/cmd_vel",
        world_name: str = "default",
        action_duration: float = 0.25,
        scan_timeout: float = 5.0,
        observation_bins: int = 32,
        collision_distance: float = 0.20,
        max_episode_steps: int = 500,
        max_linear_speed: float = 0.22,
        max_angular_speed: float = 1.5,
    ) -> None:
        super().__init__()
        if action_duration <= 0 or scan_timeout <= 0:
            raise ValueError("action_duration and scan_timeout must be positive.")
        if observation_bins <= 0 or max_episode_steps <= 0:
            raise ValueError("observation_bins and max_episode_steps must be positive.")

        self._action_duration = action_duration
        self._scan_timeout = scan_timeout
        self._observation_bins = observation_bins
        self._collision_distance = collision_distance
        self._max_episode_steps = max_episode_steps
        self._max_linear_speed = max_linear_speed
        self._max_angular_speed = max_angular_speed
        self._step_count = 0
        self._latest_observation: np.ndarray | None = None
        self._latest_min_distance = 0.0
        self._scan_version = 0
        self._scan_condition = threading.Condition()
        self._owns_rclpy = not rclpy.ok()

        if self._owns_rclpy:
            rclpy.init()
        self._node = rclpy.create_node("ppo_training_environment")
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(
            target=self._executor.spin,
            name="ppo-ros2-executor",
            daemon=True,
        )
        self._spin_thread.start()

        self._cmd_publisher = self._node.create_publisher(TwistStamped, cmd_topic, 10)
        self._scan_subscription = self._node.create_subscription(
            LaserScan,
            scan_topic,
            self._on_scan,
            10,
        )
        self._reset_service_name = f"/world/{world_name}/control"
        self._reset_client = self._node.create_client(
            ControlWorld,
            self._reset_service_name,
        )

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(observation_bins,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.asarray([0.0, -max_angular_speed], dtype=np.float32),
            high=np.asarray([max_linear_speed, max_angular_speed], dtype=np.float32),
            dtype=np.float32,
        )

    def _on_scan(self, message: LaserScan) -> None:
        observation = scan_to_observation(
            message.ranges,
            message.range_min,
            message.range_max,
            self._observation_bins,
        )
        minimum_distance = float(np.min(observation) * message.range_max)
        with self._scan_condition:
            self._latest_observation = observation
            self._latest_min_distance = minimum_distance
            self._scan_version += 1
            self._scan_condition.notify_all()

    def _wait_for_scan(self, newer_than: int) -> tuple[np.ndarray, float]:
        deadline = time.monotonic() + self._scan_timeout
        with self._scan_condition:
            while self._scan_version <= newer_than or self._latest_observation is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "No fresh LaserScan received. Check that the simulator is "
                        "running and publishing the configured scan topic."
                    )
                self._scan_condition.wait(remaining)
            return self._latest_observation.copy(), self._latest_min_distance

    def _reset_world(self) -> None:
        if not self._reset_client.wait_for_service(timeout_sec=self._scan_timeout):
            raise RuntimeError(
                "Gazebo world reset service is unavailable: "
                f"{self._reset_service_name}. Ensure ros_gz_interfaces is "
                "installed and the Gazebo ROS bridge is running. For the "
                "standard TurtleBot3 world launch, the world name is 'default'; "
                "pass --world-name default (not the world file name)."
            )

        request = ControlWorld.Request()
        request.world_control.reset.all = True
        request.world_control.pause = False
        future = self._reset_client.call_async(request)
        deadline = time.monotonic() + self._scan_timeout
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not future.done():
            raise TimeoutError(f"Timed out waiting for {self._reset_service_name}.")
        response = future.result()
        if response is None or not response.success:
            raise RuntimeError(f"Gazebo rejected world reset: {self._reset_service_name}.")

    def _publish_action(self, action: np.ndarray) -> None:
        command = TwistStamped()
        command.header.stamp = self._node.get_clock().now().to_msg()
        command.twist.linear.x = float(action[0])
        command.twist.angular.z = float(action[1])
        self._cmd_publisher.publish(command)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        with self._scan_condition:
            previous_version = self._scan_version
        self._publish_action(np.zeros(2, dtype=np.float32))
        self._reset_world()
        observation, minimum_distance = self._wait_for_scan(previous_version)
        self._step_count = 0
        return observation, {"min_obstacle_distance": minimum_distance}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        clipped_action = np.clip(action, self.action_space.low, self.action_space.high)
        with self._scan_condition:
            previous_version = self._scan_version
        self._publish_action(clipped_action)
        time.sleep(self._action_duration)
        observation, minimum_distance = self._wait_for_scan(previous_version)

        self._step_count += 1
        collision = minimum_distance <= self._collision_distance
        proximity_penalty = max(
            0.0,
            (self._collision_distance * 3.0 - minimum_distance)
            / (self._collision_distance * 3.0),
        )
        forward_reward = float(clipped_action[0] / self._max_linear_speed)
        turning_penalty = 0.1 * abs(float(clipped_action[1])) / self._max_angular_speed
        reward = 0.2 + 0.8 * forward_reward - turning_penalty - 0.5 * proximity_penalty
        if collision:
            reward -= 10.0

        truncated = self._step_count >= self._max_episode_steps
        return (
            observation,
            reward,
            collision,
            truncated,
            {
                "min_obstacle_distance": minimum_distance,
                "collision": collision,
            },
        )

    def close(self) -> None:
        if getattr(self, "_executor", None) is None:
            return
        self._publish_action(np.zeros(2, dtype=np.float32))
        self._executor.shutdown()
        self._spin_thread.join(timeout=self._scan_timeout)
        self._node.destroy_node()
        self._executor = None
        if self._owns_rclpy and rclpy.ok():
            rclpy.shutdown()
