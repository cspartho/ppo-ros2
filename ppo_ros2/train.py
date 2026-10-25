import argparse
from pathlib import Path
from stable_baselines3 import PPO
from ppo_ros2.env import Ros2PpoEnv

def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO for ROS 2 obstacle avoidance.")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--model-path", type=Path, default=Path("models/turtlebot3_ppo"))
    parser.add_argument("--scan-topic", default="/scan")
    parser.add_argument("--cmd-topic", default="/cmd_vel")
    parser.add_argument("--world-name", default="default")
    parser.add_argument("--action-duration", type=float, default=0.25)
    args = parser.parse_args()

    if args.timesteps <= 0:
        parser.error("--timesteps must be positive.")

    environment = Ros2PpoEnv(
        scan_topic=args.scan_topic,
        cmd_topic=args.cmd_topic,
        world_name=args.world_name,
        action_duration=args.action_duration,
    )
    try:
        model = PPO("MlpPolicy", environment, verbose=1)
        model.learn(total_timesteps=args.timesteps)
        args.model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(args.model_path))
    finally:
        environment.close()


if __name__ == "__main__":
    main()
