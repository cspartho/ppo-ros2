# PPO with ROS 2 and Gazebo

A ROS 2 Python package that trains a Stable-Baselines3 PPO policy for
TurtleBot3-style laser-based obstacle avoidance. The Gymnasium environment
communicates with a running Gazebo Sim world through ROS 2 topics and resets
the world through `ros_gz_interfaces/srv/ControlWorld`.

## Requirements

- ROS 2 Lyrical on Ubuntu 26.04
- Gazebo Sim with a TurtleBot3 simulation compatible with your ROS 2
  installation
- `ros_gz_interfaces` and a Gazebo/ROS bridge
- Python packages in `requirements.txt`

Install the ROS packages:

```bash
sudo apt install \
  ros-lyrical-ros-gz-interfaces \
  ros-lyrical-ros-gz-bridge \
  ros-lyrical-turtlebot3-gazebo
```

## Install

Use a virtual environment that can also see the ROS 2 Python packages, then
install the training dependencies:

```bash
source /opt/ros/lyrical/setup.bash
/usr/bin/python3.14 -m venv --system-site-packages .venv-ros
source .venv-ros/bin/activate
python -m pip install --no-cache-dir -r requirements.txt
```

Build and source the package:

```bash
source /opt/ros/lyrical/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## Start the simulator

Start a TurtleBot3 Burger world in one terminal:

```bash
source /opt/ros/lyrical/setup.bash
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

In a second terminal, start the Gazebo world-control service bridge:

```bash
source /opt/ros/lyrical/setup.bash
ros2 run ros_gz_bridge parameter_bridge \
  /world/default/control@ros_gz_interfaces/srv/ControlWorld
```

The TurtleBot3 launch bridges `sensor_msgs/msg/LaserScan` on `/scan` and
`geometry_msgs/msg/TwistStamped` on `/cmd_vel`. Verify all interfaces before
training:


```bash
ros2 topic info /scan
ros2 topic info /cmd_vel
ros2 service type /world/default/control
```

If your simulator uses different topic names, pass them to the trainer. The
`world-name` argument selects `/world/<name>/control`. TurtleBot3's launch file
loads the world as `default`, so use that name for the reset service even though
the world file is called `turtlebot3_world.world`.

If training reports that `/world/turtlebot3_world/control` is unavailable,
remove any old `--world-name turtlebot3_world` argument or replace it with
`--world-name default`. Check the service name with:

```bash
ros2 service type /world/default/control
```

### Troubleshooting: `NoneType` has no attribute `lower`

The `ros_gz_sim` launch file resolves the Gazebo executable with
`shutil.which("gz")`. If the ROS environment is not sourced in the terminal
running the launch, this can return `None` and cause the `.lower()` error.
Source ROS and verify `gz` resolves before launching:

```bash
source /opt/ros/lyrical/setup.bash
command -v gz
echo "$TURTLEBOT3_MODEL"
```

`command -v gz` should print a path (on this installation, it is under
`/opt/ros/lyrical/opt/gz_tools_vendor/bin/`). If it prints nothing, re-source
the ROS setup in that same terminal and retry. Set the robot model there too if
needed:

```bash
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

## Train

With the simulator running and unpaused:

```bash
python -m ppo_ros2.train \
  --world-name default \
  --timesteps 100000 \
  --model-path models/turtlebot3_ppo
```

Run the command after sourcing `install/setup.bash` and activating `.venv-ros`.
Alternatively, if the training dependencies are installed into the Python
environment used by ROS 2, use `ros2 run ppo_ros2 train_ppo` with the same
arguments.

For namespaced topics:

```bash
python -m ppo_ros2.train \
  --scan-topic /robot/scan \
  --cmd-topic /robot/cmd_vel
```

The environment uses 32 pooled laser observations and continuous linear and
angular velocity actions. Episodes end on a near-collision; Gazebo resets the
world between episodes. The trained model is saved to the requested path.

## Run the focused tests

```bash
python3 -m unittest discover -s test
```
