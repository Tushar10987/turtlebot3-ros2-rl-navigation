# ROS 2 + Reinforcement Learning for TurtleBot3 Navigation

This project demonstrates autonomous navigation of a TurtleBot3 robot in a Gazebo simulation using Reinforcement Learning (RL) via Stable-Baselines3's PPO algorithm. The agent learns how to reach a goal while avoiding obstacles using laser scan data and ROS 2 Humble.

## Features

- Custom Gymnasium environment integrated with ROS 2
- PPO (Proximal Policy Optimization) training using Stable-Baselines3
- Simulated robot control via `/cmd_vel`, `/scan`, `/odom`
- Dynamic obstacle spawning
- Live training monitoring using TensorBoard
- Evaluation of trained model with success metrics logging

## Project Structure
ros2_rl_ws/
└── src/
└── gym_rl_nav/
├── src/gym_rl_nav/
│ ├── gym_nav_env.py
│ └── init.py
├── train.py
├── evaluate.py
├── logs/ppo_nav/
├── README.md
├── LICENSE
└── .gitignore


## Setup Instructions

### Prerequisites

- Ubuntu 22.04
- ROS 2 Humble installed
- Conda or virtualenv for Python environment
- Gazebo Fortress or Ignition
- TurtleBot3 packages installed

### Install dependencies

```bash
git clone https://github.com/Tushar10987/turtlebot3-ros2-rl-navigation.git 
cd turtlebot3-ros2-rl-navigation

conda create -n rl_nav_env python=3.10
conda activate rl_nav_env

pip install stable-baselines3 gymnasium numpy torch matplotlib tensorboard

source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
cd gym_rl_nav

## Lauch Gazebo and turtlebot
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

## Train the Agent
python3 train_ppo.py

##View TensorBoard
tensorboard --logdir logs/ppo_nav --port 6006
http://localhost:6006/     --- search in browser to view the process

##Evaluate the Trained Model
python3 evaluate.py

