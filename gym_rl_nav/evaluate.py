#!/usr/bin/env python3
import os
import time
import csv
import rclpy
from stable_baselines3 import PPO

# Direct import from your local file
from src.gym_rl_nav.gym_nav_env import GymNavEnv


def main():
    rclpy.init()  # Initialize ROS 2

    env = GymNavEnv()

    # Load the trained model (".zip" is auto-appended by SB3)
    model_path = os.path.join(os.getcwd(), 'logs', 'ppo_nav', 'ppo_nav_final')
    model = PPO.load(model_path)

    # Prepare results CSV
    csv_file = 'results.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'total_reward', 'steps', 'success'])

    # Run evaluation
    num_episodes = 50
    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
            time.sleep(0.01)  # Give Gazebo some breathing room

        success = 1 if total_reward > 0 else 0

        # Log to CSV
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([ep, total_reward, steps, success])

        print(f"Episode {ep}: Reward={total_reward:.2f}, Steps={steps}, Success={success}")

    # Cleanup
    env.close()
    rclpy.shutdown()
    print(f"\n Evaluation completed. Results saved to: {csv_file}")


if __name__ == '__main__':
    main()

