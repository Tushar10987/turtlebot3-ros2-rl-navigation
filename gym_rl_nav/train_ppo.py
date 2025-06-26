#!/usr/bin/env python3

import os
import time
import rclpy

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from gym_rl_nav.gym_nav_env import GymNavEnv  

rclpy.init()

def make_env():
    return GymNavEnv()


def main():
    log_dir = os.path.join(os.getcwd(), "logs", "ppo_nav")
    os.makedirs(log_dir, exist_ok=True)

    vec_env = DummyVecEnv([lambda: Monitor(make_env(), log_dir)])

    # Callback to save model checkpoints every 100k steps
    checkpoint_cb = CheckpointCallback(
        save_freq=100_000,
        save_path=log_dir,
        name_prefix="ppo_nav"
    )

    # Initialize PPO model
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        clip_range=0.2,
        ent_coef=0.01,
        gamma=0.99,
    )

    # Train for 1 million timesteps
    print(" Starting training...")
    start_time = time.time()
    model.learn(total_timesteps=1_000_000, callback=checkpoint_cb)
    duration = (time.time() - start_time) / 60.0
    print(f"Training completed in {duration:.1f} minutes")

    final_model_path = os.path.join(log_dir, "ppo_nav_final")
    model.save(final_model_path)
    print(f" Final model saved to: {final_model_path}")

    rclpy.shutdown()

if __name__ == "__main__":
    main()

