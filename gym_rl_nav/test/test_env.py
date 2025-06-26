import time
import random
from gym_rl_nav.gym_nav_env import GymNavEnv

def main():
    env = GymNavEnv()
    obs = env.reset()

    total_reward = 0
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        print(f"Step {i}, Action: {action}, Reward: {reward:.2f}, Done: {done}")
        total_reward += reward
        time.sleep(0.1)
        if done:
            break

    print(f"Total reward: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    main()
