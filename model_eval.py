"""
A simple script to evaluate the performance of a trained rl model.
"""

import numpy as np
import torch
from stable_baselines3 import PPO
from herdingenv import HerdingSimEnv
import tqdm

# custom function to create environment instances
def make_env():
    def _init():
        # Simulation initialization data
        arena_length = 15  # meters
        arena_width = 15   # meters
        num_sheep = 1
        num_sheepdogs = 1
        robot_wheel_radius = 0.1  # meters
        robot_distance_between_wheels = 0.2  # meters
        max_wheel_velocity = 8.0  # m/s

        # robots = [
        #     [2.0, 2.0, np.pi/4],  # sheep-dog 1
        #     [5.0, 5.0, np.pi/4],  # sheep 1
        # ]

        # Create the environment
        env = HerdingSimEnv(arena_length, arena_width, num_sheep, num_sheepdogs, robot_distance_between_wheels, robot_wheel_radius, max_wheel_velocity)

        # print(f"Environment initialized in process ID: {os.getpid()}")

        return env
    return _init

if __name__ == "__main__":
    num_sims = 100
    env = make_env()()

    # Load model
    model_path = ""
    # model = PPO.load(model_path, env=env)
    model = PPO('MlpPolicy', env=env)

    # Initialize metrics
    metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'successful_episodes': 0,
        'success_rate': 0,
    }

    # Evaluate the model
    print(f"Starting evaluation of model")
    with torch.no_grad():
        for sim in tqdm.tqdm(range(num_sims)):
            obs, info = env.reset()
            terminated = False
            truncated = False
            episode_reward = 0
            episode_length = 0
            while not terminated and not truncated:
                action, _ = model.predict(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                episode_length += 1
                # env.render(mode='human', fps=200)

            # Record metrics
            metrics['episode_rewards'].append(episode_reward)
            metrics['episode_lengths'].append(episode_length)
            if terminated:
                metrics['successful_episodes'] += 1

            # Close the environment
            env.close()

    # Calculate success rate
    metrics['success_rate'] = metrics['successful_episodes'] / num_sims

    # Display metrics
    print(f"Model evaluation complete")
    print(f"Average episode reward: {np.mean(metrics['episode_rewards'])}")
    print(f"Average episode length: {np.mean(metrics['episode_lengths'])}")
    # print(f"Episode Lengths: {metrics['episode_lengths']}")
    print(f"Success rate: {metrics['success_rate']}")
    print(f"Number of successful episodes: {metrics['successful_episodes']}")