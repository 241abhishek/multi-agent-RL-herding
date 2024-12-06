"""
A simple script to evaluate the performance of a trained rl model.
"""

import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecNormalize
from herdingenv import HerdingSimEnv
import tqdm
import time

NUM_SHEEP = 10
NUM_SHEEPDOGS = 4

# custom function to create environment instances
def make_env():
    def _init():
        # Simulation initialization data
        arena_length = 15  # meters
        arena_width = 15   # meters
        num_sheep = NUM_SHEEP
        num_sheepdogs = NUM_SHEEPDOGS
        robot_wheel_radius = 0.1  # meters
        robot_distance_between_wheels = 0.2  # meters
        max_wheel_velocity = 10.0  # m/s

        # Create the environment
        env = HerdingSimEnv(arena_length, arena_width, num_sheep, num_sheepdogs, robot_distance_between_wheels, robot_wheel_radius, max_wheel_velocity, action_mode="multi", attraction_factor=0.0)

        return env
    return _init

if __name__ == "__main__":
    num_sims = 100
    env = make_env()()

    # Load model
    model_path = "/home/abhi2001/MSR/final_project/rl/eval_models/herding_multi_PPO-20241118-163030/model.zip"
    models = {}
    for i in range(NUM_SHEEPDOGS):
        models[i] = PPO.load(model_path, env=env)

    # Initialize metrics
    metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'successful_episodes': 0,
        'unsuccessful_episodes': 0,
        'success_rate': 0,
    }

    # Evaluate the model
    print(f"Starting evaluation of models")
    with torch.no_grad():
        for sim in tqdm.tqdm(range(num_sims)):
            observations = {}
            for i in range(NUM_SHEEPDOGS):
                observations[i], info = env.reset(robot_id=i)
            env.reset_frames()
            terminated = False
            truncated = False
            episode_reward = 0
            episode_length = 0
            while not terminated and not truncated:
                for i in range(NUM_SHEEPDOGS):
                    action, _ = models[i].predict(observations[i], deterministic=False)
                    observations[i], reward, terminated, truncated, _ = env.step(action, robot_id=i)
                episode_reward += reward
                episode_length += 1
                env.render(mode='human', fps=100)

            # Record metrics
            metrics['episode_rewards'].append(episode_reward)
            metrics['episode_lengths'].append(episode_length)
            if terminated:
                metrics['successful_episodes'] += 1
            elif truncated:
                metrics['unsuccessful_episodes'] += 1

        # Close the environment
        env.close()

    # Calculate success rate
    metrics['success_rate'] = metrics['successful_episodes'] / num_sims

    # Display metrics
    print(f"Model evaluation complete")
    print(f"Average episode reward: {np.mean(metrics['episode_rewards'])}")
    print(f"Average episode length: {np.mean(metrics['episode_lengths'])}")
    print(f"Average episode time: {np.mean(metrics['episode_lengths']) * 0.1} seconds")
    # print(f"Episode Lengths: {metrics['episode_lengths']}")
    print(f"Success rate: {metrics['success_rate']}")
    print(f"Number of successful episodes: {metrics['successful_episodes']}")
    print(f"Number of unsuccessful episodes: {metrics['unsuccessful_episodes']}")