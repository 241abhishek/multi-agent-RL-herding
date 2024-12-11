import argparse
import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecNormalize
from herdingenv import HerdingSimEnv
import tqdm
import time

def make_env(arena_length, arena_width, num_sheep, num_shepherds, robot_distance_between_wheels, robot_wheel_radius, max_wheel_velocity):
    def _init():
        env = HerdingSimEnv(
            arena_length=arena_length,
            arena_width=arena_width,
            num_sheep=num_sheep,
            num_sheepdogs=num_shepherds,
            robot_distance_between_wheels=robot_distance_between_wheels,
            robot_wheel_radius=robot_wheel_radius,
            max_wheel_velocity=max_wheel_velocity,
            action_mode="multi",
            attraction_factor=0.0
        )
        return env
    return _init

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained RL model for the herding task.")
    parser.add_argument("--num_sheep", type=int, default=1, help="Number of sheep in the simulation.")
    parser.add_argument("--num_shepherds", type=int, default=1, help="Number of shepherds in the simulation.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained RL model.")
    parser.add_argument("--save_video", type=str, default="False", help="Save videos of simulations (True/False).")
    parser.add_argument("--num_sims", type=int, default=10, help="Number of simulations to run.")
    parser.add_argument("--render_mode", type=str, default="human", choices=["human", "offscreen"], help="Render mode for the environment.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # Convert save_video to boolean
    save_video = args.save_video.lower() == "true" 

    # create a video directory if it does not exist
    if save_video:
        import os
        if not os.path.exists("videos"):
            os.makedirs("videos")

    # Environment parameters
    arena_length = 15
    arena_width = 15
    robot_wheel_radius = 0.1
    robot_distance_between_wheels = 0.2
    max_wheel_velocity = 10.0

    env = make_env(
        arena_length, arena_width, args.num_sheep, args.num_shepherds, robot_distance_between_wheels, robot_wheel_radius, max_wheel_velocity
    )()

    # Load model
    models = {}
    for i in range(args.num_shepherds):
        models[i] = PPO.load(args.model_path, env=env, device='cpu')

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
        for sim in tqdm.tqdm(range(args.num_sims)):
            observations = {}
            for i in range(args.num_shepherds):
                observations[i], info = env.reset(robot_id=i)
            terminated = False
            truncated = False
            episode_reward = 0
            episode_length = 0

            while not terminated and not truncated:
                for i in range(args.num_shepherds):
                    action, _ = models[i].predict(observations[i], deterministic=False)
                    observations[i], reward, terminated, truncated, _ = env.step(action, robot_id=i)
                episode_reward += reward
                episode_length += 1

                if args.render_mode == "human":
                    env.render(mode="human", fps=60)
                elif args.render_mode == "offscreen":
                    env.render()

            # Save video if specified
            if save_video:
                print(f"Saving video for simulation {sim}")
                env.save_video(f"videos/simulation_{sim}.mp4", fps=60)
            env.reset_frames()

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
    metrics['success_rate'] = metrics['successful_episodes'] / args.num_sims

    # Display metrics
    print(f"Model evaluation complete")
    print(f"Average episode reward: {np.mean(metrics['episode_rewards'])}")
    print(f"Average episode length: {np.mean(metrics['episode_lengths'])}")
    print(f"Average episode time: {np.mean(metrics['episode_lengths']) * 0.1} seconds")
    print(f"Success rate: {metrics['success_rate']}")
    print(f"Number of successful episodes: {metrics['successful_episodes']}")
    print(f"Number of unsuccessful episodes: {metrics['unsuccessful_episodes']}")
