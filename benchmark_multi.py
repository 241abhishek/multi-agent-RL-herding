from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from herdingenv import HerdingSimEnv
import time
import numpy as np
import os

# custom function to create environment instances
def make_env(rank):
    def _init():
        # Simulation initialization data
        arena_length = 15  # meters
        arena_width = 15   # meters
        num_sheep = 1
        num_sheepdogs = 1
        robot_wheel_radius = 0.1  # meters
        robot_distance_between_wheels = 0.2  # meters
        max_wheel_velocity = 8.0  # m/s

        robots = [
            [2.0, 2.0, np.pi/4],  # sheep-dog 1
            [5.0, 5.0, np.pi/4],  # sheep 1
        ]

        # Create the environment
        env = HerdingSimEnv(arena_length, arena_width, num_sheep, num_sheepdogs, robot_distance_between_wheels, robot_wheel_radius, max_wheel_velocity, robots)

        # print(f"Environment initialized in process ID: {os.getpid()}")

        return env
    return _init
    

# function to benchmark training time
def benchmark_vec_env(env_type, num_envs=4, total_timesteps=10000):
    """
    Benchmark function to measure time taken for training with PPO in a vectorized environment.
    
    :param env_type: Vectorized environment type, either "dummy" or "subproc".
    :param num_envs: Number of parallel environments.
    :param total_timesteps: Number of timesteps to train for.
    """
    # Create vectorized environment based on type
    if env_type == "dummy":
        env = DummyVecEnv([make_env(i) for i in range(num_envs)])
        env = VecMonitor(env)
    elif env_type == "subproc":
        env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
        env = VecMonitor(env)
    else:
        raise ValueError("Invalid environment type. Choose either 'dummy' or 'subproc'.")

    # Initialize PPO model
    model = PPO("MlpPolicy", env, device="cpu", n_steps = 5000, verbose=0)
    # model = PPO("MlpPolicy", env, n_steps=10000, verbose=0)

    # Time the training
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps)
    end_time = time.time()

    # Clean up
    env.close()
    
    # Calculate time elapsed
    time_elapsed = end_time - start_time
    print(f"Training time with {env_type.capitalize()}VecEnv: {time_elapsed:.2f} seconds")
    return time_elapsed

if __name__ == "__main__":

    # Define benchmark parameters
    for num_envs in range(20, 21, 4):

        print(f"\nBenchmarking with {num_envs} environments...")
        # total_timesteps = 150000
        total_timesteps = 200000 

        # Benchmark training time with SubprocVecEnv
        print("Benchmarking training time with SubprocVecEnv...")
        time_subproc = benchmark_vec_env("subproc", num_envs=num_envs, total_timesteps=total_timesteps)

        # Benchmark training time with DummyVecEnv
        print("Benchmarking training time with DummyVecEnv...")
        time_dummy = benchmark_vec_env("dummy", num_envs=num_envs, total_timesteps=total_timesteps)