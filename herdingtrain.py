from stable_baselines3 import PPO
import os
from herdingenv import HerdingSimEnv
import time
import wandb
from wandb.integration.sb3 import WandbCallback

# Initialize wandb for logging
name = "herding_test"
time_now = time.strftime("%Y%m%d-%H%M%S")
run = wandb.init(project='rl_herding', name=f"{name}-{time_now}" , sync_tensorboard=True, save_code=True)
# define custom metrics for logging
# wandb.define_metric("custom_step")
# wandb.define_metric("video", step_metric="custom_step")

# Create directories for saving models and logs
models_dir = f"models/herding_test/{name}-{time_now}"
logdir = f"logs/herding_test/{name}-{time_now}"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

# Simulation initialization data
arena_length = 15  # meters
arena_width = 15   # meters
num_sheep = 5
num_sheepdogs = 1
robot_wheel_radius = 0.1  # meters
robot_distance_between_wheels = 0.2  # meters
max_wheel_velocity = 8.0  # m/s

# Create the environment
env = HerdingSimEnv(arena_length, arena_width, num_sheep, num_sheepdogs, robot_distance_between_wheels, robot_wheel_radius, max_wheel_velocity)

# Initialize the model
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)
TIMESTEPS = 10000 # number of timesteps to train the model for before logging
# calculate iterations based on num_timesteps
iters = model.num_timesteps // TIMESTEPS
print(f"Starting from iteration {iters}")

# main training loop
while True:
    iters += 1
    # custom_step = TIMESTEPS*iters

    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, callback=WandbCallback(model_save_freq=TIMESTEPS, model_save_path=f"{models_dir}/{TIMESTEPS*iters}", verbose=1))

    # render a video of the trained model in action
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()

    # log the video
    video_frames = env.get_video_frames()
    wandb.log({"video": wandb.Video(video_frames, caption=f"Model at iteration {TIMESTEPS*iters}",format="mp4", fps=30)})
    print(f"Video logged at iteration {TIMESTEPS*iters}")
    env.reset()