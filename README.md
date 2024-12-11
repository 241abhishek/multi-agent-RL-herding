# Multi-Agent Reinforcement Learning for Shepherding 

## Overview
This project implements a multi-agent reinforcement learning (RL) approach to the shepherding task using Stable Baselines3 (SB3) and a custom herding environment. The simulation allows multiple shepherds to collaborate in guiding sheep to the goal region. 

<div align="center">
  <video src=https://private-user-images.githubusercontent.com/72541517/394943338-2496d16f-d17e-4756-b68d-986aa5dc873f.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzM5NTczMjYsIm5iZiI6MTczMzk1NzAyNiwicGF0aCI6Ii83MjU0MTUxNy8zOTQ5NDMzMzgtMjQ5NmQxNmYtZDE3ZS00NzU2LWI2OGQtOTg2YWE1ZGM4NzNmLm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDEyMTElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMjExVDIyNDM0NlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWFmNjA3ZDNhNzJhZTRhNzk5MTI2ZjRkZjA2YjI1M2MzNGVmMzIzYjVmNzVjODlhZGY4MTc4MGEwMDEyYzMyMTEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.CVhSpm7rBkgXiI3n8NJdiOLaL46ZtV_fXX4Rzwjs09E />
</div>

## Installation

1. Create a Python virtual environment:
```bash
python -m venv venv 
source venv/bin/activate
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Troubleshooting: If you face numpy issues, reinstall and try again.
```bash
pip install --force-reinstall numpy
```

## Project Structure
- [`multi_model_eval.py`](multi_model_eval.py): Evaluation script for trained RL models
- [`herdingtrain_multi.py`](herdingtrain_multi.py): Training script for multi-agent shepherding models
- [`herdingenv.py`](herdingenv.py): Custom gymnasium environment for sheep herding simulation
- [`herdingrobot.py`](herdingrobot.py): Dynamics definition for the differential drive robots used in the simulation. 

## Evaluation Script Usage

### Command Line Arguments
- `--num_sheep`: Number of sheep in the simulation (default: 1)
- `--num_shepherds`: Number of shepherds in the simulation (default: 1)
- `--model_path`: Path to the trained RL model (required)
- `--save_video`: Save simulation videos (True/False, default: False)
- `--num_sims`: Number of simulations to run (default: 10)
- `--render_mode`: Rendering mode, options: "human" or "offscreen"

### Example Evaluation Command
```bash
python3 multi_model_eval.py --num_sheep 10 --num_shepherds 4 --model_path trained_models/model.zip --num_sims 5 --render_mode human
```

### Example Training Command
```bash
python3 herdingtrain_multi.py
```

Look at [`herdingtrain_multi.py`](herdingtrain_multi.py) to understand and modify training configurations. The script uses Vectorized Environments to run mutiple instances of the agent simultaneosly. This increases the data collection rate speeding up the training process. 

### Control Modes for the Shepherd in the Simulation
The simulation provides four distinct control modes for shepherds:
- Wheel Mode: Directly control the left and right wheel velocities, offering granular control over the robot’s movement.
- Vector Mode: Specify a heading vector and speed for shepherds, simplifying control while still adhering to the physics of differential drive systems.
- Point Mode: Specify a destination point on the map, and the simulation calculates appropriate wheel velocities to navigate the shepherd toward the target, abstracting low-level control.
- Multi Mode: This mode is used when evaluating a model trained for a single shepherd-single sheep on multiple agents using the attention swithing mehanism. See [website post](https://abhishek-sankar.com/multi-agent-rl/) for more info.