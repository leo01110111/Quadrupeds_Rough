#!/bin/bash

# Initialize Conda (adjust path if necessary)
source /home/leo/miniconda3/etc/profile.d/conda.sh

# Activate the desired environment
conda activate env_isaaclab

# Now you can run commands within the activated environment
python python scripts/rsl_rl/play.py --task Isaac-Velocity-Flat-Unitree-Go2-Play-v0 --num_envs 1

# Deactivate the environment when done (optional)
conda deactivates