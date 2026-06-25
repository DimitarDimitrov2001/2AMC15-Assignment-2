#!/bin/bash

#SBATCH --job-name=RL_train

#SBATCH --output=RL_train_%j.out

#SBATCH --error=RL_train_%j.err

#SBATCH --partition=gpu_a100

#SBATCH --gpus-per-node=2

#SBATCH --nodes=1                  # Run all processes on a single node	

#SBATCH --ntasks=1                 # Run a single task (your python script)

#SBATCH --cpus-per-task=18         # Allocate 64 CPU cores (half a Rome node)

#SBATCH --time=10:00:00           # Maximum allowed wall time (5 Days)



# 1. Navigate to your project directory

cd /home/aszelestey/projects/2AMC15-Assignment-2



# 2. Clean environment and load required modules

module purge

module load 2023

module load Python/3.11.3-GCCcore-12.3.0



# 3. Activate the virtual environment

source .venv/bin/activate



# 4. Run the training script

python train_deep.py --agent dqn --env continuous --grid grid_configs/A1_grid.npy --wandb --log-interval 10 --episodes 10000 --device cuda --max-steps 1000
