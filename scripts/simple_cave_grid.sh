#!/bin/bash

#SBATCH --job-name=simple_cave_dqn

#SBATCH --output=simple_cave_%A_%a.out

#SBATCH --error=simple_cave_%A_%a.err

#SBATCH --partition=gpu_a100

#SBATCH --gpus-per-node=2

#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=18

#SBATCH --time=10:00:00

#SBATCH --array=0-7


# 1. Navigate to your project directory

cd /home/aszelestey/projects/2AMC15-Assignment-2


# 2. Clean environment and load required modules

module purge

module load 2023

module load Python/3.11.3-GCCcore-12.3.0


# 3. Activate the virtual environment

source .venv/bin/activate


# 4. Pick experiment config from the array task id

CONFIGS=(
  "dqn  sensors 0"
  "dqn  sensors 0.2"
  "dqn  no_sensors 0"
  "dqn  no_sensors 0.2"
  "ddqn sensors 0"
  "ddqn sensors 0.2"
  "ddqn no_sensors 0"
  "ddqn no_sensors 0.2"
)

read -r AGENT SENSORS SIGMA <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

SENSOR_FLAG=""
if [ "$SENSORS" = "no_sensors" ]; then
  SENSOR_FLAG="--no-sensors"
fi

SIGMA_FLAG=""
if [ "$SIGMA" != "0" ]; then
  SIGMA_FLAG="--sigma $SIGMA"
fi

OUT_DIR="results/simple_cave_${AGENT}_${SENSORS}_sigma${SIGMA}"


# 5. Run the training script

python train_deep.py \
  --agent "$AGENT" \
  --env continuous \
  --grid grid_configs/simple_cave_grid.npy \
  --wandb \
  --wandb-group simple_cave_grid \
  --out-dir "$OUT_DIR" \
  $SENSOR_FLAG \
  $SIGMA_FLAG
