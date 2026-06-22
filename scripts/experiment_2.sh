#!/bin/bash
# Sensor ablation: 5 seeds × 3 grids × 2 agents × 1 sensor mode = 30 runs (array 0–79).

#SBATCH --job-name=exp2_sensors
#SBATCH --output=experiment_2_%A_%a.out
#SBATCH --error=experiment_2_%A_%a.err
#SBATCH --partition=gpu_mig
#SBATCH --reservation=terv92681
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --array=0-79

set -euo pipefail

cd /home/aszelestey/projects/2AMC15-Assignment-2

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source .venv/bin/activate

SEEDS=(0 1 2 3 4)
GRIDS=(simple_cave_grid big_spaces_cave realistic_super_hard_cave)
AGENTS=(dqn ddqn)
SENSOR_MODES=(no_sensors)

task_id=$SLURM_ARRAY_TASK_ID
sensor_idx=$(( task_id % 2 ))
task_id=$(( task_id / 2 ))
agent_idx=$(( task_id % 2 ))
task_id=$(( task_id / 2 ))
grid_idx=$(( task_id % 4 ))
task_id=$(( task_id / 4 ))
seed_idx=$(( task_id % 5 ))

SEED=${SEEDS[$seed_idx]}
GRID=${GRIDS[$grid_idx]}
AGENT=${AGENTS[$agent_idx]}
SENSORS=${SENSOR_MODES[$sensor_idx]}

SENSOR_FLAG=""
if [ "$SENSORS" = "no_sensors" ]; then
  SENSOR_FLAG="--no-sensors"
fi

OUT_DIR="results/experiment_2/${GRID}_${AGENT}_${SENSORS}_seed${SEED}"

uv run python train_deep.py \
  --agent "$AGENT" \
  --env continuous \
  --grid "grid_configs/${GRID}.npy" \
  --seed "$SEED" \
  --episodes 5000 \
  --wandb \
  --wandb-group experiment_2 \
  --out-dir "$OUT_DIR" \
  --final-eval-runs 1 \
  $SENSOR_FLAG
