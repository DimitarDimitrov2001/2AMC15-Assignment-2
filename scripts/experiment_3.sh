#!/bin/bash
# Stochasticity sweep: 5 seeds × 4 grids × 2 agents × 3 sigma values = 120 runs (array 0–119).

#SBATCH --job-name=exp3_sigma
#SBATCH --output=experiment_3_%A_%a.out
#SBATCH --error=experiment_3_%A_%a.err
#SBATCH --partition=gpu_mig
#SBATCH --reservation=terv92681
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --array=0-119

set -euo pipefail

cd /home/aszelestey/projects/2AMC15-Assignment-2

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source .venv/bin/activate

SEEDS=(0 1 2 3 4)
GRIDS=(simple_cave_grid A1_grid big_spaces_cave realistic_super_hard_cave)
AGENTS=(dqn ddqn)
SIGMAS=(0.0 0.2 0.5)

task_id=$SLURM_ARRAY_TASK_ID
sigma_idx=$(( task_id % 3 ))
task_id=$(( task_id / 3 ))
agent_idx=$(( task_id % 2 ))
task_id=$(( task_id / 2 ))
grid_idx=$(( task_id % 4 ))
task_id=$(( task_id / 4 ))
seed_idx=$(( task_id % 5 ))

SEED=${SEEDS[$seed_idx]}
GRID=${GRIDS[$grid_idx]}
AGENT=${AGENTS[$agent_idx]}
SIGMA=${SIGMAS[$sigma_idx]}

SIGMA_FLAG=""
if [ "$SIGMA" != "0.0" ]; then
  SIGMA_FLAG="--sigma $SIGMA"
fi

OUT_DIR="results/experiment_3/${GRID}_${AGENT}_sigma${SIGMA}_seed${SEED}"

uv run python train_deep.py \
  --agent "$AGENT" \
  --env continuous \
  --grid "grid_configs/${GRID}.npy" \
  --seed "$SEED" \
  --episodes 10000 \
  --wandb \
  --wandb-group experiment_3 \
  --out-dir "$OUT_DIR" \
  --final-eval-runs 10 \
  $SIGMA_FLAG
