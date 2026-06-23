#!/bin/bash
# Stochasticity sweep: 5 seeds × 4 grids × 2 agents, sigma 0.5 only = 40 runs (array 0–39).

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
#SBATCH --array=0-39

set -euo pipefail

cd /home/aszelestey/projects/2AMC15-Assignment-2

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source .venv/bin/activate

SEEDS=(0 1 2 3 4)
GRIDS=(simple_cave_grid A1_grid big_spaces_cave realistic_super_hard_cave)
AGENTS=(dqn ddqn)
SIGMA=0.5

task_id=$SLURM_ARRAY_TASK_ID
agent_idx=$(( task_id % 2 ))
task_id=$(( task_id / 2 ))
grid_idx=$(( task_id % 4 ))
task_id=$(( task_id / 4 ))
seed_idx=$(( task_id % 5 ))

SEED=${SEEDS[$seed_idx]}
GRID=${GRIDS[$grid_idx]}
AGENT=${AGENTS[$agent_idx]}

OUT_DIR="results/experiment_3/${GRID}_${AGENT}_sigma${SIGMA}_seed${SEED}"

uv run python train_deep.py \
  --agent "$AGENT" \
  --env continuous \
  --grid "grid_configs/${GRID}.npy" \
  --seed "$SEED" \
  --episodes 6000 \
  --device cpu \
  --wandb \
  --wandb-group experiment_3 \
  --out-dir "$OUT_DIR" \
  --final-eval-runs 10 \
  --sigma "$SIGMA"
