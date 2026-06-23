#!/bin/bash
# Smoke-test wiring for all three experiment matrices (3 representative configs).

#SBATCH --job-name=smoke_experiments
#SBATCH --output=smoke_experiments_%A_%a.out
#SBATCH --error=smoke_experiments_%A_%a.err
#SBATCH --partition=gpu_mig
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --array=0-2

set -euo pipefail

cd /home/aszelestey/projects/2AMC15-Assignment-2

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source .venv/bin/activate

CONFIGS=(
  "experiment_1 baseline simple_cave_grid dqn 0 default 0.0 1"
  "experiment_2 no_sensors simple_cave_grid ddqn 0 no_sensors 0.0 1"
  "experiment_3 sigma05 A1_grid ddqn 1 default 0.5 10"
)

read -r EXP_LABEL VARIANT GRID AGENT SEED SENSORS SIGMA FINAL_EVAL_RUNS <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

EXTRA_FLAGS=()
if [ "$SENSORS" = "no_sensors" ]; then
  EXTRA_FLAGS+=(--no-sensors)
fi
if [ "$SIGMA" != "0.0" ]; then
  EXTRA_FLAGS+=(--sigma "$SIGMA")
fi

OUT_DIR="results/smoke/${EXP_LABEL}_${VARIANT}_${GRID}_${AGENT}_seed${SEED}"

uv run python train_deep.py \
  --agent "$AGENT" \
  --env continuous \
  --grid "grid_configs/${GRID}.npy" \
  --seed "$SEED" \
  --episodes 20 \
  --device cpu \
  --eval-interval 5 \
  --out-dir "$OUT_DIR" \
  --final-eval-runs "$FINAL_EVAL_RUNS" \
  "${EXTRA_FLAGS[@]}"
