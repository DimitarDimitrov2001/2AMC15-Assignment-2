# Reinforcement Learning in Practice

This repository contains the Assignment 2 deep reinforcement learning work for
cave exploration. The agent is trained in a continuous grid-world environment
with a discrete action space: it can rotate left, rotate right, or move
forward. The goal is to navigate cave-like layouts and reach a target while
dealing with sparse rewards, collisions, limited observations, and optional
action noise.

The main implementations are:

- DQN: a baseline Deep Q-Network with replay buffer and target network.
- DDQN: a Dueling Double DQN selected with `--agent ddqn`.

The report compares DQN and DDQN under different settings: normal LiDAR-like
distance sensors, no sensors, and stochastic actions. The report results are
included under `report_result/` and were generated from the experiment scripts
in `scripts/`.

## Setup

Requirements:

- Python 3.12 or newer
- `uv` for dependency and virtual environment management

From the repository root:

```powershell
uv sync
```

All commands below should be run from the folder that contains `train_deep.py`,
`pyproject.toml`, and `uv.lock`.

## Command-Line Interface

Use `train_deep.py` for both DQN and DDQN runs:

```powershell
uv run python train_deep.py --agent <dqn|ddqn> --grid <grid.npy> [options]
```

The default command trains DQN on `grid_configs/realistic_super_hard_cave.npy`
with sensors enabled and deterministic actions:

```powershell
uv run python train_deep.py
```

By default, runs write artifacts under `results/<agent>_<timestamp>/` unless an
explicit `--out-dir` is provided.

Most useful options:

- `--agent {dqn,ddqn}`: choose the learning architecture.
- `--grid PATH`: choose the cave grid.
- `--episodes N`: choose the training budget.
- `--no-sensors`: remove the LiDAR-like distance sensors.
- `--sigma X`: add stochasticity to the continuous environment actions.
- `--curiosity {grid-count,no}`: enable or disable count-based intrinsic reward.
- `--out-dir PATH`: choose where checkpoints, plots, and JSON files are saved.

For the complete list of parameters and short explanations:

```powershell
uv run python train_deep.py --help
```

## Experiment Grids

The report experiments use three cave layouts:

- `grid_configs/simple_cave_grid.npy` (easy)
- `grid_configs/big_spaces_cave.npy` (medium)
- `grid_configs/realistic_super_hard_cave.npy` (hard)

The current CLI default is:

```text
grid_configs/realistic_super_hard_cave.npy
```

## Default Hyperparameters

These are the current defaults used by `train_deep.py` unless overridden on the
command line.

| Setting | Default |
| --- | --- |
| Environment (`--env`) | `continuous` |
| Agent (`--agent`) | `dqn` |
| Grid (`--grid`) | `grid_configs/realistic_super_hard_cave.npy` |
| Episodes (`--episodes`) | `6000` |
| Max steps per episode (`--max-steps`) | `500` |
| Seed (`--seed`) | `0` |
| Device (`--device`) | `auto` |
| Step size (`--step-size`) | `0.5` |
| Action noise sigma (`--sigma`) | `0.0` |
| Sensors (`--no-sensors`) | enabled |
| Eval interval (`--eval-interval`) | `25` |
| Eval episodes (`--eval-episodes`) | `10` |
| Final eval runs (`--final-eval-runs`) | `10` |
| Log interval (`--log-interval`) | `1` |
| Epsilon schedule (`--epsilon-schedule`) | `linear_annealing` |
| Epsilon max / min (`--epsilon-max`, `--epsilon-min`) | `1.0` / `0.05` |
| Epsilon duration (`--epsilon-duration`) | `150000` steps |
| Epsilon start step (`--epsilon-start-step`) | `0` |
| Gamma (`--gamma`) | `0.99` |
| Learning rate (`--lr`) | `0.00025` |
| Batch size (`--batch-size`) | `64` |
| Replay capacity (`--replay-capacity`) | `100000` |
| Replay warmup (agent default, no CLI flag) | `10000` transitions |
| Observation stack size (`--stack-size`) | `1` |
| Online update frequency (`--update-freq`) | every `4` environment steps |
| Target network update frequency (`--target-update-freq`) | every `250` environment steps |
| Hidden layer width (agent default, no CLI flag) | `128` |
| Gradient clipping (`--grad-clip-norm`) | global norm `10.0` |
| Curiosity (`--curiosity`) | `grid-count` |
| Curiosity beta (`--curiosity-beta`) | `0.05` |
| Curiosity resolution (agent default, no CLI flag) | `1.0` world unit |
| Target reward (`--target-reward`) | `1.0` |
| Living penalty (`--living-penalty`) | `-0.001` |
| Collision penalty (`--collision-penalty`) | `-0.01` |

The continuous environment has three actions: rotate left, rotate right, and
move forward. With sensors enabled, the observation contains position, heading,
and eight LiDAR-like distance readings.

## Demonstration Commands

The following commands are meant to be copy-pasted from the repository root.
They use the current default hyperparameters unless an option is explicitly
shown. You can run as many of these cli scripts as you like,
or change other CLI parameters to test settings that were not part of the
report. Important to note is the first four scripts are with LiDAR sensor as this is the default.

DQN on the simple/small cave grid:

```powershell
uv run python train_deep.py `
  --agent dqn `
  --grid grid_configs/simple_cave_grid.npy `
  --out-dir results/demo/simple_cave_dqn
```

DDQN on the simple/small cave grid:

```powershell
uv run python train_deep.py `
  --agent ddqn `
  --grid grid_configs/simple_cave_grid.npy `
  --out-dir results/demo/simple_cave_ddqn
```

DQN on the big-spaces cave grid:

```powershell
uv run python train_deep.py `
  --agent dqn `
  --grid grid_configs/big_spaces_cave.npy `
  --out-dir results/demo/big_spaces_dqn
```

DDQN on the big-spaces cave grid:

```powershell
uv run python train_deep.py `
  --agent ddqn `
  --grid grid_configs/big_spaces_cave.npy `
  --out-dir results/demo/big_spaces_ddqn
```

DQN on the realistic super-hard cave grid:

```powershell
uv run python train_deep.py `
  --agent dqn `
  --grid grid_configs/realistic_super_hard_cave.npy `
  --out-dir results/demo/realistic_super_hard_dqn
```

DDQN on the realistic super-hard cave grid:

```powershell
uv run python train_deep.py `
  --agent ddqn `
  --grid grid_configs/realistic_super_hard_cave.npy `
  --out-dir results/demo/realistic_super_hard_ddqn
```

DQN without LiDAR-like sensors:

```powershell
uv run python train_deep.py `
  --agent dqn `
  --grid grid_configs/realistic_super_hard_cave.npy `
  --no-sensors `
  --out-dir results/demo/realistic_dqn_no_sensors
```

DDQN without LiDAR-like sensors:

```powershell
uv run python train_deep.py `
  --agent ddqn `
  --grid grid_configs/realistic_super_hard_cave.npy `
  --no-sensors `
  --out-dir results/demo/realistic_ddqn_no_sensors
```

DQN with stochastic actions:

```powershell
uv run python train_deep.py `
  --agent dqn `
  --grid grid_configs/realistic_super_hard_cave.npy `
  --sigma 0.5 `
  --out-dir results/demo/realistic_dqn_sigma05
```

DDQN with stochastic actions:

```powershell
uv run python train_deep.py `
  --agent ddqn `
  --grid grid_configs/realistic_super_hard_cave.npy `
  --sigma 0.5 `
  --out-dir results/demo/realistic_ddqn_sigma05
```

DDQN without intrinsic motivation:

```powershell
uv run python train_deep.py `
  --agent ddqn `
  --curiosity no `
  --out-dir results/demo/ddqn_no_curiosity
```

Short smoke test:

```powershell
uv run python train_deep.py `
  --agent dqn `
  --episodes 20 `
  --eval-interval 5 `
  --eval-episodes 2 `
  --final-eval-runs 1 `
  --out-dir results/demo/smoke_dqn
```

## Outputs

Each training run writes the following artifacts under `--out-dir`:

- `best.pt`: checkpoint with the best evaluation mean reward.
- `last.pt`: final checkpoint.
- `history.json`: per-episode training, evaluation, and agent metrics.
- `config.json`: resolved CLI and trainer configuration.
- `training_curves.png`: reward, evaluation, success-rate, and TD-loss plots.
- `evaluation_summary.txt`: compact text summary of final metrics.
- `policy_rollout.json` and `policy_rollout.png`: greedy rollout artifacts from
  the best checkpoint when a checkpoint exists.

When `--wandb` is enabled, run metrics and selected artifacts are also logged to
Weights & Biases.

## Report Experiments

The actual report experiments are the three numbered SLURM scripts:

- `scripts/experiment_1.sh`: baseline hyperparameters with sensors and deterministic actions.
- `scripts/experiment_2.sh`: no-sensor ablation with `--no-sensors`.
- `scripts/experiment_3.sh`: stochastic-action setting with `--sigma 0.5`.

Note that each experiment is applied on each implemented agent (DQN, DDQN) + on
the 3 created cave grids.

The other shell scripts were mainly used for smoke tests and wiring checks.
The numbered experiment scripts were launched on Snellius with SLURM to compute
the full matrix faster than local sequential runs. The generated outputs are
included in `report_result/`.

Technically, the experiments can also be replicated with `train_deep.py` by
running every combination of:

- agents: `dqn`, `ddqn`
- grids: `simple_cave_grid`, `big_spaces_cave`, `realistic_super_hard_cave`
- seeds: `0`, `1`, `2`, `3`, `4`
- settings: baseline, `--no-sensors`, and `--sigma 0.5`

That is a large number of long runs, which is why Snellius was used. The helper
scripts `scripts/merge_experiments.py` and `scripts/create_final_artifacts.py`
were then used to produce the report tables and plots from the completed run
folders.

## Codebase Overview

```text
.
|-- agents/                         # DQN, DDQN, replay buffer, schedules
|   |-- dqn_agent.py                 # Standard DQN implementation
|   |-- ddqn_agent.py                # DDQN agent using the dueling network
|   |-- dueling_dqn/                 # Dueling Q-network module
|   |-- replay_buffer.py             # Fixed-size experience replay buffer
|   |-- curiosity.py                 # Count-based intrinsic reward
|   `-- defaults.py                  # Agent hyperparameter defaults
|-- training/                        # Generic training loop and config
|-- world/                           # Environments and grid editor
|   |-- continuous_environment.py     # Continuous cave environment used here
|-- utils/                           # Plotting, artifact writing, logging helpers
|-- scripts/                         # Experiment scripts, smoke tests, result merging
|-- report_result/                   # Generated report outputs
|-- grid_configs/                    # Saved cave grids
|-- train_deep.py                    # Main CLI for DQN/DDQN training
|-- visualize_random_agent.py        # Rollout visualization helper
|-- pyproject.toml                   # Project dependencies
`-- uv.lock                          # Locked dependency versions
```

## Environment and Reward

The task starts from a grid map, but training uses
`world/continuous_environment.py`: the grid is interpreted as a continuous
2D space where the agent has a continuous position and heading while still
choosing from a small discrete action set. The grid cells define walls,
obstacles, empty space, and the target.

The grid encoding is:

- `0`: empty cell
- `1`: boundary wall
- `2`: obstacle
- `3`: target

The default reward values are:

- target reached: `+1.0`
- collision with boundary wall or obstacle: `-0.01`
- otherwise: `-0.001`

Wall and obstacle bumps keep the robot in place and receive the collision
penalty.

## Grids

Grid files are stored as NumPy arrays in `grid_configs/`.

To create or edit grids, run:

```powershell
uv run python world/grid_creator.py
```

Then open the local URL printed by Flask, usually:

```text
http://127.0.0.1:5000
```
