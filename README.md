# Reinforcement Learning in Practice

This repository contains the Assignment-2 deep-RL stack: two grid-world environments (continuous and minimal), a DQN agent with intrinsic motivation, and an algorithm-agnostic training pipeline.

Implemented agents:

- Random baseline
- DQN (Deep Q-Network) with optional count-based intrinsic motivation
- A3C (Asynchronous Advantage Actor-Critic) with multi-process actor-learners

## Setup

Requirements:

- Python 3.12 or newer
- `uv` for dependency and virtual environment management

```powershell
uv sync
```

All commands below should be run from the repository root, the folder that contains `train_deep.py`, `pyproject.toml`, and `uv.lock`.

## Deep-RL Training With `train_deep.py`

`train_deep.py` is the main training entry point for the deep-RL stack: the continuous/minimal environments and the algorithm-agnostic `Trainer` in `training/`. The random baseline, DQN, Dueling-DQN, and A3C agents are wired today. Further learning agents (PPO, ...) plug in through the `_build_agent` factory without changing the training loop.

General form:

```powershell
uv run python train_deep.py [options]
```

Example:

```powershell
uv run python train_deep.py
```

Options:

Environment and training:

- `--env {minimal,continuous}`: environment to train on (default `continuous`).
- `--no-sensors`: continuous env only. Drop the 8 distance sensors from the observation, leaving the bare `(x, y, theta)` state (state dim `11 → 3`). Sensors are included by default.
- `--sigma`: continuous env only. Action noise std-dev (`action_sigma`); default `0.0` keeps actions deterministic.
- `--step-size`: override the env move/step size (defaults: continuous `0.5`, minimal `0.5`). Larger values cross the map in fewer steps, so the goal is reachable in a shorter horizon and episodes run faster.
- `--agent {random,dqn,dueling-dqn,ddqn,a3c}`: agent to train (default `dqn`; `ddqn` is an alias for `dueling-dqn`).
- `--grid`: path to a `.npy` grid file (default `grid_configs/A1_grid.npy`).
- `--start-pos X Y`: fixed continuous (x, y) start for evaluation and visualization (and for training unless `--exploring-starts` is set). Omit to use the grid's START_CELL, falling back to a random empty cell each episode.
- `--eval-starting-pos X Y`: fixed continuous (x, y) start used only for evaluation and visualization. Overrides `--start-pos` for the eval/viz env while leaving the training start unchanged. Omit to fall back to `--start-pos`.
- `--exploring-starts`: use random start positions during training (exploring starts) while evaluation/visualization keep the fixed `--start-pos` / `--eval-starting-pos`. Helps discovery on sparse-reward grids by seeding the replay buffer with goal-reaching transitions from varied starts.
- `--episodes`, `--max-steps`, `--seed`: training budget and seed (default `3000` episodes, `500` max steps per episode, seed `0`).
- `--device {auto,cpu,cuda,mps}`: compute device for learning agents (default `auto`, which picks cuda > mps > cpu). For the small grid MLP, `cpu` is often fastest.
- `--eval-interval`, `--eval-episodes`: evaluation cadence and rollouts per evaluation (defaults `25` and `10`).
- `--final-eval-runs`: number of greedy evaluation rollouts from the best checkpoint after training, each with a distinct seed (default `1`). When greater than `1`, `policy_rollout.json` is a list and the combined `.png`/`.html` overlay all runs.
- `--log-interval`: print window-averaged metrics every N episodes (default `25`, same as `--eval-interval`). Terminal lines show mean reward, length, termination rate, TD loss, Q-value, and epsilon over the last `log_interval` episodes; W&B receives per-episode metrics under grouped keys (`rollout/*`, `losses/*`, `qvals/*`, `charts/*`, `eval/*`).
- `--out-dir`: output directory for checkpoints and `history.json` (default `results/<agent>_<timestamp>`).
- `--wandb`, `--wandb-group`: enable Weights & Biases logging and optionally bucket runs under a group name.
- `--no-visualize`, `--wandb-visualisations`, `--viz-max-steps`: with `--wandb`, a greedy rollout of the **best-so-far** checkpoint is rendered every `--wandb-visualisations` episodes (default `100`) and logged to W&B as `viz/rollout` (no local PNG is kept). `--no-visualize` disables these in-training W&B images. `--viz-max-steps` caps rollout length for both in-training W&B images and the post-training final evaluation.

DQN exploration (`--agent dqn` or `--agent dueling-dqn`):

- `--epsilon-max`: start epsilon for linear annealing, or fixed rate when `--epsilon-duration 0` (default `1.0`). Eval always uses greedy action selection regardless of this value.
- `--epsilon-min`: minimum epsilon after annealing (default `0.05`).
- `--epsilon-duration`: number of steps to anneal epsilon over (default `150000`; `0` keeps epsilon fixed at `--epsilon-max`).
- `--epsilon-start-step`: steps before epsilon annealing starts (default `0`).

DQN hyperparameters (`--agent dqn` or `--agent dueling-dqn`):

- `--gamma`: discount factor (default `0.99`).
- `--lr`: learning rate (default `1e-3`).
- `--batch-size`: batch size (default `64`).
- `--stack-size`: number of consecutive observations stacked into the state (default `1`).
- `--update-freq`: update the online network every N environment steps (default `1`).
- `--target-update-freq`: sync the target network every N environment steps (default `500`).
- `--replay-capacity`: replay buffer capacity in transitions (default `100000`, `agents.defaults.REPLAY_DEFAULT_CAPACITY`).
- `--reward-clip`: symmetric clip on the extrinsic reward before the intrinsic bonus is added (default `1.0`; `<=0` disables).
- `--grad-clip-norm`: max global gradient norm for clipping (default `10.0`; `<=0` only measures the norm without clipping).

The online network is a two-hidden-layer MLP (width `128`, `agents.defaults.DQN_N_HIDDEN_NODES`) trained with the Huber (SmoothL1) loss. `--agent dueling-dqn` swaps that MLP for a dueling architecture and uses Double DQN targets. Both DQN variants normalize observations in `observe()` using the environment's `observation_high` and `angular_dims`, then store the normalized states in replay. The replay buffer holds transitions in numpy ring buffers and returns sampled minibatches as torch tensors already placed on the agent's device. Learning starts once the buffer holds `agents.defaults.REPLAY_DEFAULT_START_SIZE` (`10000`) transitions.

Intrinsic motivation (`--agent dqn`, `--agent dueling-dqn`, or `--agent a3c`):

- `--curiosity {no,grid_count,grid-count}`: intrinsic exploration bonus (default `no`). `grid_count` / `grid-count` adds a count-based bonus (`beta / sqrt(visit_count)`) per discretized cell.
- `--curiosity-beta`: scale for the curiosity bonus (default `0.1`).
- Curiosity counting uses a fixed resolution of `1.0` world units (`agents.defaults.CURIOSITY_RESOLUTION_DEFAULT`), independent of `--step-size`. A3C keeps a separate visit table per worker, avoiding target-coordinate shaping while still discouraging repeated local loops.

A3C hyperparameters (`--agent a3c` only):

- A3C runs multiple asynchronous actor-learner processes that share a single network in CPU shared memory and push gradients Hogwild-style. The device is forced to `cpu` regardless of `--device` because shared-memory multiprocessing requires CPU tensors. Exploration uses the stochastic softmax policy, an entropy bonus, the explicit random-action schedule below, and optional count-based curiosity.
- `--a3c-workers`: number of asynchronous actor-learner processes (default `4`).
- `--a3c-lr`: A3C optimizer learning rate (default `1e-4`).
- `--a3c-t-max`: max rollout length between gradient pushes / n-step return horizon (default `5`).
- `--a3c-entropy-beta`: entropy regularization coefficient, uniform across workers (default `0.05`).
- `--a3c-random-action-start`, `--a3c-random-action-final`, `--a3c-random-action-decay-steps`: epsilon-soft exploration schedule (defaults `0.40 → 0.10` over `1000000` env steps). A3C samples from `(1 - eps) * policy + eps * uniform`, so exploratory actions remain part of the actor objective instead of being discarded.
- `--a3c-progress-reward-scale`: training-only reward for reducing distance to the target (default `0.0`, disabled). This is an explicit diagnostic knob rather than the default learning signal. The unshaped environment return is still logged as `rollout/episode_reward`; the reward used for A3C updates is logged separately as `rollout/shaped_reward`.
- `--a3c-value-coef`: weight on the value loss (default `0.25`).
- `--a3c-total-steps`: global environment-step budget across all workers (defaults to `--episodes * --max-steps`).
- `--gamma` is reused (default `0.99`).

A3C uses Huber critic loss and clips bootstrapped value targets to the bounded-return scale (`agents.defaults.A3C_VALUE_TARGET_CLIP`, default `100.0`) so the critic cannot amplify its own overestimates during sparse-reward exploration.

Example:

```powershell
uv run python train_deep.py --env continuous --agent a3c --a3c-workers 8 --max-steps 200 --a3c-lr 1e-4
```

The `Trainer` (`training/trainer.py`) is algorithm-agnostic and supports an optional environment-step budget (`max_env_steps`), per-episode mean of agent update metrics, best/last checkpointing via `BaseAgent.save_checkpoint`/`load_checkpoint`, and history-to-disk. Agents that own their training loop (those with `BaseAgent.trains_externally = True`, like A3C) are detected by the Trainer, which then delegates rollout generation to the agent's `train_iter()` while still owning evaluation, logging, checkpointing, and W&B on the shared global network. Both new environments accept an optional per-episode `seed` in `reset(seed=...)`. The continuous environment exposes `observation_high` per-dimension upper bounds (grid size for x/y, 360° for theta, `max_sensor_range` per sensor when sensors are enabled) and `angular_dims` (the periodic observation indices, e.g. theta); the DQN agent wraps the angular dims by their period, scales every observation by `observation_high`, and clips to `[0, 1]`.

## Outputs

`train_deep.py` writes the following artifacts under `--out-dir`, or under `results/<agent>_<timestamp>` by default:

- `best.pt` / `last.pt`: network checkpoints (best `eval/mean_reward` and final episode).
- `history.json`: per-episode metric history (the single source for tabular metrics). Each row includes rollout keys such as `rollout/episode_reward`, `rollout/episode_length`, `rollout/success`, and `rollout/collisions`, plus agent update metrics (`losses/*`, `qvals/*`, `dqn/*`), and eval keys (`eval/mean_reward`, `eval/success_rate`, …) on eval intervals.
- `config.json`: resolved CLI/trainer configuration for the run.
- `training_curves.png`: reward, eval, success-rate, and TD-loss curves.
- `policy_rollout.json` / `policy_rollout.png` / `policy_rollout.html`: greedy rollout(s) from the best checkpoint after training (when one exists). A single run writes one object; `--final-eval-runs N` with `N > 1` writes a JSON list and combined multi-path visualizations.

In-training rollout images are logged to W&B only (via `--wandb-visualisations`); no local `rollouts/` directory is created.

When `--wandb` is enabled, the post-training files, including `policy_rollout.html`, are also logged as a W&B artifact.

## Project Structure

```text
.
|-- agents/
|   |-- base_agent.py              # BaseAgent interface
|   |-- dqn_agent.py               # DQN agent with replay buffer
|   |-- a3c_agent.py               # A3C agent with multi-process actor-learners
|   |-- random_agent.py            # Uniform-random baseline
|   |-- null_agent.py              # No-op agent (testing)
|   |-- curiosity.py               # Count-based intrinsic motivation
|   |-- epsilon_schedules.py       # Constant and linear epsilon annealing
|   |-- replay_buffer.py           # Experience replay buffer
|   |-- learning_rates.py          # Learning-rate schedule implementations
|   `-- defaults.py                # Shared hyperparameter constants
|-- docs/                          # Extra documentation and examples
|-- grid_configs/                  # Saved NumPy grid files
|-- training/
|   |-- trainer.py                 # Algorithm-agnostic training loop
|   |-- config.py                  # TrainerConfig dataclass
|   `-- defaults.py                # Trainer default constants
|-- utils/                         # Evaluation, plotting, logging, artifacts
|-- world/
|   |-- environment_base.py        # BaseGridEnvironment: shared episode scaffolding
|   |-- minimal_environment.py     # Point-mass (x, y) environment
|   `-- continuous_environment.py  # Robot with heading + distance sensors
|-- train_deep.py                  # Main training CLI
|-- visualize_random_agent.py      # Rollout visualization helper
|-- pyproject.toml                 # Project dependencies and Python requirement
`-- uv.lock                        # Locked dependency versions
```

## Environment and Reward

The environment is a grid world with encoded cells:

- `0`: empty cell
- `1`: boundary wall
- `2`: obstacle
- `3`: target

The default reward function is defined in `world/environment_base.py` (`default_reward`):

- target reached: `+1.0`
- collision with boundary wall or obstacle: `-1.0`
- otherwise (living penalty): `-0.01`

A custom `reward_fn` can be passed to either environment to override these defaults. Wall and obstacle bumps keep the robot in place and receive the collision penalty.

## Visualization

Grid renderers (`visualize_random_agent.py`, `utils/rl_plots.py`, and the interactive
`policy_rollout.html` viewer) use a **top-left origin**: `(0, 0)` is the top-left
cell, `x` increases to the right, and `y` increases downward, matching NumPy
`grid[row, col]` indexing.

## SLURM experiment matrix

Three array scripts under `scripts/` launch the 240-run deep-RL matrix on
`gpu_mig` (reservation `terv92681`): 10000 episodes, `--env continuous`, `--wandb`,
seeds `0–4`, grids `simple_cave_grid`, `A1_grid`, `big_spaces_cave`,
`realistic_super_hard_cave`, agents `dqn` and `ddqn`.

| Script | Array | Runs | Variants | `--final-eval-runs` |
| --- | --- | --- | --- | --- |
| `scripts/experiment_1.sh` | `0–39` | 40 | baseline (default sensors, σ=0) | 1 |
| `scripts/experiment_2.sh` | `0–79` | 80 | `--no-sensors` vs default sensors | 1 |
| `scripts/experiment_3.sh` | `0–119` | 120 | `--sigma` 0.0 / 0.2 / 0.5 | 10 |

Outputs land under `results/experiment_<n>/<grid>_<agent>_…_seed<seed>/`.
Validate wiring first with `scripts/smoke_experiments.sh` (array `0–5`, 20
episodes, no W&B).

```bash
sbatch scripts/smoke_experiments.sh
sbatch scripts/experiment_1.sh
sbatch scripts/experiment_2.sh
sbatch scripts/experiment_3.sh
```

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
