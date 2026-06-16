# Reinforcement Learning in Practice

This repository contains a grid-world reinforcement learning environment, three implemented algorithms, and the experiment pipeline used to generate the report results.

Implemented algorithms:

- Value Iteration
- On-policy first-visit Monte Carlo control
- Q-learning

The repository is intended to be submitted and opened as a zip file. No Git setup is required to run it.

## Setup From Zip

Requirements:

- Python 3.12 or newer
- `uv` for dependency and virtual environment management

After extracting the zip file:

```powershell
cd path\to\extracted\project
uv sync
```

All commands below should be run from the repository root, the folder that contains `train.py`, `run_experiments.py`, `pyproject.toml`, and `uv.lock`.

## Running One Algorithm With `train.py`

Use `train.py` when you want to run one algorithm on one or more grid files with a chosen configuration.

General form:

```powershell
uv run python train.py <algorithm> <grid file> [options]
```

Available subcommands:

- `value_iteration`: model-based dynamic programming. It uses the known grid dynamics and reward function to compute an optimal value function and greedy policy.
- `mc`: on-policy first-visit Monte Carlo control. It samples full episodes with an epsilon-greedy policy and updates Q-values from first-visit returns.
- `q_learning`: model-free temporal-difference control. It updates Q-values after every transition using the Q-learning Bellman target.
- `random`: uniform-random baseline for evaluation only.

Useful shared options:

- `--no_gui`: disable rendering for faster training.
- `--sigma`: environment action stochasticity.
- `--gamma`: discount factor.
- `--eval_episodes`: number of post-training evaluation rollouts.
- `--eval_max_steps`: maximum steps per evaluation rollout.
- `--random_seed`: environment and agent seed.
- `--start_pos`: fixed start position as `col,row`.
- `--out_dir`: output directory for artifacts.
- `--compare_optimal`: for `mc` and `q_learning`, train a Value Iteration reference and record policy disagreement.

### Example: Value Iteration

```powershell
uv run python train.py value_iteration grid_configs/A1_grid.npy `
  --no_gui `
  --sigma 0.02 `
  --gamma 0.9 `
  --theta 1e-6 `
  --vi_max_iter 1000 `
  --eval_episodes 50 `
  --eval_max_steps 1000 `
  --out_dir results/value_iteration_example
```

Value Iteration-specific options:

- `--theta`: Bellman convergence threshold.
- `--vi_max_iter`: maximum number of Bellman sweeps.

### Example: On-Policy Monte Carlo

```powershell
uv run python train.py mc grid_configs/A1_grid.npy `
  --no_gui `
  --episodes 5000 `
  --max_episode_length 2000 `
  --alpha 0.5 `
  --epsilon 0.2 `
  --gamma 0.9 `
  --sigma 0.1 `
  --compare_optimal `
  --out_dir results/mc_example
```

MC-specific behavior:

- Samples complete episodes before updating Q-values.
- Uses epsilon-greedy behavior during training.
- Supports `--exploring_starts` to sample a random empty start cell for each training episode.

### Example: Q-learning

```powershell
uv run python train.py q_learning grid_configs/A1_grid.npy `
  --no_gui `
  --episodes 3000 `
  --max_episode_length 500 `
  --alpha 0.5 `
  --epsilon 1.0 `
  --epsilon_decay 0.995 `
  --gamma 0.9 `
  --sigma 0.1 `
  --compare_optimal `
  --out_dir results/q_learning_example
```

Q-learning-specific behavior:

- Updates Q-values after every environment step.
- Uses epsilon-greedy exploration during training.
- Switches to a greedy policy for evaluation after training.

### Shared Q-table Options for MC and Q-learning

Both `mc` and `q_learning` support:

- `--episodes`: number of training episodes.
- `--max_episode_length`: maximum steps per training episode.
- `--alpha`, `--alpha_min`, `--alpha_decay`: learning-rate parameters.
- `--lr_schedule {exponential,constant,visit_count}`: learning-rate schedule.
- `--visit_count_c`: offset for the visit-count learning-rate schedule.
- `--epsilon`, `--epsilon_min`, `--epsilon_decay`, `--fixed_epsilon`: exploration settings.
- `--q_init`, `--q_init_noise`: initial Q-values and optional tie-breaking noise.
- `--policy-stable-patience`: early stop after the tied-greedy policy is stable for this many episodes.
- `--log_interval`, `--log_q_table`: console logging options.
- `--exploring_starts`: sample a uniformly random empty cell as the training start for each episode.

For the full list of options:

```powershell
uv run python train.py value_iteration --help
uv run python train.py mc --help
uv run python train.py q_learning --help
```

## Deep-RL Training With `train_deep.py`

`train_deep.py` is a separate entry point for the Assignment-2 deep-RL stack: the continuous/minimal environments and the algorithm-agnostic `Trainer` in `training/`. It is independent of the tabular `train.py` above.

The random baseline and a DQN agent are wired today. Further learning agents (PPO, ...) plug in through the `_build_agent` factory without changing the training loop.

General form:

```powershell
uv run python train_deep.py --env {minimal|continuous} [options]
```

Example:

```powershell
uv run python train_deep.py --env minimal --agent dqn --episodes 5000 --max-steps 200 --visualize
```

Options:

- `--env {minimal,continuous}`: environment to train on (default `minimal`).
- `--agent {random,dqn}`: agent to train (default `random`).
- `--grid`: path to a `.npy` grid file (default `grid_configs/small_grid.npy`).
- `--start-pos X Y`: fixed continuous (x, y) start for evaluation and visualization (and for training unless `--exploring-starts` is set). Omit to use the grid's START_CELL, falling back to a random empty cell each episode (note: `small_grid.npy` has no START_CELL, so without this flag the start is randomized per episode).
- `--exploring-starts`: use random start positions during training (exploring starts) while evaluation/visualization keep the fixed `--start-pos`. Helps discovery on sparse-reward grids by seeding the replay buffer with goal-reaching transitions from varied starts.
- `--episodes`, `--max-steps`, `--seed`: training budget and seed (default `20000` episodes, `200` max steps per episode).
- `--device {auto,cpu,cuda,mps}`: compute device for learning agents (default `auto`, which picks cuda > mps > cpu). For the small grid MLP, `cpu` is often fastest.
- `--epsilon`: fixed epsilon-greedy exploration rate for DQN (default `0.1`). Eval always uses greedy action selection regardless of this value.
- `--log-interval`: print metrics (and log a rollout image when `--visualize`) every N episodes (default `1`).
- DQN-style agents use a replay buffer with default capacity `1_000_000` transitions (`agents.replay_buffer.DEFAULT_CAPACITY`), sized for the default episode budget.
- `--eval-interval`, `--eval-episodes`: evaluation cadence and rollouts per evaluation.
- `--out-dir`: when set, writes `best.pt`/`last.pt` checkpoints and `history.json` there.
- `--wandb`, `--wandb-group`: enable Weights & Biases logging and optionally bucket runs under a group name.
- `--visualize`, `--viz-out`, `--viz-max-steps`: save a post-training rollout path image (reuses `visualize_random_agent.py`). When combined with `--wandb`, a greedy rollout is also rendered every `--log-interval` episodes and logged to the W&B `viz/rollout` panel (frames saved under `<out-dir>/rollouts/`).

The `Trainer` (`training/trainer.py`) is algorithm-agnostic and supports an optional environment-step budget (`max_env_steps`), per-episode mean of agent update metrics, best/last checkpointing via `BaseAgent.save_checkpoint`/`load_checkpoint`, and history-to-disk. Both new environments accept an optional per-episode `seed` in `reset(seed=...)`.

## Reproducing Report Results With `run_experiments.py`

Use `run_experiments.py` to reproduce the structured experiment suite used for the report. It runs:

- Value Iteration
- On-policy MC
- Q-learning

across the predefined experiment groups in `experiments/specs.py`.

To reproduce the report results with seeds `0` through `4`:

```powershell
uv run python run_experiments.py --seeds 0 1 2 3 4
```

This takes roughly 1 to 1.5 hours depending on the computer. The repository also includes a `report_results/` folder with generated outputs from the command above. The `report_results/aggregated_overview.md` file is used to create the summary table in the report.

For a quick smoke test that keeps the same experiment structure but uses much smaller budgets (this command does not produce the report results):

```powershell
uv run python run_experiments.py --quick
```

The report defaults, grids, algorithms, and experiment cases are defined in:

```text
experiments/specs.py
```

Important objects in that file:

- `ALGORITHMS`: algorithms used by the report suite.
- `DEFAULT_GRIDS`: default grids used by the suite.
- `DEFAULTS`: report-grade default hyperparameters.
- `QUICK_OVERRIDES`: shortened budgets used by `--quick`.
- `build_cases()`: experiment groups and condition overrides.


Default report-case hyperparameters:

| Hyperparameter | Default value |
|---|---:|
| `sigma` | `0` |
| `gamma` | `0.99` |
| `eval_episodes` | `50` |
| `eval_max_steps` | `1000` |
| `random_seed` | `0` |
| `exploring_starts` | `True` |
| `alpha` | `0.2` |
| `alpha_min` | `0.01` |
| `alpha_decay` | `0.9995` |
| `lr_schedule` | `visit_count` |
| `visit_count_c` | `50` |
| `epsilon` | `0.7` |
| `epsilon_min` | `0.05` |
| `epsilon_decay` | `0.99995` |
| `fixed_epsilon` | `False` |
| `ql_episodes` | `100000` |
| `mc_episodes` | `100000` |
| `max_episode_length` | `1500` |
| `theta` | `1e-6` |
| `vi_max_iter` | `1000` |
| `policy_stable_patience` | `1000` |

The experiment suite writes a master `results.csv`, per-group CSV files, overview summaries, aggregated summaries, and plots under the output directory. By default this is:

```text
results/assignment_experiments
```

You can choose another output directory:

```powershell
uv run python run_experiments.py --seeds 0 1 2 3 4 --out_dir results/report_reproduction
```

## Experiment Groups

`run_experiments.py` uses the cases defined in `experiments/specs.py`:

- `default`: baseline settings on the primary grid.
- `grid_comparison`: baseline settings on each selected grid.
- `discount_factor`: compares `gamma=0.6` and `gamma=0.9`.
- `stochasticity`: compares `sigma=0.02` and `sigma=0.5`.
- `exploration_epsilon`: compares low fixed epsilon, high fixed epsilon, and decaying epsilon.
- `learning_rate`: compares fixed, decaying, and visit-count alpha schedules.
- `mc_episode_length`: compares `max_episode_length=500` and `max_episode_length=5000`.

## Outputs

Single `train.py` runs write artifacts to `--out_dir`, including:

- metrics JSON
- evaluation summary text
- rollout path image/text
- value/policy plots
- training curves for MC and Q-learning
- policy disagreement plots when `--compare_optimal` is enabled

`run_experiments.py` writes:

- `results.csv`: master table of all runs
- `<group>/results.csv`: per-group result tables
- `overview.csv` and `overview.md`: per-run overview
- `aggregated_overview.csv` and `aggregated_overview.md`: seed-aggregated overview
- plot folders under each experiment group

## Project Structure

```text
.
|-- agents/
|   |-- value_iteration_agent.py   # Value Iteration implementation
|   |-- mc_agent.py                # On-policy first-visit MC agent
|   |-- q_learning_agent.py        # Q-learning agent
|   |-- learning_rates.py          # Learning-rate schedule implementations
|   `-- trainers/                  # Training loops for each algorithm
|-- docs/                          # Extra documentation and examples
|-- experiments/
|   |-- specs.py                   # Report defaults and experiment cases
|   |-- runner.py                  # Executes cases and writes CSVs
|   |-- plots.py                   # Generates report plots
|   `-- overview.py                # Creates overview and aggregate summaries
|-- grid_configs/                  # Saved NumPy grid files
|-- report_results/                # Existing generated report outputs
|-- utils/                         # Evaluation, plotting, logging, artifacts
|-- world/                         # Grid-world environments, GUI, grid tools
|   |-- environment_base.py        # BaseGridEnvironment: shared episode scaffolding
|   |-- minimal_environment.py     # Point-mass (x, y) environment
|   `-- continuous_environment.py  # Robot with heading + distance sensors
|-- train.py                       # Single-run CLI
|-- run_experiments.py             # Report experiment CLI
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

## Notes

- Training is much faster with `--no_gui`.
- The report experiment defaults are in `experiments/specs.py`, not in `train.py`.
- `train.py` defaults are convenient single-run defaults; `run_experiments.py` uses the report defaults from `experiments/specs.py`.
- The `random` subcommand is a baseline and does not learn.
