# RL in Practice - Group 5

This repository contains the reinforcement learning challenge environment and the unified training entry point for Group 5.

## Requirements

- Python 3.12 or newer
- [`uv`](https://docs.astral.sh/uv/) for dependency and virtual environment management

## Installation

Clone the repository and install the project dependencies:

```bash
git clone git@github.com:szelesteya/rl-in-practice-group-5.git
cd rl-in-practice-group-5
uv sync
```

## Usage

`train.py` is the single training entry point. The first positional
argument selects the agent (`value_iteration`, `q_learning`, `mc`,
`off_policy_mc`, or `random`); the rest are the grid files to train on.

```bash
uv run python train.py value_iteration grid_configs/A1_grid.npy --no_gui
uv run python train.py q_learning      grid_configs/A1_grid.npy --no_gui --episodes 3000
uv run python train.py mc              grid_configs/A1_grid.npy --no_gui --episodes 5000
uv run python train.py off_policy_mc   grid_configs/A1_grid.npy --no_gui --episodes 5000
uv run python train.py random          grid_configs/A1_grid.npy --no_gui
```

Useful shared options (available on every subcommand):

- `--no_gui`: disable rendering for faster training.
- `--sigma`: stochasticity of the environment.
- `--gamma`: discount factor.
- `--max_steps`: max environment steps per training episode and per evaluation rollout.
- `--eval_episodes`: number of evaluation rollouts to run after training.
- `--fps`: GUI frame rate when rendering is enabled.
- `--random_seed`: environment random seed.
- `--start_pos`: agent start position as `col,row`.
- `--out_dir`: directory for artifacts (defaults to `results/<agent>_<timestamp>/`).
- `--reward {manhattan,basic}`: reward function to use (default `manhattan`). See [Reward Function](#reward-function).
- `--compare_optimal`: pre-train a Value Iteration agent and use its policy as the optimality reference. Records per-episode policy disagreement (for `q_learning`, `mc`, `off_policy_mc`), emits a spatial `*_policy_diff.png` heatmap, and adds the end-of-training disagreement scalar to the evaluation summary. No-op for `value_iteration` and `random`.
- `--wandb` / `--wandb_project NAME`: stream training metrics to Weights & Biases. The active reward-function source code and constants are pinned into `wandb.config` so each run is fully reproducible from the run alone. See [Hyperparameter Tuning](#hyperparameter-tuning).

### Agent-specific options

**Value Iteration only** (`value_iteration`):

- `--theta`: Bellman convergence threshold.
- `--vi_max_iter`: maximum Bellman sweeps.

#### Shared tabular-Q-table flags

`q_learning`, `mc`, and `off_policy_mc` all share the same flag skeleton —
only the defaults differ. The flags are organised into argparse argument
groups (visible in `--help`):

| Group | Flags | Notes |
|---|---|---|
| Episode budget | `--episodes`, `--max_episode_length` | `--max_episode_length` is exposed by `mc` and `off_policy_mc` only. |
| Learning rate (alpha) | `--alpha`, `--alpha_min`, `--alpha_decay`, `--lr_schedule`, `--visit_count_c` | See learning-rate schedule note below. |
| Exploration (epsilon) | `--epsilon`, `--epsilon_min`, `--epsilon_decay`, `--fixed_epsilon` | `--fixed_epsilon` disables decay entirely. |
| Q-table initialisation | `--q_init`, `--q_init_noise` | Per-state-action initial value plus uniform tie-breaking noise. |
| Training log | `--log_interval`, `--log_q_table` | `--log_interval 0` disables console logging; W&B logging uses its own interval. |
| Early stopping | `--policy-stable-patience` | Default 50. Stops training once the tied-greedy policy is unchanged for that many consecutive episodes; pass `0` or a negative value to disable. Honoured by `q_learning`, `mc`, and `off_policy_mc` only — VI uses its own delta-based convergence and `random` has no policy. |

**Default-value table** (only the flags whose defaults differ between agents):

| Flag | `q_learning` | `mc` | `off_policy_mc` |
|---|---|---|---|
| `--episodes` | 3000 | 5000 | 5000 |
| `--max_episode_length` | n/a | 2000 | 2000 |
| `--alpha` / `--alpha_min` / `--alpha_decay` | 0.5 / 0.05 / 0.999 | 0.5 / 0.05 / 0.9995 | 0.2 / 0.02 / 0.9998 |
| `--epsilon` / `--epsilon_min` / `--epsilon_decay` | 1.0 / 0.05 / 0.995 | 0.2 / 0.01 / 0.9995 | 0.3 / 0.02 / 0.9998 |

#### Training-time exploration

- `--exploring_starts`: at the start of every training episode, sample a uniformly random empty cell as the agent's start (Sutton & Barto §5.4 exploring starts). Evaluation rollouts always start from `--start_pos`. Available for `q_learning`, `mc`, and `off_policy_mc`. Raises if the grid has no empty cells. The sampling RNG is seeded from `--random_seed + 1`, so a single `--random_seed` is enough to fully reproduce both the env stochasticity and the start sampling. The shared implementation lives in `agents.trainers.common.build_episode_start_picker` — a future variant (curriculum starts, distance-biased sampling, …) is a one-place change there, not three.

#### `off_policy_mc` only

In addition to the shared block, off-policy MC adds:

- `--off_policy_update {weighted,alpha}`: textbook cumulative weighted-importance-sampling (`weighted`) vs constant-α importance-weighted updates (`alpha`, default).
- `--importance_weight_clip`: max importance weight in `alpha` mode before multiplying by α; pass `None` to disable.
- `--soft_target_epsilon`: ε for an epsilon-soft target policy. `0.0` (default) is the textbook deterministic greedy target. Must be strictly less than `--epsilon`. *Empirically the soft target has not outperformed the deterministic target on this setup; the flag is kept for completeness.*

> **Learning rate schedules.** `--lr_schedule exponential` (default) decays α per episode using `--alpha`/`--alpha_decay`/`--alpha_min`. `--lr_schedule constant` keeps `--alpha` fixed throughout. `--lr_schedule visit_count` uses the Robbins-Monro schedule `α = c / (c + N(s, a))` per state-action pair, with `c` set via `--visit_count_c`. The legacy `--fixed_alpha` flag has been replaced by `--lr_schedule constant`. Implementation in `agents/learning_rates.py`. For `off_policy_mc --off_policy_update weighted`, the schedule is **inert**: weighted importance sampling has its own intrinsic step size `W / C(s, a)` and layering a scheduler on top would double-count.

> **MC training notes.** On-policy first-visit MC with ε-greedy is high-variance: at the default schedule a single 5000-episode run on `A1_grid` can swing between 0% and 100% eval success across random seeds. For more stable evaluation, increase `--episodes` (10k–20k) or aggregate across multiple `--random_seed` runs. Off-policy MC inherits the same variance profile.

For the canonical per-agent reference, run:

```bash
uv run python train.py q_learning --help
uv run python train.py off_policy_mc --help
```

### Artifacts written per run

Every run writes the following files to `--out_dir` (timestamped prefix `<grid>_<agent>_<timestamp>`):

- `*_metrics.json` — captured `TrainingHistory` plus the post-training evaluation metrics.
- `*_evaluation_summary.txt` — human-readable success rate / discounted return / policy-disagreement scalar.
- `*_path.png` and `*_path.txt` — agent rollout visualisation.
- `*_value_policy.png` — value-and-policy heatmap (Value Iteration only; QL/MC derive these on demand).
- `*_policy_diff.png` — spatial disagreement heatmap vs the VI reference (only when `--compare_optimal` is set, non-VI/random).
- `*_performance_curves.png` and `*_hyperparam_traces.png` — per-episode training curves (non-VI agents only).

## Hyperparameter Tuning

There are two supported workflows, depending on what you want to do:

### 1. Single-agent sweeps via `train.py` + Weights & Biases

For ad-hoc parameter sweeps (sensitivity to one or two hyperparameters) the
fastest path is to run `train.py` directly with `--wandb` and let W&B handle
logging and grouping:

```bash
uv run python train.py q_learning grid_configs/A1_grid.npy \
    --no_gui --episodes 5000 --compare_optimal \
    --alpha 0.5 --alpha_decay 0.999 --lr_schedule exponential \
    --epsilon 1.0 --epsilon_decay 0.995 \
    --wandb --wandb_project rl-in-practice
```

Each run logs:

- per-episode `discounted_return`, `delta_q`, `epsilon`, `alpha` (and, for off-policy MC, `importance_weight`),
- the `policy_diff` curve when `--compare_optimal` is set,
- the full source of the active `world/rewards.py` and the reward constants under `wandb.config.reward_*`, so a sweep configuration on the W&B side can be re-run from a run snapshot alone.

For a real grid search, drive the same command from a shell loop or a W&B
sweep config — every CLI flag exposed above is mirrored as a `wandb.config`
key and is therefore valid as a sweep parameter.

### 2. Assignment-aligned report sweeps via `run_experiments.py`

For the report's structured comparisons across all algorithms, use
`run_experiments.py`. It runs every algorithm in `experiments/specs.py::ALGORITHMS`
(currently `value_iteration`, `mc`, `q_learning`) against the 14 predefined
cases organised into 6 setup groups:

| Setup group | Cases |
|---|---|
| `grid_comparison` | one case per `--grid` argument |
| `discount_factor` | `gamma=0.6`, `gamma=0.9` |
| `stochasticity` | `sigma=0.02`, `sigma=0.5` |
| `exploration_epsilon` | `low_fixed_epsilon` (0.1), `high_fixed_epsilon` (0.5), `decaying_epsilon` |
| `learning_rate` | `low_fixed_alpha` (0.1), `high_fixed_alpha` (0.5), `decaying_alpha`, `visit_count` |
| `mc_episode_length` | `max_episode_length=500`, `max_episode_length=5000` |

```bash
uv run python run_experiments.py --quick                            # smoke-test all cases
uv run python run_experiments.py --seeds 0 1 2 --out_dir results/r1 # report-grade sweep
```

Output: a master `results.csv`, per-group `results.csv`, and per-group
plots (metric bars, learning curves, VI convergence, value/policy panels,
policy-disagreement heatmaps). To add or modify cases, edit
`experiments/specs.py::build_cases`.

> **Note.** `run_experiments.py` does not currently sweep `--reward`, the
> `off_policy_mc` agent, or W&B logging — those code paths only run via
> `train.py`. If you need them in a structured sweep, extend
> `experiments/runner.py::_train_config` and `experiments/specs.py::ALGORITHMS`.

## Project Structure

```text
.
├── agents/             # Agent implementations and per-agent trainer modules
│   ├── learning_rates.py  # LearningRateSchedule abstraction (exponential / visit-count)
│   └── trainers/          # Pure trainer functions (one module per agent)
├── docs/               # Usage guides, runnable examples, assignment materials
├── experiments/        # Assignment-aligned hyperparameter sweep suite
├── grid_configs/       # Grid files used by the training script
├── utils/              # Plotting, evaluation, artifact-writing, training logging
├── world/              # Environment, grid, GUI, reward functions, helper code
├── train.py            # Single training entry point (per-agent CLI subcommands)
├── run_experiments.py  # Assignment sweep entry point (uses experiments/)
├── pyproject.toml      # Project metadata and dependencies
└── uv.lock             # Locked dependency versions
```

## Documentation

- [Plotting utilities](docs/plotting.md): generic training-history plotting plus links to RL grid-world plotting references.
- [Training logger](docs/training_logger.md): console logging utilities for iterative RL training diagnostics.
- [RL plotting example](docs/examples/rl_plots_example.py): runnable demonstration of value/policy and algorithm-comparison plots.

## Agents

All agents should inherit from `agents.BaseAgent`. The environment expects each agent to implement:

- `update()`: update the agent after an environment step.
- `take_action()`: choose the next action.

The repository includes benchmark agents in `agents/null_agent.py` and `agents/random_agent.py`.

## Utilities

The `utils` package contains reusable helpers for training analysis:

- `utils.plotting` defines `TrainingHistory`, `SubplotConfig`, `plot_training_history()`, and `plot_training_histories()` for visualising arbitrary training metrics.
- `utils.rl_plots` defines `plot_value_function()`, `plot_policy()`, `plot_value_and_policy()`, `plot_policy_disagreement()`, `plot_algorithm_comparison()`, and `plot_hyperparameter_comparison()` for RL grid-world diagnostics.
- `utils.training_logger` defines `TrainingLogger` and `ConsoleTrainingLogger` for printing compact training progress and optional Q-table snapshots.

Runnable examples are available in `docs/examples/`:

```bash
uv run python docs/examples/plotting_example.py
uv run python docs/examples/rl_plots_example.py
uv run python docs/examples/training_logger_example.py
```

The plotting examples write generated figures to temporary directories. The logger example prints synthetic training progress in both scroll and frame modes.

## Grids

Grid files live in `grid_configs/` and are stored as NumPy arrays. To create new grids, start the grid creator:

```bash
uv run python world/grid_creator.py
```

Then open `http://127.0.0.1:5000` in your browser. New grids are saved to `grid_configs/`.

## Environment

The `world.Environment` class owns the interaction loop between an agent and a grid. The main methods are:

- `reset()`: reset the environment state.
- `step()`: advance the environment by one time step.
- `evaluate_agent()`: evaluate an agent after training.

Rendering is useful for debugging, but training without the GUI is much faster. Use `--no_gui` for longer training runs.

## Reward Function

`world.rewards` exposes two reward functions, selectable per run via
`--reward {manhattan,basic}`. Both are functions of the attempted next
grid cell.

### `--reward basic` (assignment specification)

The minimal reward described in the assignment brief:

- Empty cell (`0`), boundary wall (`1`), obstacle (`2`): `STEP_REWARD = -3`
- Target (`3`): `TARGET_REWARD = 10`

Wall-bumps cost the same as a normal step because the agent simply stays
in place — the assignment intentionally does not single them out.

### `--reward manhattan` (default, distance-shaped)

The default reward is shaped using the Manhattan distance from the
agent's actual start to the target, plus a small distance-from-start
bonus on empty cells that biases the agent towards making forward
progress:

| Cell | Reward |
|---|---|
| Empty (`0`) | `STEP_REWARD + (manhattan(start, agent) / manhattan(start, target)) * DISTANCE_FROM_START_REWARD` |
| Wall / obstacle (`1`, `2`) | `WALL_OR_OBSTACLE_REWARD = -4` |
| Target (`3`) | `max(MIN_TARGET_REWARD, DISTANCE_MULTIPLIER * manhattan(start, target))` |

Constants live at the top of `world/rewards.py`:

```python
STEP_REWARD = -3
WALL_OR_OBSTACLE_REWARD = -4
MIN_TARGET_REWARD = 10
DISTANCE_MULTIPLIER = 5.0
DISTANCE_FROM_START_REWARD = 3.0
```

Empty cells therefore range from `-3` near the start to `0` near the
target, separately rewarding walls more harshly so wall-bumps no longer
look like progress. The target reward scales with start-target Manhattan
distance so larger grids retain a strong success signal without further
reward shaping.

> **Reproducibility.** When `--wandb` is enabled, the entire
> `world/rewards.py` source plus all four constants are pinned into
> `wandb.config` for every run, so a logged run remains re-runnable even
> if the constants change later on `main`.

## Value Iteration

The tabular Value Iteration agent (`agents/value_iteration_agent.py`)
uses the known grid dynamics, the stochasticity parameter `--sigma`,
and whichever reward function `--reward` selects. The state is the
robot position `(col, row)`; walls and obstacles are blocked, failed
moves keep the robot in place, and reaching the target terminates the
episode.

Example command for the required A1 grid (low stochasticity, default
Manhattan reward):

```bash
uv run python train.py value_iteration grid_configs/A1_grid.npy \
    --no_gui --start_pos 1,12 --sigma 0.02 --gamma 0.9 \
    --theta 1e-6 --vi_max_iter 1000 --max_steps 1000 --eval_episodes 50 \
    --out_dir results/vi_A1_low_stochasticity_sigma_0_02_gamma_0_9
```
