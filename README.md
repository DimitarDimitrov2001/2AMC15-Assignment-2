# RL in Practice - Group 5

This repository contains the reinforcement learning challenge environment and example training entry point for Group 5.

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
argument selects the agent (`value_iteration`, `q_learning`, `mc`, or
`random`), the rest are the grid files to train on.

```bash
uv run python train.py value_iteration grid_configs/A1_grid.npy --no_gui
uv run python train.py q_learning      grid_configs/A1_grid.npy --no_gui --episodes 3000
uv run python train.py mc              grid_configs/A1_grid.npy --no_gui --episodes 5000
uv run python train.py random          grid_configs/A1_grid.npy --no_gui
```

Useful shared options (available on every subcommand):

- `--no_gui`: disable rendering for faster training.
- `--sigma`: set the stochasticity of the environment.
- `--gamma`: discount factor.
- `--max_steps`: max environment steps per training episode and per evaluation rollout.
- `--eval_episodes`: number of evaluation rollouts to run after training.
- `--fps`: set the GUI frame rate when rendering is enabled.
- `--random_seed`: set the environment random seed.
- `--start_pos`: set the agent start position as `col,row`.
- `--out_dir`: directory for artifacts (defaults to `results/`).
- `--compare_optimal`: pre-train a Value Iteration agent and use its policy as the optimality reference. Records per-episode policy disagreement (for `q_learning` / `mc`), emits a spatial `*_policy_diff.png` heatmap, and adds the end-of-training disagreement scalar to the evaluation summary. No-op for `value_iteration` and `random`.

Agent-specific options:

- **`value_iteration`**: `--theta`, `--vi_max_iter`.
- **`q_learning`**: `--episodes`, `--alpha`, `--alpha_min`, `--alpha_decay`, `--lr_schedule`, `--visit_count_c`, `--epsilon`, `--epsilon_min`, `--epsilon_decay`, `--fixed_epsilon`.
- **`mc`**: same alpha/epsilon flags as `q_learning`, plus `--episodes` and `--max_episode_length`. Uses constant-α first-visit updates (`Q ← Q + α·(G − Q)`); the classical 1/N sample-mean variant is not supported.

> **Learning rate schedules.** `--lr_schedule exponential` (default) decays α per episode using `--alpha`/`--alpha_decay`/`--alpha_min`. `--lr_schedule constant` keeps `--alpha` fixed throughout. `--lr_schedule visit_count` uses the Robbins-Monro schedule `α = c / (c + N(s, a))` per state-action pair, with `c` set via `--visit_count_c`. The `--fixed_alpha` flag has been replaced by `--lr_schedule constant`.

> **MC training notes.** On-policy first-visit MC with ε-greedy is high-variance: even at the default schedule, a single 5000-episode run on `A1_grid` can swing between 0% and 100% eval success across random seeds. For more stable evaluation, increase `--episodes` (10k–20k) or aggregate across multiple `--random_seed` runs.

For the full per-agent reference, run:

```bash
uv run python train.py q_learning --help
```

## Project Structure

```text
.
├── agents/          # Base agent interface and benchmark agents
├── docs/            # Usage guides and runnable examples
├── grid_configs/    # Grid files used by the training script
├── utils/           # Plotting and training logging utilities
├── world/           # Environment, grid, GUI, and helper code
├── train.py         # Example training entry point
├── pyproject.toml   # Project metadata and dependencies
└── uv.lock          # Locked dependency versions
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

The `world.rewards` module provides a lightweight reward function based on the attempted next grid cell:

- Empty cell (`0`): `-1`
- Boundary wall (`1`) or obstacle (`2`): `-5`
- Target (`3`): `max(10, 2 * manhattan_distance(start_pos, target_pos))`

This keeps the reward easy to set up from the grid, the actual start position, and the destination. The `-1` step reward encourages shorter paths, while the `-5` wall and obstacle penalty discourages invalid moves without over-penalising stochastic actions from `--sigma`. The target reward scales with Manhattan distance so larger grids still provide a strong success signal without adding extra reward shaping.

## Value Iteration Additions

This project now includes a tabular value-iteration implementation for Assignment 1. The new agent is implemented in `agents/value_iteration_agent.py` and uses the known grid dynamics, the stochasticity parameter `sigma`, and the Manhattan reward function from `world/rewards.py`. The state is the robot position `(col, row)`, with empty cells and the target treated as valid states. Walls and obstacles are blocked, failed moves keep the robot in place, and reaching the target terminates the episode.

`train.py` exposes a `value_iteration` subcommand that wraps the agent. The value-iteration options include `--gamma`, `--theta`, and `--vi_max_iter`. Evaluation is run after training, using `--max_steps` as the maximum number of steps per rollout and `--eval_episodes` as the number of evaluation rollouts.

Evaluation and artifact writing live in helper modules: `utils/evaluation.py` computes rollout metrics such as success rate, discounted return, undiscounted return, and episode length, while `utils/artifacts.py` saves metrics, evaluation summaries, value/policy plots, and path visualisations. A typical value-iteration run writes `*_metrics.json`, `*_evaluation_summary.txt`, `*_value_policy.png`, `*_path.png`, and `*_path.txt` to the selected `--out_dir`.

Example command for the required A1 grid:

```bash
uv run python train.py value_iteration grid_configs/A1_grid.npy --no_gui --start_pos 1,12 --sigma 0.02 --gamma 0.9 --theta 1e-6 --vi_max_iter 1000 --max_steps 1000 --eval_episodes 50 --out_dir results/vi_A1_low_stochasticity_sigma_0_02_gamma_0_9
```
