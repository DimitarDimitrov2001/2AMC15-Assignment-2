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

Run the example training script with one or more grid files:

```bash
uv run python train.py grid_configs/solvable.npy
```

Useful options:

- `--no_gui`: disable rendering for faster training.
- `--sigma`: set the stochasticity of the environment.
- `--fps`: set the GUI frame rate when rendering is enabled.
- `--iter`: set the number of training iterations.
- `--random_seed`: set the environment random seed.
- `--start_pos`: set the agent start position as `col,row`.

For the full command reference, run:

```bash
uv run python train.py --help
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
- `utils.rl_plots` defines `plot_value_function()`, `plot_policy()`, `plot_value_and_policy()`, `plot_algorithm_comparison()`, and `plot_hyperparameter_comparison()` for RL grid-world diagnostics.
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
