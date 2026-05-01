# Utils Module

Reusable training diagnostics and visualization helpers used across this
project.

## Module Structure

| File | Responsibility |
| --- | --- |
| `plotting.py` | Generic training-history plots (`TrainingHistory`, `SubplotConfig`, single-run and multi-run plotting helpers). |
| `rl_plots.py` | Grid-world value/policy visualizations and multi-algorithm comparison plots. |
| `training_logger.py` | Console logger interface and implementation for iterative training progress. |

## `plotting.py` (Generic Training Curves)

`plotting.py` is format-agnostic: you pass a `TrainingHistory` instance (or a
plain dict in the same format), and the module renders metric curves.

Main API:

- `TrainingHistory`: typed wrapper for:
  - `episodes`
  - `metrics`
  - optional `hyperparams`
  - optional `metadata`
- `SubplotConfig`: per-metric visual overrides (`color`, `log_scale`,
  `threshold`, labels, etc.).
- `plot_training_history(...)`: one stacked subplot per metric for a single run.
- `plot_training_histories(...)`: panel grid for comparing multiple runs.

Both plotting functions support moving-average smoothing through
`smoothing_window`.

## `rl_plots.py` (Grid-World + Algorithm Comparison)

`rl_plots.py` builds on `TrainingHistory` and adds RL-specific visualizations
for grid tasks.

Grid and policy conventions:

- State keys use `(col, row)`.
- Grid indexing is `grid[col, row]`.
- Action integers are `0=Down`, `1=Up`, `2=Left`, `3=Right`.

Main API:

- `plot_value_function(...)`: value heatmap over the environment grid.
- `plot_policy(...)`: arrow map of greedy actions.
- `plot_value_and_policy(...)`: side-by-side value and policy plot.
- `plot_algorithm_comparison(...)`: overlays multiple algorithms on shared
  metric axes.
- `plot_hyperparameter_comparison(...)`: compares conditions in a
  metric-by-condition subplot grid.

## `training_logger.py` (Console Logging)

The logger module provides:

- `TrainingLogger` (abstract interface):
  - `log_iteration(...)`
  - optional `close()`
- `ConsoleTrainingLogger`:
  - status-line logging per training checkpoint
  - optional Q-table rendering
  - `redraw_mode="frame"` or `redraw_mode="scroll"`

## Runnable Examples

From the repository root:

```bash
uv run python docs/examples/plotting_example.py
uv run python docs/examples/rl_plots_example.py
uv run python docs/examples/training_logger_example.py
```

The plotting examples write figures to temporary directories.