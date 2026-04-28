# `plotting` -- Training Plot Utilities

## Concept

`utils/plotting.py` provides **format-agnostic** plotting for
reinforcement-learning (or any iterative optimisation) training runs.

Instead of hard-coding which metrics exist or how the training history is
structured, callers supply a simple standardised dict (or a typed
`TrainingHistory` wrapper) and the module figures out the rest:

* **One subplot per metric** -- add a new metric to the dict and it gets its
  own subplot automatically.
* **Configurable appearance** -- per-metric overrides (log scale, colours,
  threshold lines) via `SubplotConfig`.
* **Multi-run comparison** -- pass a list of histories and get a panel grid
  with one panel per run.

---

## The `TrainingHistory` Format

Every function accepts **either** a plain `dict` **or** a `TrainingHistory`
instance.  Both represent the same shape:

```python
{
    "episodes": [1, 2, 3, ...],        # x-axis values       (required)
    "metrics": {                        # metric_name -> list  (required, >= 1)
        "avg_reward": [float, ...],
        "delta_q":    [float, ...],
    },
    "hyperparams": {                    # optional -- used for auto-titles
        "epsilon": 0.1,
        "alpha":   0.01,
    },
    "metadata": {                       # optional -- extra info
        "converged": True,
    },
}
```

### Using the class

```python
from utils.plotting import TrainingHistory

# From scratch
h = TrainingHistory(
    episodes=[1, 2, 3],
    metrics={"reward": [0.1, 0.5, 0.9]},
    hyperparams={"lr": 0.01},
)

# From a dict
h = TrainingHistory.from_dict(my_dict)

# Back to a dict
d = h.to_dict()

# Dict-style access also works
h["metrics"]
```

---

## `SubplotConfig`

Per-metric visual overrides passed via `subplot_config`:

| Field        | Type             | Default | Description                                  |
|------------- |------------------|---------|----------------------------------------------|
| `label`      | `str \| None`    | `None`  | Legend label (defaults to metric name)        |
| `color`      | `str \| None`    | `None`  | Line colour (auto-picked when `None`)        |
| `log_scale`  | `bool`           | `False` | Logarithmic y-axis                           |
| `symlog`     | `bool`           | `False` | Symmetric-log y-axis (handles negatives)     |
| `threshold`  | `float \| None`  | `None`  | Horizontal reference line                    |
| `raw_alpha`  | `float`          | `0.4`   | Opacity of the raw trace (0 = hidden)        |
| `y_label`    | `str \| None`    | `None`  | Custom y-axis label                          |

---

## API Quick-Reference

### `plot_training_history`

```python
plot_training_history(
    history,                              # TrainingHistory | dict
    smoothing_window=1,
    subplot_config=None,                  # dict[str, SubplotConfig] | None
    title="Training Progress",
    figsize=(12, 7),
) -> (fig, axes, smoothed_metrics)
```

One stacked subplot per metric.  Each shows the raw trace (faded) and a
smoothed trace (bold).

### `plot_training_histories`

```python
plot_training_histories(
    histories,                            # list[TrainingHistory | dict]
    metrics_to_plot=None,                 # list[str] | None  (default: all)
    smoothing_window=1,
    columns=3,
    common_scale=True,
    subplot_config=None,
    title="Training Runs",
    figsize_per_panel=(4.5, 3.5),
) -> (fig, axes_2d)
```

Grid of panels -- one per run.  Titles are auto-generated from each
history's `hyperparams`.

---

## Example

A runnable example script lives at
[`docs/examples/plotting_example.py`](examples/plotting_example.py).

Run it with:

```bash
uv run python docs/examples/plotting_example.py
```

It generates two figures with synthetic data demonstrating both functions.
