# `training_logger` -- Console Training Logging

## Concept

`utils/training_logger.py` provides a small, environment-agnostic logger
for iterative reinforcement-learning training.  It keeps the same
`log_iteration(...)` call signature used by the current training functions, so
it can be passed directly into `q_learning`, `double_q_learning`, or
`q_learning_vfa`.

Unlike the older live logger, this module does not decode states into domain
labels.  If a Q-table is shown, states and actions are displayed as plain
integer indices.

## Logger Interface

All loggers implement:

```python
def log_iteration(
    episode: int,
    q_values: np.ndarray,
    q_delta: float,
    converged: bool,
    current_alpha: float | None = None,
    current_epsilon: float | None = None,
    mean_q_delta: float | None = None,
) -> None:
    ...
```

The existing training algorithms call this method every `logging_interval`
episodes and once more when convergence is reached.

## `ConsoleTrainingLogger`

```python
from utils.training_logger import ConsoleTrainingLogger

logger = ConsoleTrainingLogger(
    show_q_table=True,
    states=[0, 1, 2],
    redraw_mode="frame",
)
```

Options:

| Option | Type | Description |
| --- | --- | --- |
| `show_q_table` | `bool` | Include a Q-table below the status line. |
| `states` | `list[int] | None` | State rows to print. `None` prints all rows. |
| `redraw_mode` | `"frame" | "scroll"` | `"frame"` overwrites previous output on a TTY; `"scroll"` appends each log. |

Status line example:

```text
Episode: 5000 | max |dQ|: 0.001234 | mean |dQ|: 0.000890 | eps: 0.0500 | alpha: 0.0100 | converged: False
```

Optional Q-table example:

```text
state |       Q[0]       Q[1]       Q[2] |      q_max argmax
------------------------------------------------------------
    0 |    -1.2340     0.5612     0.0023 |     0.5612      1
    7 |    -0.9800    -0.1234     0.8900 |     0.8900      2
```

## Example

Run:

```bash
uv run python docs/examples/training_logger_example.py
```

The script simulates a tiny training loop with synthetic Q-values and shows both
`"scroll"` and `"frame"` modes.
