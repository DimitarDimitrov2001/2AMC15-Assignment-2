# Utils Module

Shared utilities for the reinforcement learning project: type definitions, learning algorithms, training loggers, and plotting helpers.

## Module Structure

| File                 | Responsibility                                                                                                     |
| -------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `types.py`           | Core type aliases (`State`, `Action`, `Q`, `Policy`, `Reward`, `Random`) and the abstract `Environment` base class |
| `training_logger.py` | Abstract `TrainingLogger` interface and concrete implementations for training progress output                       |
| `q_learning.py`      | Tabular Q-learning and Double Q-learning algorithms with epsilon-greedy exploration                                |
| `plotting.py`        | Matplotlib-based visualisation of training histories and hyperparameter grid search results                        |

## Type Aliases and the `Environment` ABC

All modules share a common vocabulary defined in `types.py`:

| Alias    | Underlying Type                                | Purpose                                                                           |
| -------- | ---------------------------------------------- | --------------------------------------------------------------------------------- |
| `State`  | `int`                                          | Integer-encoded environment state                                                 |
| `Action` | `int`                                          | Integer-encoded action                                                            |
| `Q`      | `np.ndarray`                                   | Q-value table of shape `(num_states, num_actions)`                                |
| `Policy` | `Callable[[State, Q, list[Action]], Action]`   | Behaviour policy: given a state, Q-table, and feasible actions, returns an action |
| `Reward` | `float`                                        | Scalar reward signal                                                              |
| `Random` | `np.random.Generator \| np.random.RandomState` | Random number generator accepted by all stochastic functions                      |

The `Environment` abstract base class requires two methods:

- **`feasible_actions(state) -> list[Action]`** -- returns the legal actions in the given state.
- **`simulate(state, action) -> (State, Reward)`** -- samples a stochastic transition and returns the next state and reward.

## Q-Learning Algorithms (`q_learning.py`)

### `q_learning(...)`

Standard off-policy Q-learning. Supports:

- Built-in `"epsilon_greedy"` behaviour policy or a custom `Policy` callable.
- Configurable learning rate (`alpha`), discount factor (`gamma`), and exploration rate (`epsilon`).
- **Epsilon decay**: exponential per-episode decay via `epsilon_decay` and `epsilon_min`. When `epsilon_decay > 0`, the effective epsilon at episode *t* is `max(epsilon_min, epsilon * (1 - epsilon_decay)^t)`.
- **Alpha decay**: harmonic per-episode decay via `alpha_decay` and `alpha_min`. When `alpha_decay > 0`, the effective alpha at episode *t* is `max(alpha_min, alpha / (1 + alpha_decay * t))`. This schedule satisfies the Robbins-Monro convergence conditions.
- Optional convergence detection via `convergence_threshold` on the maximum absolute Q-value change per episode.
- Optional periodic greedy policy evaluation recorded into the training history.
- Configurable `logging_interval` and optional `TrainingLogger` for live progress reporting.

### `double_q_learning(...)`

Double Q-learning variant that maintains two independent Q-tables (`Q1`, `Q2`). Action selection uses `Q1 + Q2`; each update step randomly updates one of the two tables to reduce maximisation bias. Has the same interface and features as `q_learning`, including decay schedules, greedy policy evaluation, and logger support.

### `epsilon_greedy(...)`

Standalone epsilon-greedy action selector used by both algorithms. With probability `epsilon` a random feasible action is chosen; otherwise the greedy action with respect to the provided Q-table is selected.

### Key Constants

| Constant                   | Default     | Description                                      |
| -------------------------- | ----------- | ------------------------------------------------ |
| `DEFAULT_ALPHA`            | `0.01`      | Initial learning rate                            |
| `DEFAULT_GAMMA`            | `0.9`       | Discount factor                                  |
| `DEFAULT_EPSILON`          | `0.1`       | Initial exploration rate                         |
| `DEFAULT_EPSILON_DECAY`    | `0.0`       | Epsilon decay rate (0 = no decay)                |
| `DEFAULT_EPSILON_MIN`      | `0.01`      | Floor for epsilon during decay                   |
| `DEFAULT_ALPHA_DECAY`      | `0.0`       | Alpha decay rate (0 = no decay)                  |
| `DEFAULT_ALPHA_MIN`        | `1e-4`      | Floor for alpha during decay                     |
| `DEFAULT_MAX_NUM_EPISODES` | `1_000_000` | Episode budget                                   |
| `GREEDY_EVAL_INTERVAL`     | `10`        | Episodes between greedy policy evaluations       |
| `GREEDY_EVAL_NUM_EPISODES` | `10`        | Rollouts per greedy evaluation                   |
| `LOGGING_INTERVAL`         | `1000`      | Default episodes between logger calls            |

## Training Loggers (`training_logger.py`)

All loggers implement the `TrainingLogger` abstract interface:

```
TrainingLogger (ABC)
├── JsonTrainingLogger        -- writes one JSON line per logged iteration to stdout
├── LiveQTableLogger          -- redraws a formatted Q-table in the terminal
└── GridSearchProgressLogger  -- writes progress to a multiprocessing-safe shared dict
```

Both algorithms call `logger.log_iteration(episode, q_values, q_delta, converged)` every `logging_interval` episodes and on convergence. The optional `close()` hook is called when training finishes.

### `GridSearchProgressLogger`

Designed for parallel grid search. Each worker process writes its progress into a `multiprocessing.Manager().dict()` keyed by a config identifier string. The companion function `run_progress_display(progress_dict, total_configs, stop_event)` is intended to run in a background thread in the main process; it polls the shared dict and renders a compact live-updating status table to stderr.

## Plotting Utilities (`plotting.py`)

### `plot_training_history(...)`

Creates a two-panel figure from a single training run:

- **Top panel**: per-episode average reward (raw + smoothed), and optionally greedy policy evaluation rewards.
- **Bottom panel**: per-episode average Q-value change (convergence indicator).

### `plot_grid_search_subplots(...)`

Arranges multiple training runs in an `n_epsilon x n_alpha` grid of subplot pairs (reward + delta-Q), with optional common y-axis scaling for cross-configuration comparison.

Both functions accept a `smoothing_window` parameter that controls moving-average smoothing.
