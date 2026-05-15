# Refactor backlog

Items are listed in **recommended execution order**. Each item shows its
dependencies, a verdict on the original draft idea, the concrete evidence
behind the verdict, and a suggested prompt for the moment the task is picked up.

## Reading guide

```text
1. Unified training entry point + trainer extraction   ← foundation; do first
2. Align agent value/policy/training-loop interfaces   ← needs 1
3. Consolidate grid cell-value constants               ← independent; cheap
4. Audit MC code for correctness                       ← easier after 2
5. Policy-disagreement metric and plots                ← needs 1 + 2
6. Use shared plotting helpers in run_experiments.py   ← needs 1 + 2
```

The biggest risk in deviating from this order: doing Item 2 before Item 1
means refactoring the MC training loop into the agent class and then
immediately ripping it back out. Doing Items 5 or 6 before Item 2 means
sprinkling `getattr`/`callable` adapters across the codebase that have to be
removed later.

---

## 1. Unified training entry point + trainer extraction

**Verdict: Foundational. Everything else gets easier once this lands.**

### Goal

Collapse 5 training entry points (`train.py`, `train_vi.py`, `train_ql.py`, `train_mc.py`, and the duplicate loops in `run_experiments.py`) into **one CLI** that dispatches to **one shared training implementation per agent**. The sweep script consumes the same trainers, so there is no second source of truth.

### Target layout

```text
train.py                       # single CLI, argparse subparsers per agent
agents/trainers/
  ├── __init__.py              # re-exports train_*; dispatch table {name: train_fn}
  ├── common.py                # setup_grid_run(), save_run_artifacts(), TrainConfig dataclass
  ├── value_iteration.py       # train(env, cfg) -> (agent, history)
  ├── q_learning.py            # train(env, cfg) -> (agent, history)
  └── mc.py                    # train(env, cfg) -> (agent, history)
run_experiments.py             # imports from agents.trainers; keeps sweep specs only
```

### Design invariants (do not violate)

- Trainer functions are **pure**: take `env` + config, return `(agent, history)`. No `argparse`, no `print`, no file I/O, no plotting.
- `setup_grid_run(grid_path, sigma, fps, no_gui, start_pos, random_seed) -> (env, initial_pos, reward_fn)` is the **only** place that does the dummy-env → reset → build reward_fn → patch dance.
- `save_run_artifacts(out_dir, prefix, agent, grid, initial_pos, eval_metrics, reward_fn, sigma, max_steps, random_seed)` is the **only** place that writes per-run artifacts (metrics JSON, eval summary, value/policy PNG for VI, path visualization).
- CLI is the **only** layer that calls `argparse`, prints to stdout, and writes artifacts. Sweep is the **only** layer that knows about experiment groups and comparison plots.
- No new abstraction unless it removes existing duplication. Three agents do not justify a factory registry.

### Steps

1. **Create `agents/trainers/common.py`**
   - `TrainConfig` dataclass (or `TypedDict`) with all hyperparameters used across agents. Agent-specific fields default to `None`.
   - `setup_grid_run(...)` — extract from `train.py:140-155`, `train_vi.py:43-54`, `train_ql.py:55-66`, `train_mc.py:54-65`, `run_experiments.py:266-272`.
   - `save_run_artifacts(...)` — extract from `train.py:213-236` and the duplicated tails of the per-agent trainers.

2. **Create `agents/trainers/value_iteration.py`**
   - `train(grid_array, reward_fn, cfg) -> (ValueIterationAgent, history_dict | None)`.
   - Move logic from `run_experiments.py:_train_value_iteration` and `train_vi.py:56-61`.
   - Returns the same history shape that the sweep and CLI both consume.

3. **Create `agents/trainers/q_learning.py`**
   - `train(env, reward_fn, cfg) -> (QLearningAgent, history_dict)`.
   - Move the training loop currently duplicated in `train.py:92-110`, `train_ql.py:78-91`, and `run_experiments.py:_train_q_learning:199-227`.
   - Build the `history` dict in one place (currently constructed inline in `run_experiments.py:219-227`).

4. **Create `agents/trainers/mc.py`**
   - `train(env, reward_fn, cfg, start_pos) -> (MCAgent, history_dict | None)`.
   - Thin wrapper around `MCAgent.train(...)` (the agent already owns its loop). The loop will be moved out of the agent class in Item 2; for now keep `MCAgent.train()` as-is.
   - Move the construction logic from `train_mc.py:67-80` and `run_experiments.py:_train_mc:234-255`.

5. **Rewrite `train.py` as a thin dispatcher**
   - `argparse` subparsers: `value_iteration`, `q_learning`, `mc`, `random`. Each subparser owns its agent-specific args.
   - Shared args (`--sigma`, `--gamma`, `--no_gui`, `--fps`, `--max_steps`, `--eval_episodes`, `--random_seed`, `--start_pos`, `--out_dir`, `GRID [GRID ...]`) go on a shared parent parser.
   - Dispatcher: build `TrainConfig` from args → `setup_grid_run` → call trainer → `evaluate_policy_metrics` → `save_run_artifacts` → `Environment.evaluate_agent` for path viz.
   - Target: under 100 lines.

6. **Migrate `run_experiments.py`**
   - Delete `_train_value_iteration`, `_train_q_learning`, `_train_mc` and the per-run env bootstrap block in `_run_one`. Call into `agents.trainers` instead.
   - Keep `DEFAULTS`, `EXPERIMENTS`, `EXP_GROUPS`, `TRAINING_CURVE_GROUPS`, the VI cache, and all plotting code unchanged — this is the part that should stay declarative.
   - Expected reduction: ~150 lines.

7. **Delete dead/competing entry points**
   - Remove `train_vi.py`, `train_ql.py`, `train_mc.py`.
   - Remove the `__main__` block and `train_mc_agent` helper from `agents/mc_agent.py` (lines 212-320). They are stale duplicates of the real trainers.

8. **Update `README.md`** (per the always-applied rule about doc alignment)
   - Replace the "Usage" section examples to use the new subcommand form:
     `uv run python train.py q_learning grid_configs/A1_grid.npy --episodes 3000 ...`
   - Update the `--help` reference and the A1 example in "Value Iteration Additions".

9. **Smoke test each subcommand end-to-end** on `grid_configs/A1_grid.npy`:
   - `train.py value_iteration A1_grid.npy --no_gui --eval_episodes 5`
   - `train.py q_learning A1_grid.npy --no_gui --episodes 100 --eval_episodes 5`
   - `train.py mc A1_grid.npy --no_gui --episodes 200 --eval_episodes 5`
   - Then run a tiny sweep: `run_experiments.py --grid grid_configs/A1_grid.npy --ql_episodes 50 --mc_episodes 50 --eval_episodes 3` and confirm CSVs + plots still generate.

### Out of scope (do not do as part of this refactor)

- Do **not** turn agents into a plugin registry or introduce a `BaseTrainer` ABC. Three free functions in three modules is enough.
- Do **not** redesign `TrainingHistory` or the plotting API.
- Do **not** change agent class internals (`MCAgent`, `QLearningAgent`, `ValueIterationAgent`) beyond removing the stale `__main__`/helper in `mc_agent.py`. Style alignment is Item 2.
- Do **not** introduce a config file format (YAML/TOML). CLI flags + the sweep spec module are sufficient.

---

## 2. Align agent value/policy/training-loop interfaces

**Dependencies: Item 1.**

**Verdict: Agree. The most worthwhile cleanup item after the entry-point refactor. Reframed from the original "MC agent match style" to make explicit that QL and possibly VI also need to change — it's not unilaterally MC's problem.**

### What's wrong today

Three agents, three shapes for the same concept:

- **Training loop ownership** — `MCAgent.train(env, n_episodes, ...)` at `agents/mc_agent.py:137-180` owns its own loop. `QLearningAgent` does not; its loop lives in the caller. `ValueIterationAgent.train()` takes no env (DP doesn't need one). Three different shapes.
- **MC owns its own `__main__` block + `train_mc_agent` helper** (`agents/mc_agent.py:212-320`). No other agent does. (Deleted in Item 1 step 7; mentioned here for completeness.)
- **Naming inconsistencies**: `self.Q` (upper-case, public) on MC vs `self.q_table` (lower-case) on QL. `self._N` (private but capitalised) on MC. `greedy_action()` on MC has no counterpart on QL.
- **`MCAgent.update()` is a no-op stub** (`mc_agent.py:85-86`) because MC really wants `_record_step` + `end_episode`. The `BaseAgent` interface is being subverted rather than extended.
- **`values`/`policy` exposure differs across agents**:
  - VI: `self.values` and `self.policy` are attributes populated in `train()`.
  - QL: `values()` and `policy()` are **methods** at `q_learning_agent.py:102-114`.
  - MC: attributes populated in `_build_value_and_policy()` after training.
- **`MCAgent._alpha_current` two-attribute design**: `self.alpha` (initial), `self._alpha_current` (mutable, possibly `None`). QL just uses `self.alpha` and mutates it directly. MC's split exists because `alpha=None` means 1/N — defensible, but it's the only agent doing it.

### Recommendation

- **Adopt VI's shape as the reference** for `values`/`policy` exposure: both are dicts, populated by the end of `train()` (or by `set_eval_mode()` for QL). Convert QL's `values()` and `policy()` methods into attributes. MC already matches.
- **Move MC's training loop out of the agent class** into `agents/trainers/mc.py` (created in Item 1 step 4). `MCAgent` should expose `start_episode()` / `take_action()` / `record_step()` / `end_episode()` and let the trainer call them in a loop, the same way QL is driven today.
- **Rename `self.Q` → `self.q_table`, `self._N` → `self._visit_counts`**. Drop `greedy_action()` since `policy[state]` will replace it.
- **Leave the `_alpha_current` split alone** — the 1/N branch genuinely needs it. Add a one-line comment instead.

### Suggested prompt

> "Align `MCAgent` and `QLearningAgent` interfaces, using `ValueIterationAgent` as the reference shape. Three concrete changes:
>
> (a) Move the training loop out of `MCAgent` (`agents/mc_agent.py:137-180`) into `agents/trainers/mc.py`. `MCAgent` should expose `start_episode()` / `take_action()` / `record_step(state, action, reward)` / `end_episode() -> float` and let the trainer drive the loop — same pattern as QL. Preserve `_build_value_and_policy()` and `_build_history()` but call them from the trainer at end-of-training, not from inside `train()`.
>
> (b) Convert `QLearningAgent.values()` and `policy()` from methods to attributes populated by `set_eval_mode()` (`agents/q_learning_agent.py:96-114`). Match VI's shape exactly: `self.values: dict[Position, float]`, `self.policy: dict[Position, int]`.
>
> (c) Rename `MCAgent.Q` → `q_table` and `MCAgent._N` → `_visit_counts` for naming consistency with QL. Drop `greedy_action()` since callers should use `policy[state]` instead. Update `run_experiments.py:158-165` accordingly.
>
> Do not change the MC or QL algorithms themselves — only the interface and where the loop lives. Verify success on `grid_configs/A1_grid.npy`: end-of-training reward and policy should match the pre-refactor outputs to within noise."

---

## 3. Consolidate grid cell-value constants

**Dependencies: None. Can be done any time.**

**Verdict: Half-agree with the original draft. The duplication is real, but the right answer is "consolidate in `world/`", not "create a new file".**

### What's duplicated

Same constants in **three** places with the same values:

- `world/rewards.py:11-14` — `EMPTY_CELL`, `BOUNDARY_WALL_CELL`, `OBSTACLE_CELL`, `TARGET_CELL`. De-facto canonical location; `WALL_OR_OBSTACLE_REWARD` and friends already live here.
- `agents/value_iteration_agent.py:27-31` — same four constants plus `START_CELL = 4`, imported nowhere else.
- `utils/rl_plots.py:29-32` — `_EMPTY`, `_BOUNDARY`, `_OBSTACLE`, `_TARGET` (renamed/prefixed but identical values).

`world/helpers.py:6-11` already owns `ACTIONS_TO_DIRECTIONS`, which is the action-side equivalent. `world/__init__.py` exports neither.

### Recommendation

- A **separate file like `world/constants.py`** is fine but slightly over-engineered. There's no maintenance cost difference between one constants file and constants alongside `rewards.py`. The constants belong somewhere in `world/` because the `world` package owns the grid format.
- The **better fix** is to (a) keep them in `world/rewards.py` (or rename to `world/grid_codes.py` if a more neutral home is preferred), (b) re-export from `world/__init__.py`, (c) delete the duplicates in `value_iteration_agent.py` and `rl_plots.py` and import them instead. `START_CELL` can move to the shared location too — it's still a grid code.
- Push back on the impulse to also extract action constants into the same file. `ACTIONS_TO_DIRECTIONS` lives correctly in `world/helpers.py`; don't move it just for symmetry.

### Suggested prompt

> "Consolidate the grid cell-value constants currently duplicated in `world/rewards.py`, `agents/value_iteration_agent.py:27-31`, and `utils/rl_plots.py:29-32`. Pick one canonical location inside `world/` (recommend keeping them in `world/rewards.py` or moving to a new `world/grid_codes.py` — make the call yourself with one sentence of justification), re-export from `world/__init__.py`, and remove the duplicates. Do not touch action constants (`ACTIONS_TO_DIRECTIONS`) — they already have a good home in `world/helpers.py`. Verify imports across `agents/`, `utils/`, and `world/` still resolve."

---

## 4. Audit MC code for correctness

**Dependencies: Item 2 (easier once `MCAgent`'s surface is smaller).**

**Verdict: Agree this should happen, but the original "look for mistakes" framing is too vague — sharpen it before delegating.**

### Specific concerns worth checking

From a quick read of `agents/mc_agent.py`:

- **`_record_step` uses `info["actual_action"]`, not the agent's chosen action** (`mc_agent.py:150`). This is correct for on-policy MC under stochastic transitions (you update the action that actually happened), but it looks like a bug at first glance and isn't commented. Could also be wrong — depends on what the project specification expects.
- **Convergence check on reward stability** (`mc_agent.py:166-177`) — patience of 200 with a window of 50/50 means at least 200 consecutive "stable" episodes are required. The `consecutive_converged` counter resets to 0 on any non-stable episode, which combined with MC's noise can mean it never triggers. Worth checking whether this is firing during sweeps or is effectively dead code.
- **First-visit logic** (`mc_agent.py:112-122`) — iterates `returns` in original order with a `visited` set. Correct for first-visit MC. Confirm the `returns.reverse()` on line 110 is intentional (it is, but easy to break).
- **`self._alpha_current` decay** (`mc_agent.py:124-127`) — decay is applied after the episode update. Fine, but in `1/N` mode (`alpha is None`), `_alpha_current` stays `None` and never decays. The user-facing `--alpha_decay` flag has no effect in 1/N mode, which is silently confusing.
- **`np.random.seed` and `random.seed` both set globally** (`mc_agent.py:63-65`), only at construction. The agent uses both `random` and `numpy.random` internally, and if multiple agents are constructed in one process (which `run_experiments.py` does heavily) they overwrite each other's seeds. This is a real bug for reproducibility — VI doesn't have it, QL doesn't either (no RNG seeding at all in QL, which is itself questionable).
- **`max_episode_length` is owned by the agent** (`mc_agent.py:147`). It should be a trainer/CLI parameter, not agent state. Largely subsumed by Item 2.

### Suggested prompt

> "Audit `agents/mc_agent.py` for correctness against textbook on-policy first-visit Monte Carlo control. Focus on these specific concerns: (a) whether updating against `info['actual_action']` vs the agent's chosen action is correct for the stochastic environment in `world/environment.py`; (b) whether the convergence-by-reward-stability check at lines 166-177 ever fires during a typical sweep, or is effectively dead; (c) the interaction between `alpha=None` (1/N mode) and `--alpha_decay`; (d) the global RNG state pollution from `random.seed`/`np.random.seed` in `__init__` when multiple agents are constructed in one process. Produce a numbered list of confirmed issues and proposed fixes, but do not apply fixes — wait for confirmation."

---

## 5. Policy-disagreement metric and plots

**Dependencies: Item 1 (trainer modules), Item 2 (uniform `values`/`policy` interface).**

**Verdict: Agree. Three deliverables — one scalar (already exists), one spatial plot, one per-episode curve.**

### Current state

`run_experiments.py:148-165` already computes `_policy_difference(vi_policy, agent)` — fraction of states where the learned greedy action disagrees with the VI policy. Stored in the CSV as `policy_difference_from_optimal` (`run_experiments.py:131`). What's missing:

- Only computed in the sweep, not in single-agent runs from `train.py`.
- Scalar at end-of-training only — no per-episode trace.
- `utils/rl_plots.py` has no visualisation for it (neither spatial nor curve-based).

### Why the per-episode curve is worth adding

I previously argued against it on cost grounds. That was wrong: with ~200 reachable states on `A1_grid`, computing `argmax` per state per episode is microseconds, and recording every 10 episodes (rather than every 1) makes the cost completely negligible. The curve is independently useful: `avg_reward` and `policy_difference_from_optimal` are **not the same signal** — under stochastic transitions a slightly-wrong policy can still achieve near-optimal reward, and an "almost-converged" policy can have very different reward. `results/plots_from_example/algorithm_comparison.png` is the right reference shape: one panel per metric, all algorithms overlaid, smoothed.

### The three deliverables

1. **Spatial heatmap** — `plot_policy_disagreement(grid, optimal_policy, learned_policy, agent_start_pos=None) -> (fig, ax)` in `utils/rl_plots.py`. Reuses `_draw_grid_background`; marks disagreeing cells (red overlay or X) over the existing grid. Closely related to `plot_policy` — start by copying it. Saved as `*_policy_diff.png` next to `*_path.png` from the per-run artifact saver.

2. **Per-episode disagreement curve** — recorded as a new metric `policy_diff` in `TrainingHistory.metrics`, sampled every N episodes (recommend N=10 for QL/MC, N=1 for VI which has only ~200 sweeps anyway). Once it's a metric in the history dict, `plot_algorithm_comparison` and `plot_hyperparameter_comparison` show it for free with no changes to the plotting code — they iterate `metrics` and plot whatever is there. Two integration points:
   - Trainers (`agents/trainers/q_learning.py`, `agents/trainers/mc.py`) gain an optional `optimal_policy: dict | None = None` parameter. When provided, the training loop records `policy_diff` alongside `avg_reward`. When `None`, it's omitted from the history. VI's trainer doesn't need this — VI **is** the optimal policy.
   - The sweep already trains and caches a VI agent per `(grid, sigma, gamma)` in `_vi_cache`. Pass that VI's `.policy` into the QL/MC trainers when running the sweep. The single-agent CLI does the same behind a `--compare_optimal` flag.

3. **End-of-training scalar** — already exists in the sweep CSV; expose it in the single-agent CLI's evaluation summary when `--compare_optimal` is set. Trivial.

### Subtlety to flag in the prompt

The disagreement metric should be computed over **VI's known reachable states** (`vi.policy.keys()`), not the learned agent's `q_table.keys()` or `Q.keys()`. The latter grows over training as new states are visited, which would make the denominator move — early-training disagreement would look artificially low because few states are even known yet. `_policy_difference` in `run_experiments.py:151` already does it the right way; preserve that convention.

### Suggested prompt

> "Add policy-disagreement-from-optimal as both a per-episode learning curve and an end-of-training spatial heatmap. Four concrete changes:
>
> (a) In `agents/trainers/q_learning.py` and `agents/trainers/mc.py`, add an optional `optimal_policy: dict | None = None` parameter. When provided, the training loop computes `_policy_difference(optimal_policy, agent)` every 10 episodes (every iteration for VI) and records it as `policy_diff` in `TrainingHistory.metrics`. Compute the metric over `optimal_policy.keys()`, not the agent's own state-keys — match the existing convention in `run_experiments.py:151-165`.
>
> (b) Add `plot_policy_disagreement(grid, optimal_policy, learned_policy, title='', agent_start_pos=None) -> (fig, ax)` to `utils/rl_plots.py`. Reuse `_draw_grid_background`; mark disagreeing cells with a red overlay or X. Use `plot_policy` as the structural starting point.
>
> (c) Wire it into the per-run artifact saver: when `--compare_optimal` is set on the single-agent CLI, train a VI agent once with the same `sigma`/`gamma`/`reward_fn`, pass its `.policy` into the chosen trainer, and emit a `*_policy_diff.png` (the spatial heatmap) alongside `*_path.png`. Also include the end-of-training disagreement scalar in the evaluation summary.
>
> (d) In `run_experiments.py`, pass the cached VI policy from `_get_vi_agent(...).policy` into the QL/MC trainer calls so the per-episode curve is recorded automatically. No changes needed to the existing plotting code — `plot_hyperparameter_comparison` already iterates `metrics` and will pick up `policy_diff` as a new row. Update `TRAINING_CURVE_GROUPS` in `run_experiments.py:89-97` so that `policy_diff` is plotted alongside `avg_reward` for the groups where it's most informative (sigma, gamma at minimum).
>
> Do not change the existing `policy_difference_from_optimal` CSV column or `_policy_difference` function — they stay as-is. Reference image for the desired curve shape: `results/plots_from_example/algorithm_comparison.png`."

---

## 6. Use shared plotting helpers in `run_experiments.py`

**Dependencies: Item 1 (cleaner experiments file), Item 2 (uniform `values`/`policy` interface).**

**Verdict: Three concrete additions, not a structural change. The original bullet was right in spirit but underspecified.**

The sweep already uses `plot_hyperparameter_comparison` (called at `run_experiments.py:353` and `:394`). What's missing is described below in priority order.

### 6a. Use overlay (not column grid) when only one algorithm is involved

`_save_training_curve_plots` (`run_experiments.py:319-364`) always renders each condition as a separate column via `plot_hyperparameter_comparison`. For groups where `algo_filter` restricts to a **single** algorithm (`mc_ep_len` today, but the rule generalises), this wastes layout: each column has one line, and the eye has to jump between columns to compare conditions that should be on the same axes.

The fix is a one-place branch in `_save_training_curve_plots`:

- If, for a given `(group, grid_stem, algo)` combination, the group contains **one algorithm only** (either because `algo_filter` is a single-element list, or because the group naturally pertains to one algo), build a `histories` dict mapping **condition label → TrainingHistory** and pass it to `plot_algorithm_comparison` instead of `plot_hyperparameter_comparison`. `plot_algorithm_comparison` already overlays multiple histories on the same axes — that's exactly what's wanted here.
- Otherwise, keep the existing `plot_hyperparameter_comparison` 2-column layout.

This requires no changes to `plot_algorithm_comparison` itself — it doesn't care whether the labels are algorithm names or condition names; it just renders one labelled curve per entry in `histories`.

**Reference shape:** `results/plots_from_example/algorithm_comparison.png` is the format wanted for `mc_ep_len`, with the two condition curves where the three algorithm curves currently are.

### 6b. Emit cross-algorithm overlay figures per `(group, grid)`

`plot_algorithm_comparison` is used in `docs/examples/rl_plots_example.py:300-314` but **not** in `run_experiments.py`. For multi-algorithm groups (`sigma`, `gamma`, `alpha`, `epsilon`, schedules), the report benefits from an extra figure that overlays VI/QL/MC on the same axes per metric, in addition to the existing per-algorithm-per-condition curves. One figure per `(group, grid)`.

VI has no per-episode `avg_reward` so it would only appear on the `delta_v` row (if included) or be omitted from the reward row — the existing example handles this cleanly.

### 6c. Emit value+policy PNGs per `(experiment, algorithm, grid)`

The sweep currently saves only curves and CSVs. The value+policy plot is arguably the most important artifact for the report and the example already shows how (`rl_plots_example.py:281-296`). One `plot_value_and_policy` PNG per `(experiment, algorithm, grid)`, written into the group folder.

After Item 2 lands, the three agents all expose `values`/`policy` as attributes, so no adapter is needed.

### Out of scope for this item

- Do **not** restructure `EXP_GROUPS`, `EXPERIMENTS`, or `TRAINING_CURVE_GROUPS`. The declarative sweep spec is the good part of the file.
- Do **not** change `_save_vi_convergence_plots` — VI's `delta_v` plot is correct as-is.
- Do **not** add new CSV columns.

### Suggested prompt

> "Three additions to `run_experiments.py`, no other changes:
>
> (a) In `_save_training_curve_plots`, when the group is restricted to a single algorithm (either via a one-element `algo_filter` or because only one algo has data for the group), render the conditions as overlaid curves on a single axes using `plot_algorithm_comparison` from `utils/rl_plots.py` instead of `plot_hyperparameter_comparison`. Build a `histories` dict mapping condition label → `TrainingHistory` and pass it directly — `plot_algorithm_comparison` doesn't care that the keys are condition labels rather than algorithm names. For multi-algo groups, keep the existing `plot_hyperparameter_comparison` behaviour unchanged. Reference shape for the single-algo case: `results/plots_from_example/algorithm_comparison.png`.
>
> (b) For each `(group, grid)` in multi-algorithm groups, also emit a cross-algorithm overlay figure using `plot_algorithm_comparison`, matching the call style in `docs/examples/rl_plots_example.py:300-314`. Save as `{grid_stem}_algo_overlay.png` in the group folder.
>
> (c) For each `(experiment, algorithm, grid)` triple, save a `plot_value_and_policy` PNG into the group folder, following `docs/examples/rl_plots_example.py:281-296`. After Item 2, all three agents expose `values`/`policy` as attributes — no adapter needed. Skip cleanly if either map is empty.
>
> Do not modify `EXPERIMENTS`, `EXP_GROUPS`, `TRAINING_CURVE_GROUPS`, the CSV schema, or `_save_vi_convergence_plots`."
