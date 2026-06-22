# Batch Changes Plan

Status: **DRAFT — awaiting answers to the open questions in §8 before implementation.**

This document describes a batch of changes to the deep-RL training stack plus a
SLURM experiment matrix. It is intentionally detailed and step-by-step, mapped
1:1 to the requested requirements. Nothing is implemented yet.

---

## 0. Context / current state (verified in code)

- Entry point: `train_deep.py` builds env + agent + `Trainer`, trains, loads the
  best checkpoint, runs **one** greedy rollout (`_run_policy_rollout`), then
  writes artifacts via `utils/artifacts.save_deep_rl_run_artifacts`.
- `Trainer` (`training/trainer.py`):
  - Periodic eval every `eval_interval` (`evaluate()` → `eval/mean_reward`,
    `eval/success_rate`, ...). `best.pt` saved on `eval/mean_reward` improvement.
  - `_maybe_log_rollout(episode)` is called at every `log_interval` and renders a
    rollout of the **current** agent via `viz_fn`, saving a PNG to
    `<out_dir>/rollouts/ep_*.png` **and** logging it to W&B.
  - `_log()` already references optional keys `"collisions"` and `"success"`, but
    **nothing currently populates `"collisions"` into `episode_metrics`** → not in
    `history.json`. (The terminal output showing `collisions=...` comes from a
    stale binary; the committed trainer does not write it.)
- Env tracks `world_stats["total_collisions"]` per episode (reset every
  `reset()`), available on both `ContinuousEnvironment` and `MinimalEnvironment`.
- Board rendering today lives in three places:
  1. `visualize_random_agent.py` (`draw_grid`/`plot_env`) — **bottom-left origin**
     (`ax.set_ylim(0, cols)` ascending). Used for the per-episode W&B rollout PNG.
  2. `utils/artifacts._write_policy_rollout_plot` → `utils/rl_plots._configure_grid_axes`
     — **already top-left origin** (`origin="upper"`, `set_ylim(n_rows-0.5, -0.5)`).
  3. `utils/artifacts._ROLLOUT_HTML_TEMPLATE` (SVG viewer) — **bottom-left** style
     `viewBox`/coords (y increases downward in SVG, but the grid is drawn with the
     raw mapping; needs explicit verification/flip).
- Package manager: **uv** (`uv.lock` present, no `poetry.lock`). All commands use
  `uv run python ...`.
- Grids present in `grid_configs/`: `simple_cave_grid.npy`, `A1_grid.npy`,
  `big_spaces_cave.npy` (⚠ untracked in git), `realistic_super_hard_cave.npy`.
- Reference run for disk sizing: `results/dueling-dqn_20260622_015311/` (3000
  episodes): total 12 MB, of which `rollouts/` = 9.6 MB (300 PNGs).

---

## 1. Visualization: top-left origin (flip vertically)

**Goal:** every board rendering uses the top-left corner as `x=0, y=0`
(y increases downward), consistent with the numpy grid indexing.

**Files & steps:**

1. `visualize_random_agent.py` → `draw_grid()`:
   - Change `ax.set_ylim(0, cols)` to `ax.set_ylim(cols, 0)` (invert y so row 0 is
     at the top). Keep `set_xlim(0, rows)`.
   - Verify the rectangle placement and the path/start/end markers still align
     after the inversion (they share the same axis, so inverting the limit is
     sufficient; no per-point transform needed).
2. `utils/artifacts.py` → `_ROLLOUT_HTML_TEMPLATE` (SVG viewer):
   - Confirm the rendered orientation. SVG y already increases downward, but the
     background loops over `data.grid[col][row]` placing cell `(col,row)` at
     `(col-0.5, row-0.5)`. Ensure the displayed orientation matches numpy
     `grid[row][col]` top-left. If it is currently transposed/flipped, adjust the
     `viewBox` and the `drawBackground` index mapping so row 0 renders at the top.
   - Action: add a short JS comment documenting the chosen axis convention.
3. `utils/rl_plots.py` → already `origin="upper"`. **Verify only**, no change
   expected. Add a one-line comment if it stays.
4. Decide whether axis labels should read `x` (horizontal) / `y` (vertical, down).
   Keep `xlabel="x"`, `ylabel="y"` and ensure `y` visually points down.

**Acceptance:** start cell renders in the top-left region for a grid whose
START_CELL is near index `(0,0)`; the three renderers agree visually.

---

## 2. Record collisions in training history

**Goal:** per-episode collision count is persisted in `history.json` /
`metrics.csv`.

**Files & steps:**

1. `training/trainer.py`, in `train()` after the per-episode loop, when building
   `episode_metrics` (around the block that sets `rollout/episode_reward`):
   - Read the env counter and add:
     `episode_metrics["rollout/collisions"] = float(self.env.world_stats.get("total_collisions", 0.0))`
   - Use the `rollout/` namespace for consistency with `rollout/episode_reward`.
2. `_log()` optional field: update the existing tuple so it reads the new key:
   change `("collisions", "collisions", ".0f")` →
   `("collisions", "rollout/collisions", ".0f")`. (Otherwise the terminal column
   stays blank.)
3. `_train_external()` (A3C path): A3C workers run their own envs, so
   `self.env.world_stats` is not meaningful there. Only populate collisions if the
   agent's per-episode `report` includes a collision count; otherwise skip
   (documented no-op for A3C). DQN/DDQN — the target agents for these experiments
   — go through the standard `train()` path, so they are covered.

**Acceptance:** `history.json` rows contain `rollout/collisions`; terminal log
prints `collisions=N`.

### 2b. Record per-episode wall-clock time

**Goal:** persist how long each training episode took (negligible overhead — two
`time.perf_counter()` reads per episode).

**Files & steps:**

1. `training/trainer.py` `train()`: capture `ep_start = time.perf_counter()` at the
   top of the episode loop, and after the episode set
   `episode_metrics["rollout/episode_time_s"] = time.perf_counter() - ep_start`.
2. Mirror the same in `_train_external()` if the A3C report does not already
   provide a duration (optional; DQN/DDQN go through `train()`).

**Acceptance:** `history.json` rows contain `rollout/episode_time_s`.

---

## 3. Final evaluation with the best checkpoint, greedy, X seeded runs

**Goal:** after training, load the best checkpoint and run the final greedy
evaluation **X times**, each with a different seed. When `X > 1`, the policy
rollout artifacts contain a **list** of X rollouts.

**New CLI arg (`train_deep.py`):**
- `--final-eval-runs` (int, default `1`, dest `final_eval_runs`): number of greedy
  evaluation rollouts at the end of training, each with a distinct seed.
  (Named distinctly from the existing `--eval-episodes`, which controls the
  *periodic* eval cadence and is unchanged.)

**Files & steps:**

1. `train_deep.py` `parse_args()`: add `--final-eval-runs`.
2. `train_deep.py` `main()`:
   - Keep the existing "load best checkpoint" step (`_best_checkpoint_path` →
     `agent.load_checkpoint`).
   - Replace the single `_run_policy_rollout(...)` call with a loop producing a
     list `rollouts = [...]` of length `final_eval_runs`, each using
     `seed = args.seed + 20_000 + i` (both for env construction and `env.reset`).
     `_run_policy_rollout` must accept a per-run seed offset (refactor its
     hard-coded `args.seed + 20_000`).
   - Compute aggregate final-eval metrics over the X rollouts (mean/std reward,
     success rate, mean steps) and write them into `evaluation_summary.txt` and,
     when W&B is on, into `wandb.run.summary`.
3. `utils/artifacts.save_deep_rl_run_artifacts`:
   - Accept `rollouts: list[dict] | None` (or keep `rollout` and branch on
     list vs dict for backwards compatibility).
   - `policy_rollout.json`: when `X == 1` keep the current single-object schema;
     when `X > 1` write a JSON **list** of rollouts.
   - **`.png` (single combined file, DECIDED):** overlay all X rollout paths on
     one grid render, each path drawn semi-transparently (e.g. `alpha≈0.35`) with
     a distinct color; start/end markers per run. Title shows aggregate stats
     (mean reward, success rate over X). For `X == 1` this collapses to the
     current single opaque path.
   - **`.html` (single combined file, DECIDED):** the SVG viewer draws **all** X
     paths as faint static polylines in the background, and adds a small run
     selector (dropdown or prev/next) that picks which rollout the animated agent
     steps through. Requires extending `_html_rollout_payload` to carry a list of
     runs and updating the embedded JS to index into the selected run. For
     `X == 1` it behaves exactly as today.

**Acceptance:** with `--final-eval-runs 10`, `policy_rollout.json` is a list of 10
seeded greedy rollouts from `best.pt`; summary reports aggregate success rate.

---

## 4. W&B-only rollout visualizations on a configurable interval

**Goal:** stop rendering a rollout at every `log_interval` and stop persisting
per-episode rollout PNGs locally. Add `--wandb-visualisations N` that, **only when
`--wandb` is set**, renders a rollout of the **best-so-far** model every `N`
episodes and logs it to W&B (no local file kept). Default `N = 100`.

**New CLI arg (`train_deep.py`):**
- `--wandb-visualisations` (int, default `100`, dest `wandb_viz_interval`):
  episode interval for W&B rollout images. No effect unless `--wandb` is set.

**Files & steps:**

1. `train_deep.py` `parse_args()`: add `--wandb-visualisations`. Decide fate of the
   existing `--no-visualize` / `--viz-out` flags (see §8 Q3). Proposed: keep
   `--no-visualize` as a global "disable W&B rollout images" switch; drop the
   local `rollouts/` directory entirely.
2. `train_deep.py` `main()`: remove creation of the `rollouts/` directory and the
   PNG-writing `viz_fn` that saves to disk. Instead pass the interval and a
   rendering callback that writes to a temporary path (e.g. `tempfile`), logs to
   W&B, then deletes the temp file.
3. `training/config.py`: add `wandb_viz_interval: int` to `TrainerConfig`
   (default 100). Wire from `_build_config`.
4. `training/trainer.py`:
   - Change `_maybe_log_rollout` to fire on `episode % wandb_viz_interval == 0`
     (independent of `log_interval`) and to **no-op unless W&B is active**.
   - Render the **best-so-far** policy, not the live one. Two options:
     a) load `best.pt` into a throwaway agent/clone before rendering, or
     b) cache a `state_dict` of the best model in `_maybe_save_best` and
        temporarily swap it into the agent for rendering, then restore.
     Proposed: (a) — load `best.pt` if it exists, else fall back to the current
     agent (early training, before any eval). Document the trade-off (small extra
     I/O every N episodes vs. keeping a clone in memory).
   - Ensure the rendered image is logged via `self._wandb.Image(...)` and the temp
     file is removed afterwards.
5. Remove the now-dead `rollouts/` references in `train_deep.py`.

**Acceptance:** no `rollouts/` directory is created; with `--wandb` and
`--wandb-visualisations 100`, a "viz/rollout" image of the best model appears in
W&B every 100 episodes; without `--wandb`, no rollout images are produced.

---

## 5. README / docs alignment

Per project rule (functional/interface changes must align with docs):

### 5b. Drop `metrics.csv` (redundant with `history.json`)

**Files & steps:**

1. `utils/artifacts.save_deep_rl_run_artifacts`: remove `out_dir / "metrics.csv"`
   from the `paths` list and stop calling `_write_metrics_csv`.
2. Delete the now-unused `_write_metrics_csv` helper (and the `csv` import if no
   longer used).
3. Update `README.md` / `docs/training_logger.md` to drop references to
   `metrics.csv`; point consumers at `history.json`.

**Acceptance:** runs no longer emit `metrics.csv`; `history.json` is the single
per-episode metrics source.

### 5c. Docs alignment

- Update `README.md` and `docs/training_logger.md` for: new CLI flags
  (`--final-eval-runs`, `--wandb-visualisations`), removal of the local
  `rollouts/` directory, the new `rollout/collisions` history key, and the
  top-left visualization convention.

---

## 6. SLURM experiment matrix

### 6.1 Shared settings (all experiments)
- Partition `gpu_mig`, `--reservation=terv92681`.
- 5 seeds × 4 grids × 2 agents (`dqn`, `ddqn`).
- 10000 training episodes each (`--episodes 10000`), `--env continuous`, `--wandb`.
- Grids: `simple_cave_grid.npy`, `A1_grid.npy`, `big_spaces_cave.npy`,
  `realistic_super_hard_cave.npy`.
- Output dirs separated per experiment with self-explanatory run names
  (style mirrors `scripts/simple_cave_grid.sh`):
  - `results/experiment_1/<grid>_<agent>_seed<seed>/`
  - `results/experiment_2/<grid>_<agent>_<sensors|no_sensors>_seed<seed>/`
  - `results/experiment_3/<grid>_<agent>_sigma<sigma>_seed<seed>/`

### 6.2 Run counts (matches the requested 240)
| Experiment | Variants | Runs | Final-eval runs |
|---|---|---|---|
| 1 — baseline | 1 (defaults) | 5×4×2×1 = **40** | 1 |
| 2 — sensors | 2 (sensors / `--no-sensors`) | 5×4×2×2 = **80** | 1 |
| 3 — stochasticity | 3 (`--sigma` 0.0 / 0.2 / 0.5) | 5×4×2×3 = **120** | 10 |
| **Total** | | **240** | |

### 6.3 Script layout (proposed — see §8 Q1)
- Three array scripts: `scripts/experiment_1.sh` (array `0-39`),
  `scripts/experiment_2.sh` (`0-79`), `scripts/experiment_3.sh` (`0-119`).
- Each script builds the full config list (seed × grid × agent × variant) and
  indexes it by `$SLURM_ARRAY_TASK_ID`, mirroring the `CONFIGS=(...)` +
  `read -r ... <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"` pattern in
  `scripts/simple_cave_grid.sh`.
- SBATCH resources for `gpu_mig`: 1 MIG slice per task (`--gpus-per-node=1`),
  `--cpus-per-task=4`, `--reservation=terv92681`. (4 CPUs is for node packing, not
  per-run speed — a single DQN/DDQN run is single-threaded, so extra cores would
  sit idle and just reduce how many MIG tasks pack onto the 36-core node.)
  `--time=08:00:00` (the SLURM wall-clock ceiling per array task) — generous
  headroom for a 10000-episode run; tasks that finish early release their MIG slice
  immediately, so over-provisioning costs nothing but slightly slower backfill.
  Must stay under the `gpu_mig` partition cap and fit inside the `terv92681`
  reservation window. `.out`/`.err` named per experiment + array id.
- Each task invokes:
  `uv run python train_deep.py --agent <a> --env continuous --grid <g> --seed <s>
  --episodes 10000 --wandb --wandb-group experiment_<n> --out-dir <dir>
  [--no-sensors] [--sigma <v>] --final-eval-runs <1|10>`.

### 6.4 Smoke script
- `scripts/smoke_experiments.sh`: runs the **same** matrix wiring but with
  `--episodes <small, e.g. 20>`, `--eval-interval 5`, `--final-eval-runs 1`, and
  ideally without `--wandb` (or a dedicated `smoke` group), launched as a small
  array (a handful of representative configs across the 3 experiments) so it can
  be checked in parallel quickly.
- Purpose: confirm each experiment's CLI wiring, dirs, and artifact writing run
  end-to-end before committing the full 240-run matrix.

---

## 7. Disk-space estimate

Derived from `results/dueling-dqn_20260622_015311/` (3000 episodes), **excluding**
the `rollouts/` directory (removed by §4) and `metrics.csv` (removed by §5b).

Per-episode-scaling file at 10000 episodes (×2 vs 5000-ep projection below), plus the two new fields
(`rollout/collisions`, `rollout/episode_time_s`):
- `history.json`: 1.60 MB @ 3000 ep → ~5.8 MB @ 10000 ep (linear in episodes)

Roughly constant per run:
- `best.pt` + `last.pt`: 0.31 + 0.31 = 0.62 MB
- `training_curves.png`: ~0.11 MB
- `config.json` + `evaluation_summary.txt`: ~0.005 MB
- final greedy rollout artifacts (single combined `.json`+`.png`+`.html`):
  - 1 run (exp 1/2): ~0.083 MB
  - 10 runs combined (exp 3): JSON list ~0.17 MB + combined png ~0.06 MB +
    combined html ~0.04 MB ≈ **~0.27 MB**

**Per run (1 final-eval rollout):** ≈ **6.6 MB**
**Per run (10 final-eval rollouts, exp 3):** ≈ **6.8 MB**

| Experiment | Runs | MB/run | Subtotal |
|---|---|---|---|
| 1 | 40 | 6.6 | ~264 MB |
| 2 | 80 | 6.6 | ~528 MB |
| 3 | 120 | 6.8 | ~816 MB |
| **Total** | **240** | | **≈ 1.6 GB** |

Plus SLURM `.out`/`.err` logs (~500 log lines/run) ≈ 15–20 MB total.

**Budget ≈ 1.6–1.7 GB** with filesystem overhead.

---

## 8. Decisions & remaining questions

**Decided (from review):**
1. **SLURM structure:** three array scripts (`experiment_1/2/3.sh`) + one smoke
   script.
2. **Multi-rollout artifacts:** single combined `.png` (all X paths, semi-
   transparent) and single combined `.html` (all paths faint + run selector);
   `policy_rollout.json` is a list when `X > 1`.
3. **Legacy viz flags:** drop the local `rollouts/` dir; `--no-visualize` becomes
   "disable W&B rollout images"; remove `--viz-out`.
4. **`big_spaces_cave.npy`:** commit it as part of this work.
5. **`--cpus-per-task=4`** per array task.
6. **`--final-eval-runs` default `1`** (only exp 3 passes `10`).
7. **Drop `metrics.csv`** (redundant with `history.json`).
8. **Record `rollout/episode_time_s`** per training episode.

**Still need confirmation before/at implementation:**
- **MIG slices per node** under reservation `terv92681` — used only to set the
  array throttle (`%N` concurrency cap), since CPUs are fixed at 4. If unknown, run
  without a throttle and let SLURM schedule by slice availability.
- **Per-task `--time=08:00:00`** (SLURM wall-clock ceiling) — generous headroom;
  finishing early releases the slice. Confirm it's under the `gpu_mig` partition
  cap and fits the reservation window.
