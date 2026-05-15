# Assignment 1 — Requirements & Rubric Summary

**Course:** 2AMC15 – Reinforcement Learning in Practice (2025/2026, Q4)  
**Deadline:** 13/05/2026  
**Total points:** 40 (Report 30 + Code 10, with up to -5 deductions)

---

## Task

Implement three tabular RL algorithms, one from each category:

| Category | Options |
|----------|---------|
| Dynamic Programming | Policy Iteration **or** Value Iteration |
| Monte-Carlo | On-policy MC **or** Off-policy MC |
| Temporal Difference | SARSA **or** Q-Learning |

Any combination (8 possible) is acceptable.

---

## Experimental Requirements

- Use the provided simulation environment with at least **two grids** (must include `A1_grid.npy` with `agent_start_pos = [1, 12]`).
- Vary at least **two discount factors** (e.g., 0.6 and 0.9).
- Vary at least **two stochasticity values** (e.g., 0.02 and 0.5).
- MC methods: at least **two max episode lengths**.
- TD methods: at least **two learning rates** (constant or variable).
- Soft policies: at least **two epsilon values** (decay allowed but must be explained).
- Starting position must be the same across setups for fair comparison.
- The problem must be Markov; all three methods must reach optimal policies.
- Report **6 final experimental setups** comparing all three algorithms.
- Use proper **RL evaluation metrics** (not problem-specific ones like "total steps" or "failed moves").

---

## Suggested Reward Function

| Event | Reward |
|-------|--------|
| Each step (empty cell) | -1 |
| Action into obstacle (agent stays) | -1 (penalty for wasted action) |
| Reaching delivery destination | +10 |

---

## Deliverables

### 1. Report (PDF, submitted separately)

- Max 6 pages excluding references (no appendix).
- Must use the provided [LaTeX template](https://www.overleaf.com/read/hqbmhrtspspb).
- Must be **anonymous** (no names, group number, or identifying info).

### 2. Code (ZIP file)

- All code in `.py` format only (no notebooks/`.ipynb`).
- Must run error-free with basic steps.
- Clear README with instructions to reproduce results.
- `requirements.txt` for additional packages.
- Must be self-contained (no private files or external dependencies).
- Must be efficient (not require hours to train).
- Must be **anonymous**.

### 3. Dual Submission

Submit to both "Group Assignment 1" **and** the FeedbackFruits review assignment (same deadline). Missing the review submission forfeits all A1 points.

---

## Rubric

### A. Report (30 points)

| Section | Points | Criteria |
|---------|--------|----------|
| **Introduction** | 5 | States and motivates 3 chosen algorithms; concisely explains how each works; provides overall technical comparison. |
| **Experimental Setups** | 10 | Describes shared settings (environment, reward, defaults, grid); explains ≥6 setups with varied hyperparameters (must include A1_grid); justifies why each experiment matters; describes and justifies ≥2 performance metrics. |
| **Experimental Results** | 10 | Presents results for all 6 cases with clear visualizations; thorough analysis comparing algorithms across setups and metrics; detailed discussion of parameter effects on performance; presents best approach on A1_grid with optimal policy. |
| **Conclusions** | 5 | Conclusion per method with best configuration; take-home message including validity discussion and future improvements. |

### B. Code (10 points)

| Points | Criteria |
|--------|----------|
| 0 | No code, or same as template, or wrong format. |
| 2 | No clear README or code gives multiple errors. |
| 4 | Runs without errors but methods are incorrect/incomplete. |
| 6 | All methods correct and run, but results not fully reproducible. |
| 8 | Clear, correct, reproducible, but could be improved (slow, hard to configure). |
| **10** | Fully reproducible, well-documented, efficient, clean, standalone tool. |

### C. Deductions (up to -5)

| Deduction | Reason |
|-----------|--------|
| -2 | Submission not entirely anonymous. |
| -2 | Report does not follow provided LaTeX format. |
| -1 | PDF not submitted separately in "Group Assignment 1". |
