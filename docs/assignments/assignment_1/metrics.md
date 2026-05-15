# Non-problem-specific RL metrics — quick reference

Context: tabular RL on a gridworld (VI / MC / Q-learning). Goal: pick a small, defensible set of metrics that are standard in the literature and not tied to one MDP.

## What you already have (with comments)

| Your metric | Standard name | Comment |
|---|---|---|
| Difference from optimal policy | Value error `‖V^π − V*‖_∞` and/or policy disagreement `Pr_s[π(s) ≠ π*(s)]` | Pin down which one. Both are reported in tabular work; they answer different questions (quality of values vs. quality of greedy actions). |
| Total discounted reward (eval) | Expected return `J(π) = E[Σ γ^t r_t]` under the greedy policy | This is the de-facto "final performance" metric. |
| Success rate | Termination/goal-reach rate over evaluation episodes | Standard for episodic / goal-conditioned tasks. |
| Total discounted reward during training | Online return / learning curve | This is *not* the same as eval return — it includes exploration cost. Treat as a separate metric. |

## Suggested additions (ranked by ROI for your setting)

### 1. Cumulative regret  `R_T = Σ_t (V*(s_0) − G_t)`
Single number that captures *both* learning speed and final performance. Standard in tabular/PAC-MDP theory and exploration papers.

- Auer, Cesa-Bianchi, Fischer (2002), *Finite-time analysis of the multiarmed bandit problem*.
- Jaksch, Ortner, Auer (2010), *Near-optimal regret bounds for reinforcement learning* (UCRL2), JMLR.
- Osband, Van Roy, Russo, Wen (2019), *Deep exploration via randomized value functions*, JMLR.

### 2. Sample efficiency / sample complexity
Number of episodes (or environment steps) until the agent reaches an ε-optimal policy (e.g. `‖V^π − V*‖_∞ ≤ ε`). Lets you compare MC vs. Q-learning fairly when both eventually converge.

- Kakade (2003), *On the sample complexity of reinforcement learning*, PhD thesis, UCL.
- Strehl, Li, Wiewiora, Langford, Littman (2006), *PAC model-free reinforcement learning*, ICML.
- Sutton & Barto (2018), *Reinforcement Learning: An Introduction*, 2nd ed., §6 (used informally throughout).

### 3. Bellman residual / value error vs. iteration  `‖T V_k − V_k‖_∞`
The natural convergence diagnostic for VI and a useful debugging plot for TD methods. Already implicit in your VI stopping criterion — just plot it.

- Sutton & Barto (2018), §4 (DP) and §6 (TD).
- Bertsekas (2019), *Reinforcement Learning and Optimal Control*, vol. I, §2.

### 4. Statistically robust reporting across seeds (IQM + bootstrap CIs)
Mean ± std over 3 seeds is no longer considered acceptable in the literature. Use the interquartile mean and stratified bootstrap CIs; report at least 5–10 seeds. This is cheap for tabular methods.

- Henderson, Islam, Bachman, Pineau, Precup, Meger (2018), *Deep reinforcement learning that matters*, AAAI. — demonstrates seed sensitivity.
- Agarwal, Schwarzer, Castro, Courville, Bellemare (2021), *Deep reinforcement learning at the edge of the statistical precipice*, NeurIPS (Outstanding Paper). — IQM, performance profiles, probability of improvement. Tooling: `rliable`.

### 5. Wall-clock time / iterations to convergence
Cheap, fair, and lets you make the obvious point that VI is fast per sweep but needs the model, while MC/Q-learning need many episodes. Often reported in tabular benchmarks.

- Henderson et al. (2018), AAAI — argues for compute reporting alongside performance.

### 6. Episode length (steps-to-goal)
Natural secondary metric for shortest-path-style grids: among policies that succeed, which is fastest? Sutton & Barto report this in the classic cliff-walking and windy-gridworld examples.

- Sutton & Barto (2018), §6.5–6.6.

## What I'd actually report in the assignment

Minimum viable, defensible set:

1. **Value error `‖V^π − V*‖_∞`** vs. iterations/episodes — convergence.
2. **Online return (training learning curve)** with IQM and 95% bootstrap CI over ≥5 seeds — learning behavior.
3. **Eval return of the greedy policy** at the end — final quality.
4. **Success rate + episode length** at eval — task-level interpretation.
5. **Cumulative regret** as a single-number summary across the run — combines (2) and (3).

Drop anything beyond this unless it makes a concrete point about a configuration (e.g. ε for Q-learning, first-visit vs. every-visit MC).

## Things I would *not* bother with here

- **KL between successive policies / policy entropy** — useful for PPO/SAC, not tabular.
- **Human-normalized score** — Atari-specific (Mnih et al. 2015).
- **State-visitation entropy / coverage** — only interesting if exploration is your research question.
- **Area under the learning curve (AULC)** — redundant with regret in your setting.

## References (compact)

- Auer, P., Cesa-Bianchi, N., Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. *Machine Learning*, 47.
- Jaksch, T., Ortner, R., Auer, P. (2010). Near-optimal regret bounds for reinforcement learning. *JMLR*, 11.
- Kakade, S. (2003). *On the sample complexity of reinforcement learning*. PhD thesis, UCL.
- Strehl, A., Li, L., Wiewiora, E., Langford, J., Littman, M. (2006). PAC model-free reinforcement learning. *ICML*.
- Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., Meger, D. (2018). Deep reinforcement learning that matters. *AAAI*.
- Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A., Bellemare, M. (2021). Deep reinforcement learning at the edge of the statistical precipice. *NeurIPS*.
- Osband, I., Van Roy, B., Russo, D., Wen, Z. (2019). Deep exploration via randomized value functions. *JMLR*, 20.
- Sutton, R., Barto, A. (2018). *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press.
- Bertsekas, D. (2019). *Reinforcement Learning and Optimal Control*, vol. I. Athena Scientific.
