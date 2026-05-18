"""Trainer for the Q-Learning agent."""

from __future__ import annotations

import numpy as np

from agents.learning_rates import build_lr_schedule
from agents.q_learning_agent import QLearningAgent
from agents.trainers.common import (
    OptimalActionSets,
    Position,
    RewardFunction,
    TrainConfig,
    _greedy_policy_from_q_table,
    build_episode_iter,
    build_episode_start_picker,
    build_logger,
    mean_tail,
    policy_disagreement_from_q_table,
    q_table_as_array,
    restore_eval_start,
    should_log,
    validate_log_interval,
)
from utils.plotting import TrainingHistory
from world import Environment

_DEFAULT_MAX_EPISODE_LENGTH = 500


def train(
    env: Environment,
    reward_fn: RewardFunction,
    cfg: TrainConfig,
    *,
    optimal_policy: OptimalActionSets | None = None,
    optimal_values: dict[Position, float] | None = None,
) -> tuple[QLearningAgent, TrainingHistory]:
    """Train a Q-learning agent on ``env`` and return the agent plus history.

    Records ``discounted_return`` (per-episode discounted return) and ``epsilon``
    (post-episode exploration rate) for downstream plotting. When
    ``optimal_policy`` is provided, also records ``policy_diff`` per
    episode — fraction of optimal-policy states the agent disagrees with.
    When ``optimal_values`` is provided, also records ``optimality_gap``
    per episode — ``V*(start_state) - episode_discounted_return``. The
    agent is switched to evaluation mode before returning so subsequent
    rollouts are greedy.
    """
    if cfg.ql_episodes is None:
        raise ValueError("TrainConfig.ql_episodes is required for Q-learning")
    validate_log_interval(cfg)

    # ------------------------------------------------------------------
    # Resolve run configuration
    # ------------------------------------------------------------------
    # The trainer owns environment-level choices such as start-state
    # sampling and episode length. The agent only receives the learning
    # hyperparameters it needs for action selection and Q-value updates.
    pick_episode_start = build_episode_start_picker(env, cfg)

    max_episode_length = (
        cfg.max_episode_length if cfg.max_episode_length is not None else _DEFAULT_MAX_EPISODE_LENGTH
    )

    # Learning-rate schedules are shared with MC. The schedule object
    # hides whether alpha is global per episode or state-action specific.
    lr_schedule = build_lr_schedule(
        cfg.lr_schedule,
        alpha=cfg.alpha if cfg.alpha is not None else 0.5,
        alpha_decay=cfg.alpha_decay if cfg.alpha_decay is not None else 0.999,
        alpha_min=cfg.alpha_min if cfg.alpha_min is not None else 0.05,
        visit_count_c=cfg.visit_count_c,
    )

    # ------------------------------------------------------------------
    # Build agent
    # ------------------------------------------------------------------
    # Q-learning updates after every transition, so the agent needs the
    # learning-rate schedule immediately during the interaction loop.
    agent = QLearningAgent(
        gamma=cfg.gamma,
        epsilon=cfg.epsilon if cfg.epsilon is not None else 1.0,
        epsilon_min=cfg.epsilon_min if cfg.epsilon_min is not None else 0.05,
        epsilon_decay=cfg.epsilon_decay if cfg.epsilon_decay is not None else 0.995,
        decaying_epsilon=not cfg.fixed_epsilon,
        n_actions=4,
        q_init=cfg.q_init,
        q_init_noise=cfg.q_init_noise,
        random_seed=cfg.random_seed,
        lr_schedule=lr_schedule,
    )

    # ------------------------------------------------------------------
    # Metrics collected during training
    # ------------------------------------------------------------------
    # These lists stay aligned by episode index and are converted into a
    # TrainingHistory at the end for plotting, CSV summaries, and W&B.
    episode_discounted_rewards: list[float] = []
    episode_undiscounted_rewards: list[float] = []
    episode_deltas: list[float] = []
    episode_epsilons: list[float] = []
    episode_alphas: list[float] = []
    episode_alpha_mins: list[float] = []
    episode_alpha_maxs: list[float] = []
    episode_policy_diffs: list[float] = []
    episode_optimality_gaps: list[float] = []

    schedule_has_global_rate = lr_schedule.get_global_rate() is not None

    # ------------------------------------------------------------------
    # Logging and stopping state
    # ------------------------------------------------------------------
    # ``build_episode_iter`` returns either a plain range or a tqdm range,
    # depending on whether a live logger is active.
    logger, log_interval = build_logger(cfg, cfg.ql_episodes)
    episode_iter = build_episode_iter(cfg.ql_episodes, logger, "Q-learning")

    # Early stopping watches the tied-greedy policy derived from the Q-table.
    # Using action sets instead of a single argmax avoids treating numerical
    # ties as policy changes.
    prev_greedy_policy: dict[Position, frozenset[int]] | None = None
    stable_streak = 0
    stopped_early = False
    stop_episode = cfg.ql_episodes
    last_logged_episode = 0

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for episode_idx in episode_iter:
        # Reset the environment for a new episode. With exploring starts
        # enabled this may be a sampled empty cell; otherwise it is the
        # fixed evaluation start.
        state = env.reset(agent_start_pos=pick_episode_start())
        episode_start = state
        env.reward_fn = reward_fn
        agent.start_episode()

        # Episode-level accumulators are kept outside the agent so all
        # trainers report metrics in the same format.
        ep_discounted_reward = 0.0
        ep_undiscounted_reward = 0.0
        ep_delta = 0.0
        gamma_power = 1.0

        # Q-learning is one-step bootstrapping: after every environment
        # transition, update Q(s, a) using the observed reward and the
        # current estimate of the next state's best action value.
        for _ in range(max_episode_length):
            action = agent.take_action(state)
            next_state, reward, terminated, _info = env.step(action)

            # Capture the value before/after the update so ``delta_q``
            # measures the largest Q-table movement in this episode.
            previous_state = agent._last_state
            old_q_value = (
                float(agent.q_table[previous_state][action])
                if previous_state is not None
                else 0.0
            )
            agent.update(next_state, reward, action, terminated=terminated)
            if previous_state is not None:
                new_q_value = float(agent.q_table[previous_state][action])
                ep_delta = max(ep_delta, abs(new_q_value - old_q_value))
            state = next_state
            ep_discounted_reward += gamma_power * reward
            ep_undiscounted_reward += reward
            gamma_power *= cfg.gamma
            if terminated:
                break

        # End-of-episode bookkeeping decays epsilon/alpha schedules inside
        # the agent after all one-step updates for this episode are done.
        agent.end_episode()

        # Store raw per-episode metrics first. Windowed means are computed
        # only when logging, so the saved history keeps the full resolution.
        episode_discounted_rewards.append(ep_discounted_reward)
        episode_undiscounted_rewards.append(ep_undiscounted_reward)
        episode_deltas.append(ep_delta)
        episode_epsilons.append(agent.epsilon)
        if agent.last_episode_mean_alpha is not None:
            episode_alphas.append(agent.last_episode_mean_alpha)
        if not schedule_has_global_rate:
            if agent.last_episode_alpha_min is not None:
                episode_alpha_mins.append(agent.last_episode_alpha_min)
            if agent.last_episode_alpha_max is not None:
                episode_alpha_maxs.append(agent.last_episode_alpha_max)

        if optimal_policy is not None:
            episode_policy_diffs.append(
                policy_disagreement_from_q_table(optimal_policy, agent.q_table)
            )

        if optimal_values is not None:
            v_star = optimal_values.get(episode_start, float("nan"))
            episode_optimality_gaps.append(v_star - ep_discounted_reward)

        # Policy stability is checked after every episode, using the current
        # Q-table rather than the final greedy policy object.
        current_greedy = _greedy_policy_from_q_table(agent.q_table)
        if prev_greedy_policy is not None and current_greedy == prev_greedy_policy:
            stable_streak += 1
        else:
            stable_streak = 0
        prev_greedy_policy = current_greedy

        episode_num = episode_idx + 1
        if logger is not None and should_log(episode_num, log_interval, cfg.ql_episodes):
            # Build live value/policy snapshots for console or W&B logging.
            # This avoids switching the agent into evaluation mode mid-run.
            live_values = {s: float(np.max(q)) for s, q in agent.q_table.items()}
            live_policy = {s: int(np.argmax(q)) for s, q in agent.q_table.items()}
            mean_discounted = mean_tail(episode_discounted_rewards, log_interval)
            mean_delta = mean_tail(episode_deltas, log_interval)
            mean_pdiff = (
                mean_tail(episode_policy_diffs, log_interval)
                if optimal_policy is not None
                else None
            )
            mean_gap = (
                mean_tail(episode_optimality_gaps, log_interval)
                if optimal_values is not None
                else None
            )
            logger.log_iteration(
                episode=episode_num,
                q_values=q_table_as_array(agent.q_table),
                q_delta=ep_delta,
                mean_q_delta=mean_delta,
                converged=False,
                current_alpha=agent.last_episode_mean_alpha,
                current_epsilon=agent.epsilon,
                policy_diff=mean_pdiff,
                discounted_return=mean_discounted,
                optimality_gap=mean_gap,
                env_grid=env.grid,
                optimal_policy=optimal_policy,
                agent_start_pos=cfg.start_pos,
                agent_values=live_values,
                agent_policy=live_policy,
            )
            last_logged_episode = episode_num

        if (
            cfg.policy_stable_patience is not None
            and stable_streak >= cfg.policy_stable_patience
        ):
            stopped_early = True
            stop_episode = episode_num
            # Emit one final log so dashboards capture the stop state
            # when the patience cutoff lands between configured log
            # intervals; skip when we just logged this same episode.
            if logger is not None and last_logged_episode != episode_num:
                live_values = {s: float(np.max(q)) for s, q in agent.q_table.items()}
                live_policy = {s: int(np.argmax(q)) for s, q in agent.q_table.items()}
                logger.log_iteration(
                    episode=episode_num,
                    q_values=q_table_as_array(agent.q_table),
                    q_delta=ep_delta,
                    mean_q_delta=mean_tail(episode_deltas, log_interval),
                    converged=True,
                    current_alpha=agent.last_episode_mean_alpha,
                    current_epsilon=agent.epsilon,
                    policy_diff=(
                        mean_tail(episode_policy_diffs, log_interval)
                        if optimal_policy is not None
                        else None
                    ),
                    discounted_return=mean_tail(episode_discounted_rewards, log_interval),
                    optimality_gap=(
                        mean_tail(episode_optimality_gaps, log_interval)
                        if optimal_values is not None
                        else None
                    ),
                    env_grid=env.grid,
                    optimal_policy=optimal_policy,
                    agent_start_pos=cfg.start_pos,
                    agent_values=live_values,
                    agent_policy=live_policy,
                )
            break

    if not stopped_early:
        stop_episode = len(episode_discounted_rewards)

    # Restore the original evaluation start after possible exploring starts,
    # and make future calls to ``take_action`` greedy rather than epsilon-greedy.
    restore_eval_start(env, cfg)
    agent.set_eval_mode()

    # ------------------------------------------------------------------
    # Build training history
    # ------------------------------------------------------------------
    # Optional traces are only included when their corresponding comparison
    # or schedule produced data.
    metrics: dict[str, list[float]] = {
        "discounted_return": episode_discounted_rewards,
        "undiscounted_return": episode_undiscounted_rewards,
        "delta_q": episode_deltas,
        "epsilon": episode_epsilons,
    }
    if episode_alphas:
        metrics["alpha"] = episode_alphas
    if episode_alpha_mins:
        metrics["alpha_min"] = episode_alpha_mins
    if episode_alpha_maxs:
        metrics["alpha_max"] = episode_alpha_maxs
    if optimal_policy is not None:
        metrics["policy_diff"] = episode_policy_diffs
    if optimal_values is not None:
        metrics["optimality_gap"] = episode_optimality_gaps

    history = TrainingHistory(
        episodes=list(range(1, len(episode_discounted_rewards) + 1)),
        metrics=metrics,
        hyperparams={
            "lr_schedule": lr_schedule.describe(),
            "epsilon": cfg.epsilon,
            "epsilon_decay": cfg.epsilon_decay,
            "gamma": cfg.gamma,
            "sigma": cfg.sigma,
            "max_episode_length": max_episode_length,
            "q_init": cfg.q_init,
            "q_init_noise": cfg.q_init_noise,
            "log_interval": log_interval,
            "log_q_table": cfg.log_q_table,
            "exploring_starts": cfg.exploring_starts,
            "policy_stable_patience": cfg.policy_stable_patience,
        },
        metadata={
            "stopped_early": stopped_early,
            "stop_episode": stop_episode,
        },
    )
    return agent, history
