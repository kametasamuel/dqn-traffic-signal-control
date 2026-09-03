"""Evaluation harness. Runs the trained agent AND the fixed-time baseline
through the identical evaluation loop (same held-out route, same episode
count, same seeds, same metric code) so the comparison is honest by
construction rather than by discipline.

Usage:
    python evaluate.py --config configs/default.yaml --seeds 7 13 42 --episodes 30
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.agent import DQNAgent
from src.baseline import FixedTimeController
from src.env_wrapper import make_eval_env
from src.logger import EpisodeLogger, aggregate_across_seeds, compute_episode_metrics, collect_throughput_and_travel_time
from src.utils import load_config, set_global_seed


def run_policy_episode(env, policy, deterministic: bool, episode_seed: int | None = None) -> dict:
    """Runs one episode with any policy exposing select_action(state, env_step,
    deterministic) -> int — both DQNAgent and FixedTimeController match this
    interface, which is what lets this single function serve both.

    episode_seed is passed to env.reset() so that each evaluation episode sees a
    different SUMO traffic realisation. Without this, sumo-rl v1.4.5 reuses the
    fixed sumo_seed on every reset(), producing bit-identical episodes and a
    within-seed std of zero — which defeats the purpose of running 30 episodes.
    """
    obs, _ = env.reset(seed=episode_seed)
    if hasattr(policy, "reset"):
        policy.reset()

    step_infos = []
    completed_travel_times = []
    seen_departures: dict = {}
    episode_reward = 0.0
    env_step = 0
    done = False

    while not done:
        action = policy.select_action(obs, env_step=env_step, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_infos.append(info)
        completed_travel_times += collect_throughput_and_travel_time(env.sumo, seen_departures)
        episode_reward += reward
        env_step += 1

    metrics = compute_episode_metrics(step_infos, completed_travel_times)
    metrics["cumulative_reward"] = episode_reward
    return metrics


def evaluate_policy(cfg: dict, policy, seed: int, n_episodes: int, tag: str, logger: EpisodeLogger) -> dict[str, list[float]]:
    env = make_eval_env(cfg["env"], seed=seed, use_gui=False)
    per_metric: dict[str, list[float]] = {"avg_waiting_time": [], "avg_queue_length": [], "avg_speed": [], "avg_travel_time": [], "vehicles_processed": [], "cumulative_reward": []}

    for ep in range(n_episodes):
        ep_seed = seed * 1000 + ep  # unique but reproducible seed per episode
        m = run_policy_episode(env, policy, deterministic=True, episode_seed=ep_seed)  # exploration disabled at evaluation
        for k in per_metric:
            per_metric[k].append(m[k])
        row = {"policy": tag, "seed": seed, "episode": ep, **m}
        logger.log(row)

    env.close()
    return per_metric


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Defaults to configs/default.yaml training.seeds if not given")
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seeds is None:
        args.seeds = cfg["training"]["seeds"]
    n_episodes = args.episodes or cfg["evaluation"]["episodes_per_seed"]

    per_episode_log = EpisodeLogger(
        Path(cfg["logging"]["log_dir"]) / "eval_per_episode.csv",
        fieldnames=["policy", "seed", "episode", "avg_waiting_time", "avg_queue_length", "avg_speed", "avg_travel_time", "vehicles_processed", "cumulative_reward"],
    )

    agent_seed_means: dict[str, list[float]] = {"avg_waiting_time": [], "avg_queue_length": [], "avg_speed": [], "avg_travel_time": [], "vehicles_processed": [], "cumulative_reward": []}
    baseline_seed_means: dict[str, list[float]] = {k: [] for k in agent_seed_means}

    model_dir = Path(cfg["logging"]["model_dir"])

    for seed in args.seeds:
        set_global_seed(seed)

        agent = DQNAgent(cfg["agent"], seed=seed)
        agent.load(str(model_dir / f"dqn_seed{seed}.pt"))
        agent_metrics = evaluate_policy(cfg, agent, seed, n_episodes, tag="dqn", logger=per_episode_log)
        for k, v in agent_metrics.items():
            agent_seed_means[k].append(float(np.mean(v)))

        baseline = FixedTimeController(
            num_phases=cfg["baseline"]["num_phases"],
            steps_per_phase=cfg["baseline"]["phase_duration_seconds"] // cfg["env"]["delta_time"],
        )
        baseline_metrics = evaluate_policy(cfg, baseline, seed, n_episodes, tag="baseline", logger=per_episode_log)
        for k, v in baseline_metrics.items():
            baseline_seed_means[k].append(float(np.mean(v)))

        print(f"[seed {seed}] dqn cumulative_reward={np.mean(agent_metrics['cumulative_reward']):.2f} "
              f"| baseline cumulative_reward={np.mean(baseline_metrics['cumulative_reward']):.2f}")

    summary_path = Path(cfg["logging"]["log_dir"]) / "eval_summary.csv"
    summary_logger = EpisodeLogger(summary_path, fieldnames=["policy", "metric", "mean", "std", "n_seeds"])
    print("\n=== Evaluation summary (deterministic policy, "
          f"{n_episodes} episodes/seed, seeds={args.seeds}) ===")
    for tag, seed_means in [("dqn", agent_seed_means), ("baseline", baseline_seed_means)]:
        for metric, values in seed_means.items():
            mean, std = aggregate_across_seeds(values)
            summary_logger.log({"policy": tag, "metric": metric, "mean": mean, "std": std, "n_seeds": len(values)})
            print(f"  {tag:8s} {metric:20s} {mean:10.3f} +/- {std:.3f}  (n={len(values)} seeds)")

    if len(args.seeds) < 3:
        print("\nWARNING: fewer than 3 seeds — per the module brief, this cannot fully "
              "support a claim of difference from the baseline. State this in the report.")


if __name__ == "__main__":
    main()