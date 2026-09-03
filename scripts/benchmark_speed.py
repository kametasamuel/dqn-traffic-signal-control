"""Times N real episodes on your actual machine and extrapolates the full
run duration. Run this right after scripts/smoke_test.py passes, BEFORE
committing to a training schedule — the per-step cost is dominated by
TraCI overhead, which varies a lot by machine/OS and cannot be reliably
estimated from first principles.

Usage:
    python scripts/benchmark_speed.py --episodes 5
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import DQNAgent
from src.env_wrapper import DomainRandomizedTrainEnv
from src.utils import load_config, set_global_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--episodes", type=int, default=5,
                         help="How many real episodes to time (more = better estimate, but costs real time itself)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = 0
    set_global_seed(seed)

    env_cfg = cfg["env"]
    driver = DomainRandomizedTrainEnv(env_cfg, env_cfg["train_route_files"], seed=seed)
    agent = DQNAgent(cfg["agent"], seed=seed)

    episode_times = []
    print(f"Timing {args.episodes} real episode(s) at num_seconds={env_cfg['num_seconds']}, "
          f"delta_time={env_cfg['delta_time']} ({env_cfg['num_seconds'] // env_cfg['delta_time']} steps/episode)...\n")

    global_step = 0
    for ep in range(args.episodes):
        t0 = time.time()
        env, route_file = driver.next_episode_env()
        obs, _ = env.reset()
        done = False
        n_steps = 0
        while not done:
            action = agent.select_action(obs, env_step=global_step, deterministic=False)
            prev_obs = obs
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.store(prev_obs, action, reward, obs, done)
            agent.learn_step()
            global_step += 1
            n_steps += 1
        dt = time.time() - t0
        episode_times.append(dt)
        print(f"  episode {ep+1}/{args.episodes}: {dt:.1f}s ({n_steps} steps, {dt/n_steps*1000:.1f}ms/step)")

    driver.close()

    mean_ep_time = sum(episode_times) / len(episode_times)
    episodes_per_seed = cfg["training"]["episodes_per_seed"]
    n_seeds = len(cfg["training"]["seeds"])

    seed_hours = mean_ep_time * episodes_per_seed / 3600
    total_hours = seed_hours * n_seeds

    print(f"\nMean episode time: {mean_ep_time:.1f}s")
    print(f"Extrapolated: {episodes_per_seed} episodes/seed -> {seed_hours:.2f} hours/seed")
    print(f"Extrapolated: {n_seeds} seeds (sequential) -> {total_hours:.2f} hours total")
    print(f"\nIf this is too long: cut configs/default.yaml -> training.episodes_per_seed. "
          f"Do NOT cut training.seeds — see proposal Section 6 / module brief.")


if __name__ == "__main__":
    main()
