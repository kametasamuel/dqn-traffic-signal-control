"""Single entry point for training. Trains one DQN agent per seed, with
identical hyperparameters across seeds (module brief: "a different
configuration per seed invalidates the comparison") and domain-randomized
training routes (proposal Section 6).

Usage:
    python train.py --config configs/default.yaml --seeds 7 13 42
    python train.py --config configs/default.yaml --seeds 7 13 42 --episodes 40  # compute-limited: cut episodes, not seeds
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.agent import DQNAgent
from src.env_wrapper import DomainRandomizedTrainEnv
from src.logger import EpisodeLogger
from src.utils import load_config, set_global_seed


def train_one_seed(cfg: dict, seed: int) -> None:
    set_global_seed(seed)

    env_cfg = cfg["env"]
    agent_cfg = cfg["agent"]
    train_cfg = cfg["training"]
    log_cfg = cfg["logging"]

    episodes = train_cfg["episodes_per_seed"]
    if train_cfg["domain_randomization"]:
        route_files = env_cfg["train_route_files"]          # all three route files, round-robin
    else:
        route_files = env_cfg["train_route_files"][:1]      # first route only (single-distribution ablation)

    driver = DomainRandomizedTrainEnv(env_cfg, route_files, seed=seed)
    agent = DQNAgent(agent_cfg, seed=seed)

    log_path = Path(log_cfg["log_dir"]) / f"train_seed{seed}.csv"
    logger = EpisodeLogger(log_path, fieldnames=["episode", "route_file", "episode_reward", "mean_loss", "wall_time_s"])

    global_step = 0
    print(f"[seed {seed}] starting training: {episodes} episodes across {len(route_files)} route file(s)")

    for ep in range(episodes):
        t0 = time.time()
        env, route_file = driver.next_episode_env()
        obs, _ = env.reset()

        episode_reward = 0.0
        losses = []
        done = False

        while not done:
            action = agent.select_action(obs, env_step=global_step, deterministic=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store(obs, action, reward, next_obs, done)
            loss = agent.learn_step()
            if loss is not None:
                losses.append(loss)

            obs = next_obs
            episode_reward += reward
            global_step += 1

        mean_loss = sum(losses) / len(losses) if losses else float("nan")
        logger.log({
            "episode": ep,
            "route_file": Path(route_file).name,
            "episode_reward": episode_reward,
            "mean_loss": mean_loss,
            "wall_time_s": time.time() - t0,
        })
        print(f"[seed {seed}] episode {ep+1}/{episodes} | route={Path(route_file).name} "
              f"| reward={episode_reward:.2f} | eps={agent.epsilon(global_step):.3f}")

    driver.close()

    model_dir = Path(log_cfg["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"dqn_seed{seed}.pt"
    agent.save(str(model_path))
    print(f"[seed {seed}] done. weights saved to {model_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                         help="Overrides configs/default.yaml training.seeds if given")
    parser.add_argument("--episodes", type=int, default=None,
                         help="Overrides episodes_per_seed. Use THIS to cut compute, not --seeds.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seeds is not None:
        cfg["training"]["seeds"] = args.seeds
    if args.episodes is not None:
        cfg["training"]["episodes_per_seed"] = args.episodes

    seeds = cfg["training"]["seeds"]
    print(f"Training seeds: {seeds} (hyperparameters held constant across all of them)")

    for seed in seeds:
        train_one_seed(cfg, seed)


if __name__ == "__main__":
    main()
