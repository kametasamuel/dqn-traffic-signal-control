"""Environment construction and domain randomization.

This is one of the pieces the module brief calls out as assessed original
work (env definition / wrapper, observation construction, termination
logic), so it is kept explicit rather than hidden behind SUMO-RL defaults
wherever a design choice is being made.

TODO (verify against your installed sumo-rl==1.4.5 before first run):
    The exact SumoEnvironment constructor kwargs below match the 1.4.x API
    as of the proposal's pilot run. If `pip show sumo-rl` gives a different
    version, diff the constructor signature and update accordingly — do not
    silently guess at defaults.
"""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
from sumo_rl import SumoEnvironment

# Reward function weights — match the MDP specification exactly.
# R(t) = -(ALPHA * mean_waiting_time(t) + BETA * mean_queue(t))
# BETA=0.0 so reward depends only on waiting time, which is the primary metric.
ALPHA = 1.0
BETA = 0.0


class TrafficSignalWrapper(gym.Wrapper):
    """Replaces SUMO-RL's default differential reward with the explicit equation
    from the MDP specification:

        R(t) = -(ALPHA * mean_waiting_time(t) + BETA * mean_queue(t))

    SUMO-RL's built-in default for SumoEnvironment is -(diff_waiting_time),
    the *change* in total waiting time per step — a differential signal that
    can be positive. Our reward is a level signal (always <= 0) that directly
    penalises the absolute waiting time, which is what the evaluation metrics
    measure. Using the same signal for training and evaluation is required for
    the comparison to be honest.

    Observation and action spaces are unchanged from the underlying env.
    """

    def step(self, action):
        obs, _default_reward, terminated, truncated, info = self.env.step(action)
        reward = -(ALPHA * info.get("system_mean_waiting_time", 0.0)
                   + BETA * info.get("system_total_stopped", 0.0))
        return obs, float(reward), terminated, truncated, info

    def __getattr__(self, name: str):
        # gymnasium.Wrapper (v0.29.1) does not define __getattr__, so attributes
        # like env.sumo (the raw TraCI connection used by logger.py) would raise
        # AttributeError without this proxy.
        return getattr(self.env, name)


def make_env(cfg: dict, route_file: str, seed: int, use_gui: bool | None = None):
    """Construct a single-agent SUMO-RL environment for one route file.

    single_agent=True is correct here: the network (Section 3) has exactly
    one intersection, so SUMO-RL exposes a plain Gymnasium Env rather than
    the multi-agent PettingZoo interface.
    """
    return TrafficSignalWrapper(SumoEnvironment(
        net_file=cfg["net_file"],
        route_file=route_file,
        use_gui=cfg["use_gui"] if use_gui is None else use_gui,
        num_seconds=cfg["num_seconds"],
        delta_time=cfg["delta_time"],
        yellow_time=cfg["yellow_time"],
        min_green=cfg["min_green"],
        single_agent=True,
        sumo_seed=seed,  # controls SUMO's own traffic-generation randomness — see utils.set_global_seed note
    ))


class DomainRandomizedTrainEnv:
    """Round-robins across the training route files episode-by-episode.

    Not a gymnasium.Env subclass on purpose: SUMO-RL environments must be
    freshly constructed (or reset with a route change, depending on version)
    per episode when the route file changes, so this is a thin episode-level
    driver rather than a wrapper around a single persistent env instance.
    Train.py calls `.next_episode_env()` once per episode.
    """

    def __init__(self, cfg: dict, route_files: list[str], seed: int):
        self.cfg = cfg
        self.route_files = route_files
        self.rng = np.random.default_rng(seed)
        self._base_seed = seed
        self._episode_count = 0
        self._current_env: SumoEnvironment | None = None

    def next_episode_env(self):
        """Close the previous episode's env (frees the SUMO/TraCI process)
        and build a new one on the next route file in the round-robin."""
        if self._current_env is not None:
            self._current_env.close()

        route_file = self.route_files[self._episode_count % len(self.route_files)]
        # Vary the SUMO seed per episode (deterministically, from the run
        # seed) so episodes on the same route aren't bit-identical, while
        # the whole round-robin sequence remains reproducible from
        # self._base_seed.
        episode_seed = self._base_seed * 1000 + self._episode_count
        self._current_env = make_env(self.cfg, route_file, seed=episode_seed)
        self._episode_count += 1
        return self._current_env, route_file

    def close(self):
        if self._current_env is not None:
            self._current_env.close()


def make_eval_env(cfg: dict, seed: int, use_gui: bool = False):
    """Held-out evaluation environment — always cfg['eval_route_file'],
    never one of the training routes. This is the genuine train/test split
    referenced in Section 6 of the proposal."""
    return make_env(cfg, cfg["eval_route_file"], seed=seed, use_gui=use_gui)
