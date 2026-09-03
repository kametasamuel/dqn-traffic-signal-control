"""DQN agent: epsilon-greedy behaviour policy + target-network learning update.

Every hyperparameter here is read from configs/default.yaml, not hardcoded —
this file should not need editing to run a different config.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.network import QNetwork
from src.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(self, cfg: dict, seed: int, device: str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)

        state_dim = cfg["state_dim"]
        action_dim = cfg["action_dim"]
        hidden = cfg["hidden_sizes"]

        self.q_net = QNetwork(state_dim, action_dim, hidden).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=cfg["learning_rate"])
        self.buffer = ReplayBuffer(cfg["buffer_capacity"], state_dim, self.rng)

        self.action_dim = action_dim
        self.gamma = cfg["gamma"]
        self.batch_size = cfg["batch_size"]
        self.target_update_every = cfg["target_update_every_steps"]
        self.min_replay_before_learning = cfg["min_replay_before_learning"]

        self.eps_start = cfg["epsilon_start"]
        self.eps_end = cfg["epsilon_end"]
        self.eps_decay_steps = cfg["epsilon_decay_steps"]

        self._train_steps = 0  # counts learning updates, drives target sync

    # ------------------------------------------------------------------ #
    # Behaviour policy
    # ------------------------------------------------------------------ #
    def epsilon(self, env_step: int) -> float:
        """Linear decay from eps_start to eps_end over eps_decay_steps env steps."""
        frac = min(1.0, env_step / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def select_action(self, state: np.ndarray, env_step: int, deterministic: bool = False) -> int:
        """deterministic=True is the evaluation-time setting (Section 6 of the
        proposal): exploration disabled, epsilon effectively 0."""
        if not deterministic and self.rng.random() < self.epsilon(env_step):
            return int(self.rng.integers(0, self.action_dim))
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_net(state_t)
            return int(torch.argmax(q_values, dim=1).item())

    # ------------------------------------------------------------------ #
    # Learning
    # ------------------------------------------------------------------ #
    def store(self, state, action, reward, next_state, done) -> None:
        self.buffer.add(state, action, reward, next_state, done)

    def can_learn(self) -> bool:
        return len(self.buffer) >= max(self.batch_size, self.min_replay_before_learning)

    def learn_step(self) -> float | None:
        """One gradient update from a sampled minibatch. Returns the loss, or
        None if the buffer isn't warm enough yet (caller should skip)."""
        if not self.can_learn():
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_t = torch.as_tensor(states, device=self.device)
        actions_t = torch.as_tensor(actions, device=self.device).long()
        rewards_t = torch.as_tensor(rewards, device=self.device)
        next_states_t = torch.as_tensor(next_states, device=self.device)
        dones_t = torch.as_tensor(dones, device=self.device)

        q_values = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_states_t).max(dim=1).values
            # dones here means truncation (Section 4.4: episodes truncate at
            # the time limit, no early terminal state) — bootstrapping is
            # still stopped at episode end so the buffer never mixes reward
            # across episode boundaries.
            targets = rewards_t + self.gamma * next_q_values * (1.0 - dones_t)

        loss = nn.functional.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._train_steps += 1
        if self._train_steps % self.target_update_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        torch.save(self.q_net.state_dict(), path)

    def load(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)
