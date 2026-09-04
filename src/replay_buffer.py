"""Fixed-capacity experience replay buffer (Section 5: capacity 15,000).

A plain ring buffer over numpy arrays — avoids the overhead of a deque of
Python tuples at this capacity, and makes batch sampling a single fancy-index
op rather than a Python-level loop.
"""
from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, rng: np.random.Generator):
        self.capacity = capacity
        self.rng = rng
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)  # truncation, not just termination — see note in agent.py
        self._ptr = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(self, state, action, reward, next_state, done) -> None:
        i = self._ptr
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = float(done)
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = self.rng.integers(0, self._size, size=batch_size)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )
