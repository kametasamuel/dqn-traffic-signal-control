"""Q-network: MLP mapping the 21-dim state to 4 action-values.

Architecture fixed by the proposal (Section 5): 21 -> 64 -> 64 -> 4,
ReLU on hidden layers, linear output. Kept as a plain nn.Module rather than
anything fancier — the assessed contribution here is the MDP/environment/
reward/evaluation design, not network novelty.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))  # linear output: raw Q-values
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
