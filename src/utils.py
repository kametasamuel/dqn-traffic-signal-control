"""Shared utilities: seeding and config loading.

Kept deliberately tiny and dependency-light — this module is imported by
train.py, evaluate.py and every script, so a bug here silently invalidates
every seed's reproducibility.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_global_seed(seed: int) -> None:
    """Seed every source of randomness we control.

    NOTE: this does NOT seed SUMO's internal traffic-generation randomness —
    that is controlled separately via the `seed` kwarg passed to the SUMO-RL
    environment constructor in env_wrapper.py. Both must be set for a run to
    be reproducible, and both must be recorded in the log filename/header.
    """
    random.seed(seed)
    np.random.seed(seed)  # seeds the legacy global Generator; all RNG in this codebase uses
                          # np.random.default_rng(seed) (new-style Generator, independent of this)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)
