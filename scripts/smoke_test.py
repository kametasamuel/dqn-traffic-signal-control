"""Fast sanity check: does the environment build, and do observation/action
shapes match what Section 4 of the proposal claims (21-dim state,
Discrete(4) action)? Run this before writing a single line of training code
against a new machine — per the module brief, installation difficulty is
not grounds for changing environment after the deadline, so this needs to
pass early.

Usage: python scripts/smoke_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env_wrapper import make_env
from src.utils import load_config


def main():
    cfg = load_config("configs/default.yaml")
    env_cfg = cfg["env"]

    print("Building environment on first training route...")
    env = make_env(env_cfg, env_cfg["train_route_files"][0], seed=0, use_gui=False)

    obs, info = env.reset()
    print(f"Observation shape: {obs.shape} (expected: (21,))")
    print(f"Action space: {env.action_space} (expected: Discrete(4))")
    assert obs.shape == (21,), "State dimension does not match proposal Section 4.1 — investigate before proceeding"
    assert env.action_space.n == 4, "Action space does not match proposal Section 4.2 — investigate before proceeding"

    print("Taking 5 random steps...")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  step {i}: action={action} reward={reward:.3f} done={terminated or truncated}")

    env.close()
    print("\nSmoke test passed — environment matches the proposal's MDP spec.")


if __name__ == "__main__":
    main()
