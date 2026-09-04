"""Per-seed metric breakdown — one row per (policy, seed), not pooled."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

df = pd.read_csv("logs/eval_per_episode.csv")

metrics = ["avg_waiting_time", "avg_queue_length", "avg_speed", "cumulative_reward"]

per_seed = (
    df.groupby(["policy", "seed"])[metrics]
    .mean()
    .round(2)
    .sort_values(["policy", "seed"])
)

print("Per-seed means (each row = one trained agent/baseline instance, averaged over its 30 eval episodes):\n")
print(per_seed.to_string())

print("\n--- Spread check: does one seed dominate the pooled std? ---")
for metric in metrics:
    print(f"\n{metric}:")
    for policy in ["dqn", "baseline"]:
        vals = per_seed.loc[policy, metric]
        print(f"  {policy:8s} seed means: {dict(vals)}  "
              f"(range: {vals.max() - vals.min():.2f})")
