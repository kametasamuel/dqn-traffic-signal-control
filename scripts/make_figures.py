"""Regenerates every figure in the report from logs/. Per the module brief:
"Figures that cannot be traced to committed logs are not credited" — so
every plot the report uses must come out of this file, not a one-off
notebook cell that never got committed.

Usage: python scripts/make_figures.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # traffic-dqn/ regardless of CWD
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd

LOG_DIR = ROOT / "logs"
FIG_DIR = ROOT / "logs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_training_curves(seeds: list[int], rolling_window: int = 7):
    """Mean training reward across seeds, with spread — analogous to the
    pilot's Figure 1, but across the full 3+ seed protocol rather than one
    seed per condition."""
    fig, ax = plt.subplots(figsize=(7, 4))

    all_runs = []
    for seed in seeds:
        path = LOG_DIR / f"train_seed{seed}.csv"
        if not path.exists():
            print(f"  (skipping seed {seed}: {path} not found — run train.py first)")
            continue
        df = pd.read_csv(path)
        df["reward_smoothed"] = df["episode_reward"].rolling(rolling_window, min_periods=1).mean()
        all_runs.append(df.set_index("episode")["reward_smoothed"])
        ax.plot(df["episode"], df["reward_smoothed"], alpha=0.3, label=f"seed {seed}")

    if all_runs:
        combined = pd.concat(all_runs, axis=1)
        mean = combined.mean(axis=1)
        std = combined.std(axis=1)
        ax.plot(mean.index, mean, color="black", linewidth=2, label="mean across seeds")
        ax.fill_between(mean.index, mean - std, mean + std, color="black", alpha=0.15, label="+/- 1 std")

    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Training reward ({rolling_window}-episode rolling mean)")
    ax.set_title("Training reward across seeds")
    ax.legend()
    fig.tight_layout()
    out = FIG_DIR / "training_curves.png"
    fig.savefig(out, dpi=150)
    print(f"  wrote {out}")


def plot_eval_comparison():
    """Bar chart: DQN vs baseline, mean +/- std across seeds, for every
    metric — the held-out evaluation table (analogous to pilot Figure 2)."""
    path = LOG_DIR / "eval_summary.csv"
    if not path.exists():
        print(f"  (skipping: {path} not found — run evaluate.py first)")
        return

    df = pd.read_csv(path)
    metrics = df["metric"].unique()

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        sub = df[df["metric"] == metric]
        ax.bar(sub["policy"], sub["mean"], yerr=sub["std"], capsize=5)
        ax.set_title(metric)

    fig.suptitle("DQN vs. fixed-time baseline (held-out route, mean +/- std across seeds)")
    fig.tight_layout()
    out = FIG_DIR / "eval_comparison.png"
    fig.savefig(out, dpi=150)
    print(f"  wrote {out}")


def main():
    import yaml
    cfg = yaml.safe_load(open(ROOT / "configs" / "default.yaml"))
    seeds = cfg["training"]["seeds"]

    print("Generating training curves...")
    plot_training_curves(seeds)

    print("Generating evaluation comparison...")
    plot_eval_comparison()


if __name__ == "__main__":
    main()
