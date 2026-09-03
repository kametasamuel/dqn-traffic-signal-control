# Adaptive Traffic Signal Control with DQN

MSc/MPhil Data Science — DSCD614 Deep Reinforcement Learning Group Project
**Option DQN-1: Intelligent Traffic Signal Control**

A Deep Q-Network agent learns to allocate green time at a single SUMO
intersection based on lane-level density and queue occupancy, evaluated
against a fixed-time baseline controller. See `docs/proposal.pdf` for the
full MDP formulation and experimental protocol; this README covers how to
install, train, evaluate, and reproduce every figure in the report.

## 1. Installation

Requires Python 3.10+ and SUMO (Simulation of Urban MObility) installed
separately — SUMO is a system package, not a pip package.

```bash
# 1. Install SUMO (Linux; see https://sumo.dlr.de/docs/Installing/index.html
#    for macOS/Windows instructions)
sudo apt-get install sumo sumo-tools sumo-doc
export SUMO_HOME=/usr/share/sumo   # add to ~/.bashrc for persistence

# 2. Create a virtual environment and install pinned Python dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Verify the environment loads before doing anything else
python -c "import sumo_rl, gymnasium as gym; print('OK')"
```

**Verify before you build on top of this**: run `python scripts/smoke_test.py`.
It builds the environment, takes 5 random steps, and prints the observation/
action shapes. If this doesn't run, nothing downstream will either — fix
installation first per the module brief (installation difficulty is not
grounds for changing environment after the selection deadline, so this has
to work early).

## 2. Repository structure

```
traffic-dqn/
├── configs/
│   └── default.yaml        # all hyperparameters in one place (single source of truth)
├── src/
│   ├── env_wrapper.py      # environment construction, domain randomization, obs/action spec
│   ├── network.py          # Q-network (MLP)
│   ├── replay_buffer.py    # experience replay
│   ├── agent.py            # DQN agent: epsilon-greedy policy, learning update, target net
│   ├── baseline.py         # fixed-time controller (non-learning baseline)
│   ├── logger.py           # run logging: per-episode metrics to CSV, seeded run directories
│   └── utils.py            # global seeding, config loading
├── train.py                 # SINGLE ENTRY POINT: trains all seeds, writes logs + weights
├── evaluate.py               # loads trained weights, runs held-out evaluation + baseline, writes results table
├── scripts/
│   ├── smoke_test.py         # fast sanity check the environment + agent wire up correctly
│   └── make_figures.py       # regenerates every figure in the report from logs/
├── logs/                      # raw experiment logs (committed — figures must trace back here)
├── models/                    # saved weights for the seeds used in the report/demo
└── requirements.txt            # pinned exact versions
```

## 3. Training

```bash
python train.py --config configs/default.yaml --seeds 7 13 42
```

This is the single entry point referenced above: it round-robins training
episodes across the domain-randomization route files (Section 6 of the
proposal), trains one agent per seed with identical hyperparameters, and
writes:
- `logs/train_seed{N}.csv` — per-episode training reward
- `models/dqn_seed{N}.pt` — final weights for each seed

Training on `num_seconds=3600` episodes is slow. To reduce compute, cut
`--episodes` in the config — **do not** cut `--seeds`; see Section 6 of the
proposal for why (a compute-limited run reduces steps, not the number of
seeds needed to distinguish a real effect from noise).

## 4. Evaluation

```bash
python evaluate.py --config configs/default.yaml --seeds 7 13 42 --episodes 30
```

Runs each trained agent, deterministically (epsilon=0, greedy action
selection — stated explicitly in stdout and in the results CSV header), on
the held-out route (`single-intersection-gen.rou.xml`) for 30 episodes per
seed, and runs the identical evaluation on the fixed-time baseline for
comparison under identical conditions (same episodes, same seeds, same
metric code — `src/logger.py`'s `compute_metrics` is shared by both).
Writes `logs/eval_results.csv` with mean ± std across seeds for every
metric — no single number is reported without a spread.

## 5. Reproducing report figures

```bash
python scripts/make_figures.py
```

Regenerates every figure from the committed logs in `logs/` — nothing here
is a screenshot or a one-off notebook cell; if a figure changes, it changes
because a log changed.

## 6. Notes on the from-scratch pilot

The proposal's preliminary pilot (Section 9) used a from-scratch NumPy DQN
to validate the pipeline without a framework dependency. This repository is
the PyTorch reimplementation for the full submission — same MDP, same
hyperparameters (Section 5 of the proposal), GPU-capable. `logs/pilot/`
retains the original NumPy pilot logs for traceability to Figures 1–2 of
the proposal.
