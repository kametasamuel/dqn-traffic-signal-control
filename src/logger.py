"""Run logging and metric computation, shared between train.py and
evaluate.py so training curves and evaluation tables can never silently use
different metric code — the module brief requires evaluating the baseline
"using the same episodes, seeds and metric code" as the agent.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def compute_episode_metrics(step_infos: list[dict], vehicle_travel_times: list[float] | None = None) -> dict[str, float]:
    """Aggregate per-step SUMO-RL info dicts into episode-level metrics.

    vehicle_travel_times, if provided, is the list of completed-trip travel
    times (seconds) collected during the episode via TraCI directly (see
    collect_throughput_and_travel_time() below) — SUMO-RL's default info
    dict does not include these, so they are NOT silently zero/omitted;
    the proposal (Section 7) explicitly promises them for the full
    submission, so they must come from somewhere outside the library.
    """
    waiting_times = [s.get("system_total_waiting_time", np.nan) for s in step_infos]
    queue_lengths = [s.get("system_total_stopped", np.nan) for s in step_infos]
    speeds = [s.get("system_mean_speed", np.nan) for s in step_infos]

    metrics = {
        "avg_waiting_time": float(np.nanmean(waiting_times)),
        "avg_queue_length": float(np.nanmean(queue_lengths)),
        "avg_speed": float(np.nanmean(speeds)),
    }

    if vehicle_travel_times is not None:
        metrics["vehicles_processed"] = float(len(vehicle_travel_times))
        metrics["avg_travel_time"] = float(np.mean(vehicle_travel_times)) if vehicle_travel_times else 0.0
    else:
        metrics["vehicles_processed"] = float("nan")
        metrics["avg_travel_time"] = float("nan")

    return metrics


def collect_throughput_and_travel_time(sumo_conn, seen_departures: dict) -> list[float]:
    """Call once per environment step (after env.step()) with the raw TraCI
    connection (env.sumo) and a persistent dict tracking each vehicle's
    depart time. Returns the travel times of any vehicles that ARRIVED
    (completed their trip) this step.
    """
    now = sumo_conn.simulation.getTime()
    for vid in sumo_conn.simulation.getDepartedIDList():
        seen_departures[vid] = now

    completed = []
    for vid in sumo_conn.simulation.getArrivedIDList():
        depart_time = seen_departures.pop(vid, None)
        if depart_time is not None:
            completed.append(now - depart_time)
    return completed


class EpisodeLogger:
    """Appends one row per episode to a CSV, creating the header on first
    write. Used identically by train.py (training reward per episode) and
    evaluate.py (held-out metrics per episode) — same code path, per the
    module brief's evaluation-harness requirement.
    """

    def __init__(self, path: str | Path, fieldnames: list[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames
        if not self.path.exists():
            with open(self.path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    def log(self, row: dict) -> None:
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(row)


def aggregate_across_seeds(per_seed_values: list[float]) -> tuple[float, float]:
    """Mean and standard deviation across seeds. Every reported metric must
    go through this — a single number without a spread is not accepted."""
    arr = np.asarray(per_seed_values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0