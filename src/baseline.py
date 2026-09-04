"""Fixed-time baseline controller (Section 8 of the proposal).

Cycles through the 4 phases on a fixed 15-second schedule (3 simulation
steps of delta_time=5s each) regardless of real-time traffic state. This
is a non-learning policy with the identical action interface as the DQN
agent, so evaluate.py can run both through the same evaluation loop with
zero special-casing — that shared code path is what makes the comparison
"under identical conditions" rather than two separately-written scripts
that could silently diverge.
"""
from __future__ import annotations


class FixedTimeController:
    def __init__(self, num_phases: int = 4, steps_per_phase: int = 3):
        """steps_per_phase=3 at delta_time=5s => 15s per phase, matching the
        proposal's Section 8 baseline exactly."""
        self.num_phases = num_phases
        self.steps_per_phase = steps_per_phase
        self._step_count = 0

    def select_action(self, state, env_step: int, deterministic: bool = True) -> int:
        """Signature matches DQNAgent.select_action so evaluate.py can treat
        both policies polymorphically. `state` and `env_step` are unused —
        the whole point of this baseline is that it ignores traffic state."""
        phase = (self._step_count // self.steps_per_phase) % self.num_phases
        self._step_count += 1
        return phase

    def reset(self) -> None:
        """Must be called at the start of every episode, or the phase clock
        drifts across episodes and the baseline stops being a clean fixed
        15s-per-phase cycle from episode start."""
        self._step_count = 0
