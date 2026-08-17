"""Lightweight metric tracking for training loops - structured enough to
plot or export later, without pulling in a dependency (TensorBoard,
wandb, ...) for what a training loop this size actually needs.
"""

from __future__ import annotations

import csv


class History:
    """Records named metrics (loss, accuracy, gradient norm, learning
    rate, ...) against a step number over the course of training.

    Deliberately just an in-memory dict of lists rather than anything
    fancier - a training loop calls `history.log(step, loss=..., lr=...)`
    once per step/epoch, and `History` does no aggregation or plotting
    itself. Multiple metrics don't need to be logged at the same steps
    (e.g. train loss every step, val loss only every eval_interval) -
    each metric keeps its own (step, value) pairs.
    """

    def __init__(self) -> None:
        self.records: dict[str, list[tuple[int, float]]] = {}

    def log(self, step: int, **metrics: float) -> None:
        """Record one or more metric values at the given step.

        Example: `history.log(step, train_loss=0.42, lr=3e-4)`.
        """
        for name, value in metrics.items():
            self.records.setdefault(name, []).append((step, float(value)))

    def steps(self, name: str) -> list[int]:
        """The steps at which `name` was logged, in order."""
        return [step for step, _ in self.records.get(name, [])]

    def values(self, name: str) -> list[float]:
        """The logged values for `name`, in the same order as `steps(name)`."""
        return [value for _, value in self.records.get(name, [])]

    def last(self, name: str) -> float | None:
        """The most recently logged value for `name`, or None if never logged."""
        record = self.records.get(name)
        return record[-1][1] if record else None

    def to_csv(self, path: str) -> None:
        """Write every metric to a single CSV with columns (step, metric, value).

        A long/tidy format (one row per (step, metric) pair) rather than
        one column per metric, since different metrics can be logged at
        different steps (see the class docstring) and a wide format
        would need to pad those gaps with something.
        """
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "metric", "value"])
            for name, record in self.records.items():
                for step, value in record:
                    writer.writerow([step, name, value])
