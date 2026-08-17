from __future__ import annotations

import math

from nabla.optim.optimizer import Optimizer


class LRScheduler:
    """Base class for learning-rate schedules.

    A scheduler doesn't wrap or replace an Optimizer - it just mutates
    `optimizer.lr` in place after every `step()`, the same "external
    controller" relationship PyTorch's `lr_scheduler`s have to their
    optimizer. That works here because nabla's Optimizer has a single
    global `.lr` (no per-parameter-group learning rates), so a scheduler
    only ever needs to compute one number.

    Subclasses implement `get_lr(step)` to define the schedule; this base
    class handles calling it and writing the result back onto the
    optimizer, and remembers `base_lr` (the optimizer's lr at
    construction time) since most schedules are defined as a fraction of
    it rather than an absolute value.

    Args:
        optimizer: The optimizer whose `.lr` this schedule controls.
    """

    def __init__(self, optimizer: Optimizer) -> None:
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.step_count = 0

    def get_lr(self, step: int) -> float:
        """Return the learning rate for the given step. Must be overridden."""
        raise NotImplementedError

    def step(self) -> None:
        """Advance the schedule by one step and update optimizer.lr."""
        self.step_count += 1
        self.optimizer.lr = self.get_lr(self.step_count)


class WarmupCosineLR(LRScheduler):
    """Linear warmup followed by cosine decay - the schedule most
    Transformer training recipes use (e.g. "Attention Is All You Need",
    GPT-2/3): the learning rate ramps up linearly from 0 for
    `warmup_steps` steps (large gradients early in training, before the
    model has learned anything useful, are what warmup protects
    against), then decays smoothly along a cosine curve down to
    `min_lr` by `total_steps`.

    Args:
        optimizer: The optimizer whose `.lr` this schedule controls.
        warmup_steps: Number of steps to linearly ramp up from 0 to base_lr.
        total_steps: Total number of steps the schedule spans (warmup + decay).
        min_lr: The learning rate at and after total_steps.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
    ) -> None:
        super().__init__(optimizer)
        if warmup_steps > total_steps:
            raise ValueError(f"warmup_steps ({warmup_steps}) must not exceed total_steps ({total_steps}).")
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr

    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.base_lr * step / self.warmup_steps

        if step >= self.total_steps:
            return self.min_lr

        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine


class StepLR(LRScheduler):
    """Decay the learning rate by a factor of `gamma` every `step_size` steps.

    Simpler and less common in Transformer training than
    `WarmupCosineLR`, but a standard choice for CNNs (e.g.
    `examples/mnist_cnn.py`) where a smooth cosine schedule matters less.

    Args:
        optimizer: The optimizer whose `.lr` this schedule controls.
        step_size: Number of steps between each decay.
        gamma: Multiplicative decay factor applied every step_size steps.
    """

    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1) -> None:
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self, step: int) -> float:
        return self.base_lr * (self.gamma ** (step // self.step_size))
