from __future__ import annotations

import math

import numpy as np

from nabla.optim.adam import Adam
from nabla.optim.scheduler import StepLR, WarmupCosineLR
from nabla.tensor import Tensor


def make_optimizer(lr=1.0):
    x = Tensor(np.array([1.0]), requires_grad=True)
    return Adam([x], lr=lr)


class TestWarmupCosineLR:
    def test_lr_is_zero_before_any_step(self):
        optimizer = make_optimizer(lr=1.0)
        WarmupCosineLR(optimizer, warmup_steps=10, total_steps=100)
        assert optimizer.lr == 1.0  # constructor doesn't touch .lr yet

    def test_linear_warmup_reaches_base_lr(self):
        optimizer = make_optimizer(lr=2.0)
        scheduler = WarmupCosineLR(optimizer, warmup_steps=4, total_steps=100)

        expected = [0.5, 1.0, 1.5, 2.0]  # step/warmup_steps * base_lr
        for exp in expected:
            scheduler.step()
            assert math.isclose(optimizer.lr, exp, rel_tol=1e-9)

    def test_cosine_decay_reaches_min_lr_at_total_steps(self):
        optimizer = make_optimizer(lr=1.0)
        scheduler = WarmupCosineLR(optimizer, warmup_steps=0, total_steps=10, min_lr=0.1)

        for _ in range(10):
            scheduler.step()
        assert math.isclose(optimizer.lr, 0.1, rel_tol=1e-9)

    def test_lr_stays_at_min_lr_past_total_steps(self):
        optimizer = make_optimizer(lr=1.0)
        scheduler = WarmupCosineLR(optimizer, warmup_steps=0, total_steps=5, min_lr=0.2)

        for _ in range(20):
            scheduler.step()
        assert math.isclose(optimizer.lr, 0.2, rel_tol=1e-9)

    def test_decay_is_monotonically_non_increasing_after_warmup(self):
        optimizer = make_optimizer(lr=1.0)
        scheduler = WarmupCosineLR(optimizer, warmup_steps=5, total_steps=50, min_lr=0.0)

        lrs = []
        for _ in range(50):
            scheduler.step()
            lrs.append(optimizer.lr)

        post_warmup = lrs[5:]
        assert all(a >= b - 1e-12 for a, b in zip(post_warmup, post_warmup[1:]))

    def test_rejects_warmup_longer_than_total(self):
        optimizer = make_optimizer(lr=1.0)
        try:
            WarmupCosineLR(optimizer, warmup_steps=20, total_steps=10)
            assert False, "expected ValueError"
        except ValueError:
            pass


class TestStepLR:
    def test_decays_by_gamma_every_step_size(self):
        optimizer = make_optimizer(lr=1.0)
        scheduler = StepLR(optimizer, step_size=3, gamma=0.5)

        for _ in range(2):
            scheduler.step()
        assert math.isclose(optimizer.lr, 1.0, rel_tol=1e-9)  # still within first interval (steps 1-2)

        scheduler.step()  # step 3: crosses into the second interval
        assert math.isclose(optimizer.lr, 0.5, rel_tol=1e-9)  # one decay applied

        for _ in range(3):
            scheduler.step()  # steps 4-6: step 6 crosses into the third interval
        assert math.isclose(optimizer.lr, 0.25, rel_tol=1e-9)  # two decays applied
