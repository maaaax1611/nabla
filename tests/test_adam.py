from __future__ import annotations

import numpy as np

from nabla.optim.adam import Adam
from nabla.tensor import Tensor


class TestAdam:
    def test_single_step_matches_manual_calculation(self):
        # x starts at 1.0 with a gradient of 2.0
        x = Tensor(np.array([1.0]), requires_grad=True)
        x.grad = np.array([2.0])

        lr, beta1, beta2, eps = 0.1, 0.9, 0.999, 1e-8
        optimizer = Adam([x], lr=lr, betas=(beta1, beta2), eps=eps)
        optimizer.step()

        # Manual calculation for t=1, starting from m=0, v=0
        grad = 2.0
        m = (1 - beta1) * grad
        v = (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**1)
        v_hat = v / (1 - beta2**1)
        expected = 1.0 - lr * m_hat / (np.sqrt(v_hat) + eps)

        assert np.allclose(x.data, np.array([expected]))

    def test_step_moves_parameter_towards_lower_loss(self):
        # positive gradient should decrease the parameter value
        x = Tensor(np.array([5.0]), requires_grad=True)
        x.grad = np.array([1.0])

        optimizer = Adam([x], lr=0.1)
        optimizer.step()

        assert x.data[0] < 5.0

    def test_multiple_steps_reduce_loss(self):
        # simple quadratic: loss = x^2, gradient = 2x
        x = Tensor(np.array([10.0]), requires_grad=True)
        optimizer = Adam([x], lr=0.5)

        for _ in range(50):
            x.grad = 2 * x.data
            optimizer.step()

        assert np.abs(x.data[0]) < 1.0

    def test_step_counter_increments(self):
        x = Tensor(np.array([1.0]), requires_grad=True)
        x.grad = np.array([1.0])
        optimizer = Adam([x], lr=0.1)

        assert optimizer.t == 0
        optimizer.step()
        assert optimizer.t == 1
        optimizer.step()
        assert optimizer.t == 2

    def test_skips_parameters_without_gradient(self):
        x = Tensor(np.array([1.0]), requires_grad=True)
        x.grad = None
        optimizer = Adam([x], lr=0.1)

        optimizer.step()

        assert np.array_equal(x.data, np.array([1.0]))

    def test_zero_grad_resets_gradients(self):
        x = Tensor(np.array([1.0]), requires_grad=True)
        y = Tensor(np.array([2.0]), requires_grad=True)
        x.grad = np.array([1.0])
        y.grad = np.array([2.0])

        optimizer = Adam([x, y], lr=0.1)
        optimizer.zero_grad()

        assert x.grad is None
        assert y.grad is None

    def test_independent_moment_estimates_per_parameter(self):
        # opposite gradient signs must move parameters in opposite directions
        x = Tensor(np.array([1.0]), requires_grad=True)
        y = Tensor(np.array([1.0]), requires_grad=True)
        x.grad = np.array([1.0])
        y.grad = np.array([-1.0])

        optimizer = Adam([x, y], lr=0.1)
        optimizer.step()

        assert x.data[0] < 1.0
        assert y.data[0] > 1.0

    def test_moment_magnitude_affects_later_steps(self):
        # after several steps, larger gradients should accumulate larger
        # (uncorrected) second moment estimates than smaller ones
        x = Tensor(np.array([1.0]), requires_grad=True)
        y = Tensor(np.array([1.0]), requires_grad=True)

        optimizer = Adam([x, y], lr=0.1)
        for _ in range(5):
            x.grad = np.array([1.0])
            y.grad = np.array([10.0])
            optimizer.step()

        assert optimizer.v[0][0] < optimizer.v[1][0]
