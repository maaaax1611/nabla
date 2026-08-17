from __future__ import annotations

import numpy as np

from nabla.optim.clip import clip_grad_norm_
from nabla.tensor import Tensor


class TestClipGradNorm:
    def test_returns_combined_norm_before_clipping(self):
        x = Tensor(np.zeros(1), requires_grad=True)
        x.grad = np.array([3.0, 4.0])  # norm = 5

        total_norm = clip_grad_norm_([x], max_norm=100.0)
        assert np.isclose(total_norm, 5.0)

    def test_no_clipping_when_under_max_norm(self):
        x = Tensor(np.zeros(2), requires_grad=True)
        x.grad = np.array([3.0, 4.0])

        clip_grad_norm_([x], max_norm=10.0)
        assert np.allclose(x.grad, [3.0, 4.0])

    def test_scales_down_when_over_max_norm(self):
        x = Tensor(np.zeros(2), requires_grad=True)
        x.grad = np.array([3.0, 4.0])  # norm = 5

        clip_grad_norm_([x], max_norm=1.0)

        # clipped gradient should have norm ~= max_norm, same direction
        clipped_norm = np.linalg.norm(x.grad)
        assert np.isclose(clipped_norm, 1.0, atol=1e-4)
        assert np.allclose(x.grad / clipped_norm, [3.0, 4.0] / np.linalg.norm([3.0, 4.0]))

    def test_norm_is_combined_across_multiple_parameters(self):
        x = Tensor(np.zeros(1), requires_grad=True)
        y = Tensor(np.zeros(1), requires_grad=True)
        x.grad = np.array([3.0])
        y.grad = np.array([4.0])

        total_norm = clip_grad_norm_([x, y], max_norm=100.0)
        assert np.isclose(total_norm, 5.0)  # sqrt(3^2 + 4^2), not summed separately

    def test_clipping_scales_all_parameters_by_the_same_factor(self):
        x = Tensor(np.zeros(1), requires_grad=True)
        y = Tensor(np.zeros(1), requires_grad=True)
        x.grad = np.array([3.0])
        y.grad = np.array([4.0])

        clip_grad_norm_([x, y], max_norm=1.0)  # combined norm 5 -> clip_coef ~ 1/5

        assert np.isclose(x.grad[0] / 3.0, y.grad[0] / 4.0)

    def test_skips_parameters_without_gradient(self):
        x = Tensor(np.zeros(1), requires_grad=True)
        y = Tensor(np.zeros(1), requires_grad=True)
        x.grad = np.array([3.0])
        y.grad = None

        total_norm = clip_grad_norm_([x, y], max_norm=100.0)
        assert np.isclose(total_norm, 3.0)
        assert y.grad is None

    def test_no_gradients_returns_zero(self):
        x = Tensor(np.zeros(1), requires_grad=True)
        x.grad = None

        assert clip_grad_norm_([x], max_norm=1.0) == 0.0
