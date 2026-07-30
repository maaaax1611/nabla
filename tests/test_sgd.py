from __future__ import annotations

import numpy as np

from nabla.optim.sgd import SGD
from nabla.regularizers import L2Regularizer
from nabla.tensor import Tensor


class TestSGD:
    def test_step_updates_parameter(self):
        x = Tensor(np.array([1.0]), requires_grad=True)
        x.grad = np.array([2.0])
        optimizer = SGD([x], lr=0.1)
        optimizer.step()
        assert np.allclose(x.data, np.array([1.0 - 0.1 * 2.0]))

    def test_zero_grad_resets_gradients(self):
        x = Tensor(np.array([1.0]), requires_grad=True)
        x.grad = np.array([1.0])
        optimizer = SGD([x], lr=0.1)
        optimizer.zero_grad()
        assert x.grad is None

    def test_skips_parameters_without_gradient(self):
        x = Tensor(np.array([1.0]), requires_grad=True)
        x.grad = None
        optimizer = SGD([x], lr=0.1)
        optimizer.step()
        assert np.array_equal(x.data, np.array([1.0]))

    def test_step_with_l2_regularizer(self):
        x = Tensor(np.array([2.0]), requires_grad=True)
        x.grad = np.array([1.0])
        reg = L2Regularizer(weight_decay=0.1)
        optimizer = SGD([x], lr=0.5, regularizers=[reg])
        optimizer.step()

        # effective grad = param.grad + weight_decay * param.data = 1.0 + 0.1 * 2.0 = 1.2
        expected = 2.0 - 0.5 * 1.2
        assert np.allclose(x.data, np.array([expected]))

    def test_regularizer_does_not_mutate_raw_gradient(self):
        x = Tensor(np.array([2.0]), requires_grad=True)
        x.grad = np.array([1.0])
        reg = L2Regularizer(weight_decay=0.1)
        optimizer = SGD([x], lr=0.5, regularizers=[reg])
        optimizer.step()

        # the raw gradient itself must stay untouched by the regularizer
        assert np.allclose(x.grad, np.array([1.0]))

    def test_no_regularizers_behaves_like_plain_sgd(self):
        x = Tensor(np.array([2.0]), requires_grad=True)
        x.grad = np.array([1.0])
        optimizer = SGD([x], lr=0.5, regularizers=None)
        optimizer.step()
        assert np.allclose(x.data, np.array([2.0 - 0.5 * 1.0]))
