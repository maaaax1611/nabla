from __future__ import annotations

import numpy as np

from nabla.regularizers import L1Regularizer, L2Regularizer
from nabla.tensor import Tensor


class TestL2Regularizer:
    def test_apply(self):
        param = Tensor(np.array([1.0, -2.0, 3.0]), requires_grad=True)
        reg = L2Regularizer(weight_decay=0.1)
        assert np.allclose(reg.apply(param), np.array([0.1, -0.2, 0.3]))

    def test_zero_weight_decay_has_no_effect(self):
        param = Tensor(np.array([1.0, -2.0, 3.0]), requires_grad=True)
        reg = L2Regularizer(weight_decay=0.0)
        assert np.allclose(reg.apply(param), np.zeros(3))


class TestL1Regularizer:
    def test_apply(self):
        param = Tensor(np.array([1.0, -2.0, 0.0]), requires_grad=True)
        reg = L1Regularizer(weight_decay=0.1)
        assert np.allclose(reg.apply(param), np.array([0.1, -0.1, 0.0]))

    def test_zero_weight_decay_has_no_effect(self):
        param = Tensor(np.array([1.0, -2.0, 3.0]), requires_grad=True)
        reg = L1Regularizer(weight_decay=0.0)
        assert np.allclose(reg.apply(param), np.zeros(3))
