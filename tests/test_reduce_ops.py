from nabla.tensor import Tensor
import numpy as np


class TestSum:
    def test_forward(self):
        arr = np.array([1, 2, 3, 4])
        x = Tensor(arr)
        y = x.sum()
        assert np.isclose(y.data, np.sum(arr))

    def test_backward(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        x = Tensor(arr, requires_grad=True)
        y = x.sum()
        y.backward(np.array(1.0))
        expected_grad = np.ones_like(arr)
        assert np.allclose(x.grad, expected_grad)


class TestMean:
    def test_forward(self):
        arr = np.array([1, 2, 3, 4])
        x = Tensor(arr)
        y = x.mean()
        assert np.isclose(y.data, np.mean(arr))

    def test_backward(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        x = Tensor(arr, requires_grad=True)
        y = x.mean()
        y.backward(np.array(1.0))
        expected_grad = np.ones_like(arr) / arr.size
        assert np.allclose(x.grad, expected_grad)
