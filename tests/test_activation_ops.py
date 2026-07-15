from nabla.tensor import Tensor
import numpy as np


class TestReLU:
    def test_forward_positive(self):
        x = Tensor(np.array([1.0, 2.0, 3.0]))
        z = x.relu()
        assert np.array_equal(z.data, np.array([1.0, 2.0, 3.0]))

    def test_forward_negative(self):
        x = Tensor(np.array([-1.0, -2.0, -3.0]))
        z = x.relu()
        assert np.array_equal(z.data, np.array([0.0, 0.0, 0.0]))

    def test_forward_mixed(self):
        x = Tensor(np.array([-1.0, 0.0, 1.0]))
        z = x.relu()
        assert np.array_equal(z.data, np.array([0.0, 0.0, 1.0]))

    def test_backward(self):
        x = Tensor(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]), requires_grad=True)
        z = x.relu()
        z.backward(np.array([1.0, 1.0, 1.0, 1.0, 1.0]))
        assert np.array_equal(x.grad, np.array([0.0, 0.0, 0.0, 1.0, 1.0]))


class TestSigmoid:
    def test_forward(self):
        x = Tensor(np.array([0.0]))
        z = x.sigmoid()
        assert np.allclose(z.data, np.array([0.5]))

    def test_forward_positive(self):
        x = Tensor(np.array([100.0]))
        z = x.sigmoid()
        assert np.allclose(z.data, np.array([1.0]))

    def test_forward_negative(self):
        x = Tensor(np.array([-100.0]))
        z = x.sigmoid()
        assert np.allclose(z.data, np.array([0.0]))

    def test_backward(self):
        x = Tensor(np.array([0.0, 1.0, -1.0]), requires_grad=True)
        z = x.sigmoid()
        z.backward(np.array([1.0, 1.0, 1.0]))
        # sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        sig = 1 / (1 + np.exp(-np.array([0.0, 1.0, -1.0])))
        expected = sig * (1 - sig)
        assert np.allclose(x.grad, expected)
