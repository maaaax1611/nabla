from nabla.tensor import Tensor
import numpy as np

class TestAdd:
    def test_forward(self):
        x = Tensor(np.array([1, 2, 3]))
        y = Tensor(np.array([4, 5, 6]))
        z = x + y
        assert np.array_equal(z.data, np.array([5, 7, 9]))

    def test_backward(self):
        x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        y = Tensor(np.array([4.0, 5.0, 6.0]), requires_grad=True)
        z = x + y
        z.backward(np.array([1.0, 1.0, 1.0]))
        assert np.array_equal(x.grad, np.array([1.0, 1.0, 1.0]))
        assert np.array_equal(y.grad, np.array([1.0, 1.0, 1.0]))


class TestSubtract:
    def test_forward(self):
        x = Tensor(np.array([4, 5, 6]))
        y = Tensor(np.array([1, 2, 3]))
        z = x - y
        assert np.array_equal(z.data, np.array([3, 3, 3]))

    def test_backward(self):
        x = Tensor(np.array([4.0, 5.0, 6.0]), requires_grad=True)
        y = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        z = x - y
        z.backward(np.array([1.0, 1.0, 1.0]))
        assert np.array_equal(x.grad, np.array([1.0, 1.0, 1.0]))
        assert np.array_equal(y.grad, np.array([-1.0, -1.0, -1.0]))


class TestMultiply:
    def test_forward(self):
        x = Tensor(np.array([1, 2, 3]))
        y = Tensor(np.array([4, 5, 6]))
        z = x * y
        assert np.array_equal(z.data, np.array([4, 10, 18]))

    def test_backward(self):
        x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        y = Tensor(np.array([4.0, 5.0, 6.0]), requires_grad=True)
        z = x * y
        z.backward(np.array([1.0, 1.0, 1.0]))
        assert np.array_equal(x.grad, np.array([4.0, 5.0, 6.0]))
        assert np.array_equal(y.grad, np.array([1.0, 2.0, 3.0]))


class TestDivide:
    def test_forward(self):
        x = Tensor(np.array([4, 5, 6]))
        y = Tensor(np.array([2, 5, 3]))
        z = x / y
        assert np.array_equal(z.data, np.array([2.0, 1.0, 2.0]))

    def test_backward(self):
        x = Tensor(np.array([4.0, 5.0, 6.0]), requires_grad=True)
        y = Tensor(np.array([2.0, 5.0, 3.0]), requires_grad=True)
        z = x / y
        z.backward(np.array([1.0, 1.0, 1.0]))
        assert np.allclose(x.grad, np.array([0.5, 0.2, 0.33333333]))
        assert np.allclose(y.grad, np.array([-1.0, -0.2, -0.66666667]))


class TestChainedOperations:
    def test_forward(self):
        x = Tensor(np.array([1, 2, 3]))
        y = Tensor(np.array([4, 5, 6]))
        z = (x + y) * (x - y)
        assert np.array_equal(z.data, np.array([-15, -21, -27]))

    def test_backward(self):
        x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        y = Tensor(np.array([4.0, 5.0, 6.0]), requires_grad=True)
        z = (x + y) * (x - y)
        z.backward(np.array([1.0, 1.0, 1.0]))
        assert np.allclose(x.grad, np.array([2.0, 4.0, 6.0]))
        assert np.allclose(y.grad, np.array([-8.0, -10.0, -12.0]))
