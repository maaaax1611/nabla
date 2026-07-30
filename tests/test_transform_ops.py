from nabla.tensor import Tensor
import numpy as np


class TestMatMul:
    def test_forward(self):
        a = Tensor(np.array([[1, 2], [3, 4]]))
        b = Tensor(np.array([[5, 6], [7, 8]]))
        c = a.matmul(b)
        expected = np.dot(a.data, b.data)
        assert np.array_equal(c.data, expected)

    def test_backward(self):
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
        b = Tensor(np.array([[5.0, 6.0], [7.0, 8.0]]), requires_grad=True)
        c = a.matmul(b)
        grad_output = np.array([[1.0, 1.0], [1.0, 1.0]])
        c.backward(grad_output)
        
        # Expected gradients
        expected_grad_a = np.dot(grad_output, b.data.T)
        expected_grad_b = np.dot(a.data.T, grad_output)
        
        assert np.allclose(a.grad, expected_grad_a)
        assert np.allclose(b.grad, expected_grad_b)


class TestReshape:
    def test_forward(self):
        x = Tensor(np.arange(6).astype(float))
        y = x.reshape((2, 3))
        expected = x.data.reshape((2, 3))
        assert y.data.shape == (2, 3)
        assert np.array_equal(y.data, expected)

    def test_backward_restores_original_shape(self):
        x = Tensor(np.arange(6).astype(float).reshape(2, 3), requires_grad=True)
        y = x.reshape((3, 2))
        grad_output = np.ones((3, 2))
        y.backward(grad_output)

        assert x.grad.shape == x.data.shape
        assert np.array_equal(x.grad, np.ones((2, 3)))

    def test_backward_values_correct(self):
        x = Tensor(np.arange(6).astype(float).reshape(2, 3), requires_grad=True)
        y = x.reshape((6,))
        grad_output = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        y.backward(grad_output)

        expected_grad = grad_output.reshape(2, 3)
        assert np.array_equal(x.grad, expected_grad)


class TestTranspose:
    def test_forward_default_axes(self):
        x = Tensor(np.arange(6).astype(float).reshape(2, 3))
        y = x.transpose()
        assert np.array_equal(y.data, x.data.T)

    def test_forward_explicit_axes(self):
        x = Tensor(np.arange(24).astype(float).reshape(2, 3, 4))
        y = x.transpose((2, 0, 1))
        expected = np.transpose(x.data, (2, 0, 1))
        assert y.data.shape == (4, 2, 3)
        assert np.array_equal(y.data, expected)

    def test_backward_default_axes(self):
        x = Tensor(np.arange(6).astype(float).reshape(2, 3), requires_grad=True)
        y = x.transpose()
        grad_output = np.arange(6).astype(float).reshape(3, 2)
        y.backward(grad_output)

        expected_grad = grad_output.T
        assert np.array_equal(x.grad, expected_grad)

    def test_backward_explicit_axes_restores_original_shape(self):
        x = Tensor(np.arange(24).astype(float).reshape(2, 3, 4), requires_grad=True)
        y = x.transpose((2, 0, 1))
        grad_output = np.ones((4, 2, 3))
        y.backward(grad_output)

        assert x.grad.shape == x.data.shape
        assert np.array_equal(x.grad, np.ones((2, 3, 4)))

    def test_transpose_then_transpose_back_is_identity_grad(self):
        # Sanity check: transposing forward and back with the inverse
        # permutation should propagate gradients unchanged.
        x = Tensor(np.arange(24).astype(float).reshape(2, 3, 4), requires_grad=True)
        y = x.transpose((2, 0, 1)).transpose((1, 2, 0))
        assert np.array_equal(y.data, x.data)

        grad_output = np.ones((2, 3, 4))
        y.backward(grad_output)
        assert np.array_equal(x.grad, grad_output)


class TestT:
    def test_forward_matches_numpy(self):
        x = Tensor(np.arange(6).astype(float).reshape(2, 3))
        assert np.array_equal(x.T.data, x.data.T)

    def test_backward(self):
        x = Tensor(np.arange(6).astype(float).reshape(2, 3), requires_grad=True)
        y = x.T
        grad_output = np.arange(6).astype(float).reshape(3, 2)
        y.backward(grad_output)

        assert np.array_equal(x.grad, grad_output.T)