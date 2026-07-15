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