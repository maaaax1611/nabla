from nabla.tensor import Tensor
from nabla.ops.pooling import AvgPool2D, MaxPool2D
import numpy as np


def naive_max_pool2d(x, kernel_size, stride):
    batch, channels, H, W = x.shape
    out_h = (H - kernel_size) // stride + 1
    out_w = (W - kernel_size) // stride + 1

    out = np.zeros((batch, channels, out_h, out_w))
    for b in range(batch):
        for c in range(channels):
            for i in range(out_h):
                for j in range(out_w):
                    h0, w0 = i * stride, j * stride
                    out[b, c, i, j] = np.max(x[b, c, h0 : h0 + kernel_size, w0 : w0 + kernel_size])
    return out


def naive_avg_pool2d(x, kernel_size, stride):
    batch, channels, H, W = x.shape
    out_h = (H - kernel_size) // stride + 1
    out_w = (W - kernel_size) // stride + 1

    out = np.zeros((batch, channels, out_h, out_w))
    for b in range(batch):
        for c in range(channels):
            for i in range(out_h):
                for j in range(out_w):
                    h0, w0 = i * stride, j * stride
                    out[b, c, i, j] = np.mean(x[b, c, h0 : h0 + kernel_size, w0 : w0 + kernel_size])
    return out


def numerical_gradient(f, t, grad_output, eps=1e-5):
    """Central-difference gradient of sum(f() * grad_output) w.r.t. t.data."""
    grad = np.zeros_like(t.data)
    it = np.nditer(t.data, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        original = t.data[idx]

        t.data[idx] = original + eps
        out_plus = f().copy()

        t.data[idx] = original - eps
        out_minus = f().copy()

        t.data[idx] = original
        grad[idx] = np.sum((out_plus - out_minus) / (2 * eps) * grad_output)
    return grad


class TestMaxPool2D:
    def test_forward_shape(self):
        x = Tensor(np.random.randn(2, 3, 6, 6))
        out = MaxPool2D.apply(x, kernel_size=2, stride=2)
        assert out.data.shape == (2, 3, 3, 3)

    def test_forward_matches_naive_reference(self):
        np.random.seed(0)
        x = Tensor(np.random.randn(2, 3, 7, 7))
        out = MaxPool2D.apply(x, kernel_size=3, stride=2)
        expected = naive_max_pool2d(x.data, kernel_size=3, stride=2)
        assert np.allclose(out.data, expected)

    def test_forward_default_stride_equals_kernel_size(self):
        x = Tensor(np.arange(16).astype(float).reshape(1, 1, 4, 4))
        out = MaxPool2D.apply(x, kernel_size=2)
        expected = np.array([[[[5.0, 7.0], [13.0, 15.0]]]])
        assert np.array_equal(out.data, expected)

    def test_backward_routes_gradient_to_max_position(self):
        x = Tensor(np.array([[[[1.0, 2.0], [4.0, 3.0]]]]), requires_grad=True)
        out = MaxPool2D.apply(x, kernel_size=2, stride=2)
        out.backward(np.array([[[[1.0]]]]))

        expected_grad = np.array([[[[0.0, 0.0], [1.0, 0.0]]]])
        assert np.array_equal(x.grad, expected_grad)

    def test_backward_matches_numerical_gradient_no_overlap(self):
        np.random.seed(1)
        x = Tensor(np.random.randn(2, 3, 6, 6), requires_grad=True)
        kernel_size, stride = 2, 2

        def forward():
            return MaxPool2D(kernel_size=kernel_size, stride=stride).forward(x)

        grad_output = np.random.randn(*forward().shape)

        out = MaxPool2D.apply(x, kernel_size=kernel_size, stride=stride)
        out.backward(grad_output)

        assert np.allclose(x.grad, numerical_gradient(forward, x, grad_output), atol=1e-6)

    def test_backward_matches_numerical_gradient_with_overlap(self):
        np.random.seed(2)
        x = Tensor(np.random.randn(2, 3, 5, 5), requires_grad=True)
        kernel_size, stride = 2, 1  # stride < kernel_size -> overlapping windows

        def forward():
            return MaxPool2D(kernel_size=kernel_size, stride=stride).forward(x)

        grad_output = np.random.randn(*forward().shape)

        out = MaxPool2D.apply(x, kernel_size=kernel_size, stride=stride)
        out.backward(grad_output)

        assert np.allclose(x.grad, numerical_gradient(forward, x, grad_output), atol=1e-6)


class TestAvgPool2D:
    def test_forward_shape(self):
        x = Tensor(np.random.randn(2, 3, 6, 6))
        out = AvgPool2D.apply(x, kernel_size=2, stride=2)
        assert out.data.shape == (2, 3, 3, 3)

    def test_forward_matches_naive_reference(self):
        np.random.seed(3)
        x = Tensor(np.random.randn(2, 3, 7, 7))
        out = AvgPool2D.apply(x, kernel_size=3, stride=2)
        expected = naive_avg_pool2d(x.data, kernel_size=3, stride=2)
        assert np.allclose(out.data, expected)

    def test_forward_simple_values(self):
        x = Tensor(np.arange(16).astype(float).reshape(1, 1, 4, 4))
        out = AvgPool2D.apply(x, kernel_size=2)
        expected = np.array([[[[2.5, 4.5], [10.5, 12.5]]]])
        assert np.array_equal(out.data, expected)

    def test_backward_distributes_gradient_equally(self):
        x = Tensor(np.zeros((1, 1, 2, 2)), requires_grad=True)
        out = AvgPool2D.apply(x, kernel_size=2, stride=2)
        out.backward(np.array([[[[4.0]]]]))

        assert np.allclose(x.grad, np.full((1, 1, 2, 2), 1.0))

    def test_backward_matches_numerical_gradient_no_overlap(self):
        np.random.seed(4)
        x = Tensor(np.random.randn(2, 3, 6, 6), requires_grad=True)
        kernel_size, stride = 2, 2

        def forward():
            return AvgPool2D(kernel_size=kernel_size, stride=stride).forward(x)

        grad_output = np.random.randn(*forward().shape)

        out = AvgPool2D.apply(x, kernel_size=kernel_size, stride=stride)
        out.backward(grad_output)

        assert np.allclose(x.grad, numerical_gradient(forward, x, grad_output), atol=1e-6)

    def test_backward_matches_numerical_gradient_with_overlap(self):
        np.random.seed(5)
        x = Tensor(np.random.randn(2, 3, 5, 5), requires_grad=True)
        kernel_size, stride = 2, 1  # stride < kernel_size -> overlapping windows

        def forward():
            return AvgPool2D(kernel_size=kernel_size, stride=stride).forward(x)

        grad_output = np.random.randn(*forward().shape)

        out = AvgPool2D.apply(x, kernel_size=kernel_size, stride=stride)
        out.backward(grad_output)

        assert np.allclose(x.grad, numerical_gradient(forward, x, grad_output), atol=1e-6)


class TestTensorPoolingWrappers:
    def test_max_pool2d_matches_function(self):
        x = Tensor(np.random.randn(2, 3, 6, 6))
        via_wrapper = x.max_pool2d(kernel_size=2, stride=2)
        via_function = MaxPool2D.apply(x, kernel_size=2, stride=2)
        assert np.array_equal(via_wrapper.data, via_function.data)

    def test_avg_pool2d_matches_function(self):
        x = Tensor(np.random.randn(2, 3, 6, 6))
        via_wrapper = x.avg_pool2d(kernel_size=2, stride=2)
        via_function = AvgPool2D.apply(x, kernel_size=2, stride=2)
        assert np.array_equal(via_wrapper.data, via_function.data)

    def test_max_pool2d_default_stride_equals_kernel_size(self):
        x = Tensor(np.random.randn(1, 1, 4, 4))
        assert x.max_pool2d(kernel_size=2).data.shape == (1, 1, 2, 2)

    def test_max_pool2d_backward_fills_gradient(self):
        x = Tensor(np.random.randn(2, 3, 4, 4), requires_grad=True)
        out = x.max_pool2d(kernel_size=2)
        out.sum().backward()
        assert x.grad.shape == x.data.shape
