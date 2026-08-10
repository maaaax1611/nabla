from nabla.tensor import Tensor
from nabla.ops.conv import Conv2D
import numpy as np


def naive_conv2d(x, weight, bias, stride=1, padding=0):
    """Reference implementation via plain nested loops (no im2col)."""
    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))

    batch, _, H, W = x.shape
    out_channels, in_channels, kh, kw = weight.shape
    out_h = (H - kh) // stride + 1
    out_w = (W - kw) // stride + 1

    out = np.zeros((batch, out_channels, out_h, out_w))
    for b in range(batch):
        for oc in range(out_channels):
            for i in range(out_h):
                for j in range(out_w):
                    h0, w0 = i * stride, j * stride
                    patch = x[b, :, h0:h0 + kh, w0:w0 + kw]
                    out[b, oc, i, j] = np.sum(patch * weight[oc]) + bias[oc]
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


class TestConv2D:
    def test_forward_shape_no_padding_stride1(self):
        x = Tensor(np.random.randn(2, 3, 5, 5))
        weight = Tensor(np.random.randn(4, 3, 3, 3))
        bias = Tensor(np.random.randn(4))

        out = Conv2D.apply(x, weight, bias, stride=1, padding=0)

        assert out.data.shape == (2, 4, 3, 3)

    def test_forward_matches_naive_reference(self):
        np.random.seed(0)
        x = Tensor(np.random.randn(2, 3, 7, 7))
        weight = Tensor(np.random.randn(4, 3, 3, 3))
        bias = Tensor(np.random.randn(4))

        out = Conv2D.apply(x, weight, bias, stride=1, padding=0)
        expected = naive_conv2d(x.data, weight.data, bias.data, stride=1, padding=0)

        assert np.allclose(out.data, expected)

    def test_forward_with_stride(self):
        np.random.seed(1)
        x = Tensor(np.random.randn(2, 3, 9, 9))
        weight = Tensor(np.random.randn(4, 3, 3, 3))
        bias = Tensor(np.random.randn(4))

        out = Conv2D.apply(x, weight, bias, stride=2, padding=0)
        expected = naive_conv2d(x.data, weight.data, bias.data, stride=2, padding=0)

        assert out.data.shape == (2, 4, 4, 4)
        assert np.allclose(out.data, expected)

    def test_forward_with_padding(self):
        np.random.seed(2)
        x = Tensor(np.random.randn(2, 3, 5, 5))
        weight = Tensor(np.random.randn(4, 3, 3, 3))
        bias = Tensor(np.random.randn(4))

        out = Conv2D.apply(x, weight, bias, stride=1, padding=1)
        expected = naive_conv2d(x.data, weight.data, bias.data, stride=1, padding=1)

        assert out.data.shape == (2, 4, 5, 5)
        assert np.allclose(out.data, expected)

    def test_backward_bias_is_sum_over_batch_and_spatial_dims(self):
        x = Tensor(np.random.randn(2, 3, 5, 5), requires_grad=True)
        weight = Tensor(np.random.randn(4, 3, 3, 3), requires_grad=True)
        bias = Tensor(np.random.randn(4), requires_grad=True)

        out = Conv2D.apply(x, weight, bias, stride=1, padding=0)
        grad_output = np.ones_like(out.data)
        out.backward(grad_output)

        expected_grad_bias = grad_output.sum(axis=(0, 2, 3))
        assert np.allclose(bias.grad, expected_grad_bias)

    def test_backward_matches_numerical_gradient_no_padding(self):
        np.random.seed(3)
        x = Tensor(np.random.randn(2, 3, 6, 6), requires_grad=True)
        weight = Tensor(np.random.randn(4, 3, 3, 3), requires_grad=True)
        bias = Tensor(np.random.randn(4), requires_grad=True)
        stride, padding = 2, 0

        def forward():
            return Conv2D(stride=stride, padding=padding).forward(x, weight, bias)

        grad_output = np.random.randn(*forward().shape)

        out = Conv2D.apply(x, weight, bias, stride=stride, padding=padding)
        out.backward(grad_output)

        assert np.allclose(x.grad, numerical_gradient(forward, x, grad_output), atol=1e-6)
        assert np.allclose(weight.grad, numerical_gradient(forward, weight, grad_output), atol=1e-6)
        assert np.allclose(bias.grad, numerical_gradient(forward, bias, grad_output), atol=1e-6)

    def test_backward_matches_numerical_gradient_with_padding_and_stride(self):
        np.random.seed(4)
        x = Tensor(np.random.randn(2, 3, 7, 7), requires_grad=True)
        weight = Tensor(np.random.randn(4, 3, 3, 3), requires_grad=True)
        bias = Tensor(np.random.randn(4), requires_grad=True)
        stride, padding = 2, 1

        def forward():
            return Conv2D(stride=stride, padding=padding).forward(x, weight, bias)

        grad_output = np.random.randn(*forward().shape)

        out = Conv2D.apply(x, weight, bias, stride=stride, padding=padding)
        out.backward(grad_output)

        assert np.allclose(x.grad, numerical_gradient(forward, x, grad_output), atol=1e-6)
        assert np.allclose(weight.grad, numerical_gradient(forward, weight, grad_output), atol=1e-6)
        assert np.allclose(bias.grad, numerical_gradient(forward, bias, grad_output), atol=1e-6)
