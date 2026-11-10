"""Python numbers and plain arrays as the other operand of a Tensor op."""

import numpy as np
import pytest

from nabla.tensor import Tensor


def make():
    return Tensor(np.array([1.0, 2.0, 4.0]), requires_grad=True)


@pytest.mark.parametrize(
    "expr, value, grad",
    [
        (lambda x: x + 2, [3.0, 4.0, 6.0], [1.0, 1.0, 1.0]),
        (lambda x: 2 + x, [3.0, 4.0, 6.0], [1.0, 1.0, 1.0]),
        (lambda x: x - 1, [0.0, 1.0, 3.0], [1.0, 1.0, 1.0]),
        (lambda x: 1 - x, [0.0, -1.0, -3.0], [-1.0, -1.0, -1.0]),
        (lambda x: x * 3, [3.0, 6.0, 12.0], [3.0, 3.0, 3.0]),
        (lambda x: 3 * x, [3.0, 6.0, 12.0], [3.0, 3.0, 3.0]),
        (lambda x: x / 2, [0.5, 1.0, 2.0], [0.5, 0.5, 0.5]),
        (lambda x: 4 / x, [4.0, 2.0, 1.0], [-4.0, -1.0, -0.25]),
        (lambda x: -x, [-1.0, -2.0, -4.0], [-1.0, -1.0, -1.0]),
        (lambda x: x ** 2, [1.0, 4.0, 16.0], [2.0, 4.0, 8.0]),
    ],
)
def test_scalar_ops(expr, value, grad):
    x = make()
    out = expr(x)
    out.backward()
    assert np.allclose(out.data, value)
    assert np.allclose(x.grad, grad)


def test_numpy_array_on_either_side():
    arr = np.array([10.0, 20.0, 30.0])
    x = make()
    assert np.allclose((x + arr).data, [11.0, 22.0, 34.0])
    left = arr - x  # must dispatch to Tensor.__rsub__, not numpy broadcasting
    assert isinstance(left, Tensor)
    assert np.allclose(left.data, [9.0, 18.0, 26.0])
    left.backward()
    assert np.allclose(x.grad, [-1.0, -1.0, -1.0])


def test_scalar_does_not_promote_float32():
    x = Tensor(np.ones(3, dtype=np.float32), requires_grad=True)
    out = (x * 0.5 + 1) / 3
    assert out.data.dtype == np.float32
    out.backward()
    assert x.grad.dtype == np.float32


def test_constant_operand_gets_no_gradient():
    x = make()
    out = x * 2
    out.backward()
    constant = out._prev  # graph is freed after backward
    assert constant == ()
