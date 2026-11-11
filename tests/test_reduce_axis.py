"""Sum / Mean over an axis (the scalar case lives in test_reduce_ops.py)."""

import numpy as np
import pytest

from nabla.tensor import Tensor

AXES = [None, 0, 1, -1, (0, 2), (1, -1)]


@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("op", ["sum", "mean"])
def test_forward_matches_numpy(op, axis, keepdims):
    data = np.random.default_rng(0).standard_normal((2, 3, 4))
    out = getattr(Tensor(data), op)(axis=axis, keepdims=keepdims)
    expected = getattr(np, op)(data, axis=axis, keepdims=keepdims)
    assert out.data.shape == expected.shape
    assert np.allclose(out.data, expected)


@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("op", ["sum", "mean"])
def test_backward_weights_each_element_by_upstream_gradient(op, axis, keepdims):
    data = np.random.default_rng(1).standard_normal((2, 3, 4))
    x = Tensor(data, requires_grad=True)
    out = getattr(x, op)(axis=axis, keepdims=keepdims)
    upstream = np.random.default_rng(2).standard_normal(out.data.shape)
    out.backward(upstream)

    # d/dx of sum(reduce(x) * upstream): each x element gets the upstream
    # value of the output cell it was reduced into (times 1/n for mean)
    expanded = upstream if keepdims or axis is None else np.expand_dims(upstream, axis)
    expected = np.broadcast_to(expanded, data.shape)
    if op == "mean":
        expected = expected / (data.size / out.data.size)
    assert x.grad.shape == data.shape
    assert np.allclose(x.grad, expected)


def test_gradient_is_writable_and_independent():
    # broadcast_to returns a read-only view - backward must not leak that,
    # optimizers and gradient clipping modify .grad in place
    x = Tensor(np.ones((2, 3)), requires_grad=True)
    x.sum(axis=0).backward(np.ones(3))
    x.grad *= 2.0
    assert np.array_equal(x.grad, np.full((2, 3), 2.0))


def test_mean_over_axis_then_scalar_loss():
    x = Tensor(np.arange(6.0).reshape(2, 3), requires_grad=True)
    loss = (x.mean(axis=1) ** 2).sum()
    loss.backward()
    row_means = np.array([[1.0], [4.0]])
    assert np.allclose(x.grad, np.broadcast_to(2 * row_means / 3, (2, 3)))
