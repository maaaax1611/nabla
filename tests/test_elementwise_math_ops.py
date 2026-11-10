"""exp / log / pow / neg / tanh / gelu, each checked against a numerical gradient."""

import numpy as np
import pytest

import nabla.functional as F
from nabla.tensor import Tensor


def numerical_gradient(f, t, eps=1e-6):
    """Central-difference gradient of sum(f()) w.r.t. t.data."""
    grad = np.zeros_like(t.data)
    it = np.nditer(t.data, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        original = t.data[idx]
        t.data[idx] = original + eps
        plus = np.sum(f())
        t.data[idx] = original - eps
        minus = np.sum(f())
        t.data[idx] = original
        grad[idx] = (plus - minus) / (2 * eps)
    return grad


CASES = {
    "exp": (lambda t: t.exp(), np.exp, False),
    "log": (lambda t: t.log(), np.log, True),
    "sqrt": (lambda t: t.sqrt(), np.sqrt, True),
    "square": (lambda t: t ** 2, lambda a: a ** 2, False),
    "reciprocal": (lambda t: t ** -1, lambda a: 1 / a, True),
    "neg": (lambda t: -t, lambda a: -a, False),
    "tanh": (lambda t: t.tanh(), np.tanh, False),
}


@pytest.mark.parametrize("name", CASES)
def test_forward_matches_numpy(name):
    op, reference, positive_only = CASES[name]
    rng = np.random.default_rng(0)
    data = rng.uniform(0.5, 2.0, (3, 4)) if positive_only else rng.standard_normal((3, 4))
    assert np.allclose(op(Tensor(data)).data, reference(data))


@pytest.mark.parametrize("name", [*CASES, "gelu"])
def test_backward_matches_numerical_gradient(name):
    op = (lambda t: t.gelu()) if name == "gelu" else CASES[name][0]
    positive_only = name != "gelu" and CASES[name][2]
    rng = np.random.default_rng(1)
    data = rng.uniform(0.5, 2.0, (3, 4)) if positive_only else rng.standard_normal((3, 4))
    x = Tensor(data, requires_grad=True)
    op(x).sum().backward()
    expected = numerical_gradient(lambda: op(Tensor(x.data)).data, x)
    assert np.allclose(x.grad, expected, atol=1e-6)


def test_gelu_reference_values():
    # the tanh approximation stays within ~1e-3 of the exact (erf) GELU
    x = Tensor(np.array([-3.0, -1.0, 0.0, 1.0, 3.0]))
    exact = np.array([-0.00404951, -0.15865525, 0.0, 0.84134475, 2.99595049])
    assert np.allclose(x.gelu().data, exact, atol=1e-3)
    assert x.gelu().data[2] == 0.0


def test_functional_wrappers_match_methods():
    x = Tensor(np.random.default_rng(2).uniform(0.5, 2.0, (2, 3)))
    assert np.array_equal(F.exp(x).data, x.exp().data)
    assert np.array_equal(F.log(x).data, x.log().data)
    assert np.array_equal(F.tanh(x).data, x.tanh().data)
    assert np.array_equal(F.gelu(x).data, x.gelu().data)


def test_pow_rejects_tensor_exponent():
    x = Tensor(np.ones(3))
    with pytest.raises(TypeError):
        x ** Tensor(np.ones(3))


def test_softmax_composed_from_primitives_matches_fused_op():
    """exp / sum(axis) / divide are enough to rebuild softmax - and autodiff
    through the composition has to agree with the hand-derived Softmax op."""
    data = np.random.default_rng(3).standard_normal((4, 5))
    upstream = np.random.default_rng(4).standard_normal((4, 5))

    a = Tensor(data.copy(), requires_grad=True)
    e = a.exp()
    composed = e / e.sum(axis=-1, keepdims=True)
    composed.backward(upstream)

    b = Tensor(data.copy(), requires_grad=True)
    fused = b.softmax(axis=-1)
    fused.backward(upstream)

    assert np.allclose(composed.data, fused.data)
    assert np.allclose(a.grad, b.grad)
