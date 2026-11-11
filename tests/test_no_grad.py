import numpy as np

import nabla
from nabla import nn
from nabla.tensor import Tensor


def test_no_grad_skips_graph_construction():
    w = Tensor(np.ones((2, 2)), requires_grad=True)
    with nabla.no_grad():
        out = (w * 3).sum()
    assert out.requires_grad is False
    assert out._ctx is None and out._prev == ()


def test_grad_mode_is_restored_after_block_and_on_exception():
    assert nabla.is_grad_enabled()
    with nabla.no_grad():
        assert not nabla.is_grad_enabled()
        with nabla.no_grad():
            assert not nabla.is_grad_enabled()
        assert not nabla.is_grad_enabled()
    assert nabla.is_grad_enabled()

    try:
        with nabla.no_grad():
            raise RuntimeError
    except RuntimeError:
        pass
    assert nabla.is_grad_enabled()


def test_no_grad_as_decorator():
    w = Tensor(np.ones(3), requires_grad=True)

    @nabla.no_grad()
    def evaluate():
        return (w * 2).sum()

    assert evaluate().requires_grad is False
    assert (w * 2).sum().requires_grad is True


def test_ops_on_constants_are_never_recorded():
    # no input requires grad -> nothing to backprop into -> no graph node
    a = Tensor(np.ones(3))
    out = a * 2 + 1
    assert out._ctx is None and out.requires_grad is False


def test_training_still_works_after_an_eval_pass():
    np.random.seed(0)
    layer = nn.Linear(3, 2)
    x = Tensor(np.random.randn(4, 3))
    with nabla.no_grad():
        layer(x)
    layer(x).sum().backward()
    assert layer.weight.grad is not None


def test_batchnorm_in_training_mode_under_no_grad_updates_running_stats():
    # the layer used to read the batch statistics from out._ctx, which
    # does not exist when no graph is recorded
    bn = nn.BatchNorm2D(3)
    x = Tensor(np.random.default_rng(0).normal(2.0, 1.0, (4, 3, 5, 5)))
    with nabla.no_grad():
        out = bn(x)
    assert out.requires_grad is False
    assert np.allclose(bn.running_mean, 0.1 * x.data.mean(axis=(0, 2, 3)))
