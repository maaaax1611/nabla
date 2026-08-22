import numpy as np

from nabla.grad_mode import is_grad_enabled, no_grad
from nabla.tensor import Tensor


class TestNoGrad:
    def test_grad_enabled_by_default(self):
        assert is_grad_enabled() is True

    def test_disabled_inside_block(self):
        with no_grad():
            assert is_grad_enabled() is False
        assert is_grad_enabled() is True

    def test_restores_previous_state_on_exception(self):
        try:
            with no_grad():
                raise ValueError("boom")
        except ValueError:
            pass
        assert is_grad_enabled() is True

    def test_nesting_restores_outer_state(self):
        with no_grad():
            with no_grad():
                assert is_grad_enabled() is False
            assert is_grad_enabled() is False
        assert is_grad_enabled() is True

    def test_output_does_not_require_grad_inside_block(self):
        x = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        with no_grad():
            y = x * Tensor(np.array(2.0))
        assert y.requires_grad is False

    def test_output_has_no_graph_links_inside_block(self):
        x = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        with no_grad():
            y = x * Tensor(np.array(2.0))
        assert y._ctx is None
        assert y._prev == ()

    def test_forward_value_is_unaffected(self):
        x = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        with no_grad():
            y = x * Tensor(np.array(2.0))
        assert np.array_equal(y.data, np.array([2.0, 4.0]))

    def test_graph_outside_block_is_unaffected(self):
        x = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        y = x * Tensor(np.array(2.0))
        assert y.requires_grad is True
        assert y._ctx is not None
