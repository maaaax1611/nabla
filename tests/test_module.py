from nabla.nn.module import Module
from nabla.tensor import Tensor
import numpy as np


class Inner(Module):
    def __init__(self):
        super().__init__()
        self.weight = Tensor(np.ones(3), requires_grad=True)

    def forward(self, x):
        return x


class Outer(Module):
    def __init__(self):
        super().__init__()
        self.bias = Tensor(np.zeros(2), requires_grad=True)
        self.non_trainable = Tensor(np.array([1.0, 2.0]), requires_grad=False)
        self.inner = Inner()

    def forward(self, x):
        return x


class TestModule:
    def test_starts_in_training_mode(self):
        module = Outer()
        assert module.training is True
        assert module.inner.training is True

    def test_eval_switches_self_and_submodules(self):
        module = Outer()
        result = module.eval()
        assert module.training is False
        assert module.inner.training is False
        assert result is module  # chainable, returns self

    def test_train_switches_back(self):
        module = Outer()
        module.eval()
        module.train()
        assert module.training is True
        assert module.inner.training is True

    def test_train_with_explicit_mode_false_matches_eval(self):
        module = Outer()
        module.train(False)
        assert module.training is False
        assert module.inner.training is False

    def test_parameters_collects_own_and_nested_trainable_tensors(self):
        module = Outer()
        params = module.parameters()
        assert len(params) == 2
        assert module.bias in params
        assert module.inner.weight in params

    def test_parameters_excludes_non_trainable_tensors(self):
        module = Outer()
        params = module.parameters()
        assert module.non_trainable not in params

    def test_zero_grad_clears_all_parameter_grads(self):
        module = Outer()
        module.bias.grad = np.ones(2)
        module.inner.weight.grad = np.ones(3)

        module.zero_grad()

        assert module.bias.grad is None
        assert module.inner.weight.grad is None

    def test_forward_not_implemented_by_default(self):
        class Empty(Module):
            pass

        empty = Empty()
        try:
            empty.forward()
            assert False, "expected NotImplementedError"
        except NotImplementedError:
            pass

    def test_call_dispatches_to_forward(self):
        module = Outer()
        x = Tensor(np.array([1.0, 2.0]))
        assert module(x) is x
