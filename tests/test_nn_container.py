import numpy as np

from nabla.nn.container import ModuleList
from nabla.nn.linear import Linear
from nabla.tensor import Tensor


class TestModuleList:
    def test_len_and_indexing(self):
        layers = ModuleList([Linear(4, 4), Linear(4, 4), Linear(4, 4)])
        assert len(layers) == 3
        assert isinstance(layers[0], Linear)
        assert layers[0] is not layers[1]

    def test_iteration_preserves_order(self):
        linears = [Linear(4, 4) for _ in range(3)]
        layers = ModuleList(linears)
        assert list(layers) == linears

    def test_parameters_include_every_submodule(self):
        layers = ModuleList([Linear(4, 4) for _ in range(3)])
        params = layers.parameters()
        assert len(params) == 6  # weight + bias per Linear
        for linear in layers:
            assert linear.weight in params
            assert linear.bias in params

    def test_train_eval_propagates_to_every_submodule(self):
        from nabla.nn.dropout import Dropout

        layers = ModuleList([Dropout(0.5) for _ in range(3)])
        assert all(layer.training for layer in layers)
        layers.eval()
        assert all(not layer.training for layer in layers)
        layers.train()
        assert all(layer.training for layer in layers)

    def test_backward_fills_gradients_through_every_layer(self):
        layers = ModuleList([Linear(4, 4) for _ in range(3)])
        x = Tensor(np.random.randn(2, 4), requires_grad=True)
        for layer in layers:
            x = layer(x).relu()
        x.sum().backward()

        for linear in layers:
            assert linear.weight.grad is not None
            assert linear.bias.grad is not None
