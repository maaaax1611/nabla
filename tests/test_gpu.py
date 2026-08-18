"""GPU backend tests: skipped entirely on machines without CuPy/a working
CUDA device, so `uv run pytest` stays green on CPU-only machines (like CI).
Run these on a GPU machine to actually exercise them - see docs/19-gpu-support.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from nabla.backend import gpu_available, to_device
from nabla.nn.conv import Conv2D
from nabla.nn.embedding import Embedding
from nabla.nn.linear import Linear
from nabla.nn.loss import CrossEntropyLoss
from nabla.nn.module import Module
from nabla.nn.positional_encoding import PositionalEncoding
from nabla.nn.transformer import TransformerBlock
from nabla.nn.vit import VisionTransformer
from nabla.optim.adam import Adam
from nabla.tensor import Tensor

requires_gpu = pytest.mark.skipif(not gpu_available(), reason="no working CUDA device / CuPy install found")


@requires_gpu
class TestTensorDeviceTransfer:
    def test_to_cuda_then_cpu_round_trip_preserves_values(self):
        import cupy as cp

        x = Tensor(np.arange(6).astype(float).reshape(2, 3))
        x.to("cuda")
        assert isinstance(x.data, cp.ndarray)

        x.to("cpu")
        assert isinstance(x.data, np.ndarray)
        assert np.array_equal(x.data, np.arange(6).astype(float).reshape(2, 3))

    def test_to_cuda_moves_existing_gradient_too(self):
        import cupy as cp

        x = Tensor(np.ones((2, 2)), requires_grad=True)
        x.grad = np.ones((2, 2))
        x.to("cuda")
        assert isinstance(x.grad, cp.ndarray)

    def test_unknown_device_raises(self):
        x = Tensor(np.ones(3))
        with pytest.raises(ValueError):
            x.to("tpu")


@requires_gpu
class TestModuleDeviceTransfer:
    def test_to_cuda_moves_learnable_parameters(self):
        import cupy as cp

        layer = Linear(4, 3)
        layer.to("cuda")
        for param in layer.parameters():
            assert isinstance(param.data, cp.ndarray)

    def test_to_cuda_moves_non_tensor_buffers(self):
        # PositionalEncoding.table is a plain ndarray, not a Tensor - make
        # sure Module.to() finds and moves it too, not just parameters()
        import cupy as cp

        pe = PositionalEncoding(embed_dim=8, max_len=16)
        pe.to("cuda")
        assert isinstance(pe.table, cp.ndarray)

    def test_to_cuda_moves_fixed_non_grad_tensor_buffers(self):
        # a Tensor attribute with requires_grad=False is excluded from
        # parameters(), but Module.to() must still move it
        import cupy as cp

        class WithFixedBuffer(Module):
            def __init__(self) -> None:
                super().__init__()
                self.fixed = Tensor(np.ones((2, 2)))  # requires_grad=False

        model = WithFixedBuffer()
        model.to("cuda")
        assert isinstance(model.fixed.data, cp.ndarray)


@requires_gpu
class TestGPUForwardBackwardMatchesCPU:
    def test_linear_matches_cpu(self):
        np.random.seed(0)
        x_data = np.random.randn(4, 5)

        cpu_layer = Linear(5, 3)
        cpu_out = cpu_layer(Tensor(x_data))
        cpu_out.sum().backward()

        gpu_layer = Linear(5, 3)
        gpu_layer.weight.data = cpu_layer.weight.data.copy()
        gpu_layer.bias.data = cpu_layer.bias.data.copy()
        gpu_layer.to("cuda")
        gpu_out = gpu_layer(Tensor(x_data).to("cuda"))
        gpu_out.sum().backward()

        assert np.allclose(to_device(gpu_out.data, "cpu"), cpu_out.data, atol=1e-6)
        assert np.allclose(to_device(gpu_layer.weight.grad, "cpu"), cpu_layer.weight.grad, atol=1e-6)

    def test_conv2d_matches_cpu(self):
        np.random.seed(1)
        x_data = np.random.randn(2, 1, 8, 8)

        cpu_conv = Conv2D(1, 4, kernel_size=3, padding=1)
        cpu_out = cpu_conv(Tensor(x_data))
        cpu_out.sum().backward()

        gpu_conv = Conv2D(1, 4, kernel_size=3, padding=1)
        gpu_conv.weight.data = cpu_conv.weight.data.copy()
        gpu_conv.bias.data = cpu_conv.bias.data.copy()
        gpu_conv.to("cuda")
        gpu_out = gpu_conv(Tensor(x_data).to("cuda"))
        gpu_out.sum().backward()

        assert np.allclose(to_device(gpu_out.data, "cpu"), cpu_out.data, atol=1e-5)
        assert np.allclose(to_device(gpu_conv.weight.grad, "cpu"), cpu_conv.weight.grad, atol=1e-4)

    def test_embedding_matches_cpu(self):
        np.random.seed(2)
        indices_data = np.random.randint(0, 10, size=(3, 4))

        cpu_emb = Embedding(10, 6)
        cpu_out = cpu_emb(Tensor(indices_data))
        cpu_out.sum().backward()

        gpu_emb = Embedding(10, 6)
        gpu_emb.weight.data = cpu_emb.weight.data.copy()
        gpu_emb.to("cuda")
        gpu_out = gpu_emb(Tensor(indices_data).to("cuda"))
        gpu_out.sum().backward()

        assert np.allclose(to_device(gpu_out.data, "cpu"), cpu_out.data, atol=1e-6)
        assert np.allclose(to_device(gpu_emb.weight.grad, "cpu"), cpu_emb.weight.grad, atol=1e-6)

    def test_transformer_block_with_causal_mask_matches_cpu(self):
        np.random.seed(3)
        x_data = np.random.randn(2, 5, 8)
        mask = np.triu(np.full((5, 5), -1e9), k=1)

        cpu_block = TransformerBlock(embed_dim=8, num_heads=2, hidden_dim=16, dropout=0.0)
        cpu_out = cpu_block(Tensor(x_data), mask=mask)
        cpu_out.sum().backward()

        gpu_block = TransformerBlock(embed_dim=8, num_heads=2, hidden_dim=16, dropout=0.0)
        for gp, cp_param in zip(gpu_block.parameters(), cpu_block.parameters()):
            gp.data = cp_param.data.copy()
        gpu_block.to("cuda")
        gpu_out = gpu_block(Tensor(x_data).to("cuda"), mask=mask)
        gpu_out.sum().backward()

        assert np.allclose(to_device(gpu_out.data, "cpu"), cpu_out.data, atol=1e-4)

    def test_vision_transformer_end_to_end_on_gpu(self):
        # not a CPU-parity check (weights/dropout differ across separate
        # constructions) - just confirms the full model trains one step
        # on the GPU without error and every parameter gets a gradient
        np.random.seed(4)
        model = VisionTransformer(
            img_size=8, patch_size=4, in_channels=1, num_classes=5,
            embed_dim=12, num_heads=2, hidden_dim=16, num_layers=2, dropout=0.0,
        )
        model.to("cuda")
        optimizer = Adam(model.parameters(), lr=1e-3)
        criterion = CrossEntropyLoss()

        x = Tensor(np.random.randn(3, 1, 8, 8)).to("cuda")
        y = np.array([0, 1, 2])  # plain CPU labels, like straight from a DataLoader

        logits = model(x)
        loss = criterion(logits, y)
        model.zero_grad()
        loss.backward()
        optimizer.step()

        assert all(p.grad is not None for p in model.parameters())
