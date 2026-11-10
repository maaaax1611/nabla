"""Default dtype: float64 unless changed, and a float32 model stays float32
through forward, backward and the optimizer step."""

import numpy as np
import pytest

import nabla
from nabla import nn, optim
from nabla.tensor import Tensor


@pytest.fixture
def float32():
    nabla.set_default_dtype(np.float32)
    yield
    nabla.set_default_dtype(np.float64)


def test_default_is_float64():
    assert nabla.get_default_dtype() == np.float64
    assert Tensor([1.0, 2.0]).dtype == np.float64
    assert nn.Linear(2, 2).weight.dtype == np.float64


def test_rejects_non_float_default():
    with pytest.raises(ValueError):
        nabla.set_default_dtype(np.int32)


def test_python_floats_take_default_but_arrays_keep_theirs(float32):
    assert Tensor([1.0, 2.0]).dtype == np.float32
    assert Tensor(1.5).dtype == np.float32
    assert Tensor(np.ones(2, dtype=np.float64)).dtype == np.float64
    assert Tensor([1, 2]).dtype.kind == "i"


def test_astype_converts_data_and_grad():
    x = Tensor(np.ones(3), requires_grad=True)
    x.sum().backward()
    x.astype(np.float32)
    assert x.dtype == np.float32 and x.grad.dtype == np.float32


def _assert_all_float32(model, loss):
    assert loss.dtype == np.float32
    for p in model.parameters():
        assert p.dtype == np.float32
        assert p.grad is not None and p.grad.dtype == np.float32


def test_cnn_stays_float32(float32):
    np.random.seed(0)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2D(1, 4, kernel_size=3, padding=1)
            self.bn = nn.BatchNorm2D(4)
            self.fc = nn.Linear(4 * 4 * 4, 3)

        def forward(self, x):
            x = self.bn(self.conv(x)).relu().max_pool2d(2)
            return self.fc(x.reshape((x.shape[0], -1)))

    model = Net()
    x = Tensor(np.random.randn(5, 1, 8, 8).astype(np.float32))
    y = np.array([0, 1, 2, 0, 1])
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()
    _assert_all_float32(model, loss)
    optim.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    assert all(p.dtype == np.float32 and p.grad.dtype == np.float32 for p in model.parameters())
    assert model.bn.running_mean.dtype == np.float32

    model.eval()
    with nabla.no_grad():
        assert model(x).dtype == np.float32


def test_vision_transformer_stays_float32(float32):
    np.random.seed(0)
    model = nn.VisionTransformer(
        img_size=8, patch_size=4, in_channels=1, num_classes=3, embed_dim=8, num_heads=2, hidden_dim=16, num_layers=1
    )
    x = Tensor(np.random.randn(2, 1, 8, 8).astype(np.float32))
    loss = nn.CrossEntropyLoss()(model(x), np.array([0, 2]))
    loss.backward()
    _assert_all_float32(model, loss)


def test_text_transformer_stays_float32(float32):
    np.random.seed(0)

    class LM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(11, 8)
            self.pos = nn.PositionalEncoding(8, max_len=16)
            self.block = nn.TransformerBlock(8, 2, hidden_dim=16)
            self.head = nn.Linear(8, 11)

        def forward(self, ids, mask):
            return self.head(self.block(self.pos(self.embed(ids)), mask=mask))

    model = LM()
    ids = Tensor(np.random.randint(0, 11, (2, 6)))
    mask = np.triu(np.full((6, 6), -1e9), k=1)
    logits = model(ids, mask)
    loss = nn.CrossEntropyLoss()(logits.reshape((12, 11)), np.random.randint(0, 11, 12))
    loss.backward()
    _assert_all_float32(model, loss)
