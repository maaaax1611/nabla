from __future__ import annotations

import os

import numpy as np

from nabla.checkpoint import load_checkpoint, save_checkpoint
from nabla.nn.linear import Linear
from nabla.optim.adam import Adam
from nabla.tensor import Tensor


class TestModuleStateDict:
    def test_state_dict_matches_parameter_order(self):
        layer = Linear(4, 3)
        state = layer.state_dict()
        params = layer.parameters()

        assert len(state) == len(params)
        for saved, param in zip(state, params):
            assert np.array_equal(saved, param.data)

    def test_state_dict_is_a_copy_not_a_view(self):
        layer = Linear(4, 3)
        state = layer.state_dict()
        layer.weight.data[:] = 999.0

        assert not np.array_equal(state[0], layer.weight.data)

    def test_load_state_dict_restores_values(self):
        layer = Linear(4, 3)
        original_weight = layer.weight.data.copy()
        state = layer.state_dict()

        layer.weight.data[:] = 0.0
        layer.load_state_dict(state)

        assert np.array_equal(layer.weight.data, original_weight)

    def test_load_state_dict_rejects_mismatched_length(self):
        layer = Linear(4, 3)
        try:
            layer.load_state_dict([np.zeros(1)])
            assert False, "expected ValueError"
        except ValueError:
            pass


class TestSaveLoadCheckpoint(object):
    def test_round_trips_model_weights(self, tmp_path):
        path = os.path.join(tmp_path, "ckpt.pkl")
        model = Linear(4, 3)
        save_checkpoint(path, model)

        model.weight.data[:] = 0.0
        original = Linear(4, 3)  # fresh model with different random weights
        load_checkpoint(path, original)

        assert not np.array_equal(original.weight.data, np.zeros((4, 3)))

    def test_round_trips_optimizer_momentum(self, tmp_path):
        path = os.path.join(tmp_path, "ckpt.pkl")
        model = Linear(4, 3)
        optimizer = Adam(model.parameters(), lr=1e-3)

        for _ in range(3):
            for p in model.parameters():
                p.grad = np.random.randn(*p.data.shape)
            optimizer.step()

        save_checkpoint(path, model, optimizer)
        t_before = optimizer.t
        m_before = [m.copy() for m in optimizer.m]

        fresh_model = Linear(4, 3)
        fresh_optimizer = Adam(fresh_model.parameters(), lr=1e-3)
        load_checkpoint(path, fresh_model, fresh_optimizer)

        assert fresh_optimizer.t == t_before
        for restored, expected in zip(fresh_optimizer.m, m_before):
            assert np.allclose(restored, expected)

    def test_round_trips_metadata(self, tmp_path):
        path = os.path.join(tmp_path, "ckpt.pkl")
        model = Linear(4, 3)
        save_checkpoint(path, model, epoch=12, step=4000, best_val_loss=0.42)

        fresh_model = Linear(4, 3)
        metadata = load_checkpoint(path, fresh_model)

        assert metadata == {"epoch": 12, "step": 4000, "best_val_loss": 0.42}

    def test_load_without_optimizer_is_optional(self, tmp_path):
        path = os.path.join(tmp_path, "ckpt.pkl")
        model = Linear(4, 3)
        optimizer = Adam(model.parameters(), lr=1e-3)
        save_checkpoint(path, model, optimizer)

        fresh_model = Linear(4, 3)
        metadata = load_checkpoint(path, fresh_model)  # no optimizer passed
        assert metadata == {}
