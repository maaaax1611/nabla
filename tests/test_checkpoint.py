from __future__ import annotations

import os

import numpy as np

from nabla.checkpoint import load_checkpoint, save_checkpoint
from nabla.nn.batchnorm import BatchNorm2D
from nabla.nn.linear import Linear
from nabla.optim.adam import Adam
from nabla.tensor import Tensor


class TestModuleStateDict:
    def test_state_dict_matches_parameter_order(self):
        layer = Linear(4, 3)
        state = layer.state_dict()
        params = layer.parameters()

        assert len(state["params"]) == len(params)
        for saved, param in zip(state["params"], params):
            assert np.array_equal(saved, param.data)

    def test_state_dict_is_a_copy_not_a_view(self):
        layer = Linear(4, 3)
        state = layer.state_dict()
        layer.weight.data[:] = 999.0

        assert not np.array_equal(state["params"][0], layer.weight.data)

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
            layer.load_state_dict({"params": [np.zeros(1)], "buffers": []})
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_state_dict_includes_batchnorm_running_stats(self):
        # the bug this guards against: running_mean/running_var are
        # plain ndarrays, not Tensor parameters, so a state_dict that
        # only walked parameters() would silently drop them - a loaded
        # checkpoint would then normalize with construction-time
        # defaults (mean=0, var=1) instead of what training learned
        bn = BatchNorm2D(4)
        bn.running_mean[:] = np.array([1.0, 2.0, 3.0, 4.0])
        bn.running_var[:] = np.array([5.0, 6.0, 7.0, 8.0])

        state = bn.state_dict()
        assert len(state["buffers"]) == 2
        assert np.array_equal(state["buffers"][0], bn.running_mean)
        assert np.array_equal(state["buffers"][1], bn.running_var)

    def test_load_state_dict_restores_batchnorm_running_stats(self):
        bn = BatchNorm2D(4)
        bn.running_mean[:] = np.array([1.0, 2.0, 3.0, 4.0])
        bn.running_var[:] = np.array([5.0, 6.0, 7.0, 8.0])
        state = bn.state_dict()

        fresh = BatchNorm2D(4)  # running_mean=0, running_var=1 (construction defaults)
        fresh.load_state_dict(state)

        assert np.array_equal(fresh.running_mean, bn.running_mean)
        assert np.array_equal(fresh.running_var, bn.running_var)


class TestSaveLoadCheckpoint(object):
    def test_round_trips_batchnorm_running_stats(self, tmp_path):
        path = os.path.join(tmp_path, "ckpt.pkl")
        bn = BatchNorm2D(4)
        bn.running_mean[:] = np.array([1.0, 2.0, 3.0, 4.0])
        bn.running_var[:] = np.array([5.0, 6.0, 7.0, 8.0])
        save_checkpoint(path, bn)

        fresh = BatchNorm2D(4)  # running_mean=0, running_var=1 (construction defaults)
        load_checkpoint(path, fresh)

        assert np.array_equal(fresh.running_mean, bn.running_mean)
        assert np.array_equal(fresh.running_var, bn.running_var)

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
