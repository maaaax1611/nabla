from __future__ import annotations

import csv
import os

from nabla.logging import History


class TestHistory:
    def test_log_and_retrieve_single_metric(self):
        history = History()
        history.log(1, loss=0.5)
        history.log(2, loss=0.3)

        assert history.steps("loss") == [1, 2]
        assert history.values("loss") == [0.5, 0.3]

    def test_log_multiple_metrics_at_once(self):
        history = History()
        history.log(1, loss=0.5, lr=1e-3)

        assert history.values("loss") == [0.5]
        assert history.values("lr") == [1e-3]

    def test_metrics_can_be_logged_at_different_steps(self):
        # e.g. train loss every step, val loss only every eval_interval
        history = History()
        history.log(1, train_loss=1.0)
        history.log(2, train_loss=0.9)
        history.log(2, val_loss=1.1)

        assert history.steps("train_loss") == [1, 2]
        assert history.steps("val_loss") == [2]

    def test_last_returns_most_recent_value(self):
        history = History()
        history.log(1, loss=1.0)
        history.log(5, loss=0.2)

        assert history.last("loss") == 0.2

    def test_last_returns_none_for_unlogged_metric(self):
        history = History()
        assert history.last("nonexistent") is None

    def test_empty_metric_lookups_return_empty_lists(self):
        history = History()
        assert history.steps("nonexistent") == []
        assert history.values("nonexistent") == []

    def test_to_csv_writes_tidy_long_format(self, tmp_path):
        history = History()
        history.log(1, loss=0.5)
        history.log(2, loss=0.3)
        history.log(2, val_loss=0.6)

        path = os.path.join(tmp_path, "history.csv")
        history.to_csv(path)

        with open(path, newline="") as f:
            rows = list(csv.reader(f))

        assert rows[0] == ["step", "metric", "value"]
        data_rows = {(r[0], r[1]): r[2] for r in rows[1:]}
        assert data_rows[("1", "loss")] == "0.5"
        assert data_rows[("2", "loss")] == "0.3"
        assert data_rows[("2", "val_loss")] == "0.6"
