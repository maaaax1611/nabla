from nabla.nn.loss import CrossEntropyLoss, MSELoss
from nabla.tensor import Tensor
import numpy as np


class TestMSELoss:
    def test_forward_matches_manual_computation(self):
        predictions = Tensor(np.array([1.0, 2.0, 3.0]))
        targets = Tensor(np.array([1.5, 2.5, 2.5]))
        loss = MSELoss()(predictions, targets)
        expected = np.mean((predictions.data - targets.data) ** 2)
        assert np.isclose(loss.data, expected)

    def test_forward_zero_when_predictions_match_targets(self):
        predictions = Tensor(np.array([1.0, 2.0, 3.0]))
        targets = Tensor(np.array([1.0, 2.0, 3.0]))
        loss = MSELoss()(predictions, targets)
        assert np.isclose(loss.data, 0.0)

    def test_backward_fills_prediction_gradient(self):
        predictions = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        targets = Tensor(np.array([1.5, 2.5, 2.5]))
        loss = MSELoss()(predictions, targets)
        loss.backward()

        n = predictions.data.shape[0]
        expected_grad = 2 * (predictions.data - targets.data) / n
        assert np.allclose(predictions.grad, expected_grad)

    def test_forward_rejects_mismatched_shapes(self):
        predictions = Tensor(np.array([1.0, 2.0]))
        targets = Tensor(np.array([1.0, 2.0, 3.0]))
        try:
            MSELoss()(predictions, targets)
            assert False, "expected ValueError"
        except ValueError:
            pass


class TestCrossEntropyLoss:
    def naive_softmax_cross_entropy(self, logits, targets):
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)
        picked = probs[np.arange(len(targets)), targets]
        return -np.mean(np.log(picked))

    def test_forward_matches_naive_reference(self):
        np.random.seed(0)
        logits = Tensor(np.random.randn(6, 4))
        targets = np.array([0, 1, 2, 3, 1, 0])

        loss = CrossEntropyLoss()(logits, targets)
        expected = self.naive_softmax_cross_entropy(logits.data, targets)
        assert np.isclose(loss.data, expected)

    def test_forward_is_low_for_confident_correct_predictions(self):
        logits = Tensor(np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]]))
        targets = np.array([0, 1])
        loss = CrossEntropyLoss()(logits, targets)
        assert loss.data < 1e-3

    def test_forward_is_high_for_confident_wrong_predictions(self):
        logits = Tensor(np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]]))
        targets = np.array([1, 0])  # both predictions confidently wrong
        loss = CrossEntropyLoss()(logits, targets)
        assert loss.data > 5.0

    def test_accepts_plain_array_targets(self):
        logits = Tensor(np.random.randn(4, 3))
        targets = [0, 1, 2, 1]  # plain list, not a Tensor
        loss = CrossEntropyLoss()(logits, targets)
        assert loss.data.shape == ()

    def test_backward_matches_numerical_gradient(self):
        np.random.seed(1)
        logits = Tensor(np.random.randn(5, 4), requires_grad=True)
        targets = np.array([0, 2, 1, 3, 0])

        def forward():
            return CrossEntropyLoss()(logits, targets).data

        loss = CrossEntropyLoss()(logits, targets)
        loss.backward()

        eps = 1e-5
        num_grad = np.zeros_like(logits.data)
        it = np.nditer(logits.data, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            original = logits.data[idx]
            logits.data[idx] = original + eps
            loss_plus = forward()
            logits.data[idx] = original - eps
            loss_minus = forward()
            logits.data[idx] = original
            num_grad[idx] = (loss_plus - loss_minus) / (2 * eps)

        assert np.allclose(logits.grad, num_grad, atol=1e-6)

    def test_backward_gradient_sums_to_zero_per_row(self):
        # d(loss)/d(logits) = (softmax - one_hot) / batch, and softmax rows
        # sum to 1 just like one_hot rows do, so each row's gradient sums to 0
        np.random.seed(2)
        logits = Tensor(np.random.randn(4, 5), requires_grad=True)
        targets = np.array([0, 1, 2, 3])

        loss = CrossEntropyLoss()(logits, targets)
        loss.backward()

        assert np.allclose(logits.grad.sum(axis=1), 0.0, atol=1e-8)

    def test_forward_rejects_wrong_logits_rank(self):
        logits = Tensor(np.random.randn(4, 3, 2))
        targets = np.array([0, 1, 2, 1])
        try:
            CrossEntropyLoss()(logits, targets)
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_forward_rejects_mismatched_target_length(self):
        logits = Tensor(np.random.randn(4, 3))
        targets = np.array([0, 1])
        try:
            CrossEntropyLoss()(logits, targets)
            assert False, "expected ValueError"
        except ValueError:
            pass
