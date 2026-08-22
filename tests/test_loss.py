from nabla.nn.loss import CrossEntropyLoss, DiceLoss, MSELoss
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


class TestDiceLoss:
    def test_forward_is_zero_for_perfect_overlap(self):
        predictions = Tensor(np.array([[1.0, 0.0, 1.0, 0.0]]))
        targets = Tensor(np.array([[1.0, 0.0, 1.0, 0.0]]))
        loss = DiceLoss()(predictions, targets)
        assert np.isclose(loss.data, 0.0, atol=1e-4)

    def test_forward_is_zero_for_two_empty_masks(self):
        # both mask and prediction are entirely "no tumor" - a perfect
        # match, not the 0/0 degenerate case eps is there to avoid
        predictions = Tensor(np.zeros((1, 4)))
        targets = Tensor(np.zeros((1, 4)))
        loss = DiceLoss()(predictions, targets)
        assert np.isclose(loss.data, 0.0, atol=1e-4)

    def test_forward_is_near_one_for_complete_mismatch(self):
        predictions = Tensor(np.array([[1.0, 1.0, 0.0, 0.0]]))
        targets = Tensor(np.array([[0.0, 0.0, 1.0, 1.0]]))
        loss = DiceLoss()(predictions, targets)
        assert np.isclose(loss.data, 1.0, atol=1e-4)

    def test_forward_matches_naive_reference(self):
        np.random.seed(0)
        eps = 1e-6
        predictions = Tensor(np.random.rand(3, 2, 4, 4))
        targets = Tensor((np.random.rand(3, 2, 4, 4) > 0.5).astype(float))

        loss = DiceLoss(eps=eps)(predictions, targets)

        axes = tuple(range(1, predictions.data.ndim))
        intersection = (predictions.data * targets.data).sum(axis=axes)
        union = predictions.data.sum(axis=axes) + targets.data.sum(axis=axes)
        expected = (1 - (2 * intersection + eps) / (union + eps)).mean()
        assert np.isclose(loss.data, expected)

    def test_backward_matches_numerical_gradient(self):
        np.random.seed(1)
        eps = 1e-6
        predictions = Tensor(np.random.rand(2, 3, 3), requires_grad=True)
        targets = Tensor((np.random.rand(2, 3, 3) > 0.5).astype(float))

        def forward():
            return DiceLoss(eps=eps)(predictions, targets).data

        loss = DiceLoss(eps=eps)(predictions, targets)
        loss.backward()

        num_eps = 1e-6
        num_grad = np.zeros_like(predictions.data)
        it = np.nditer(predictions.data, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            original = predictions.data[idx]
            predictions.data[idx] = original + num_eps
            loss_plus = forward()
            predictions.data[idx] = original - num_eps
            loss_minus = forward()
            predictions.data[idx] = original
            num_grad[idx] = (loss_plus - loss_minus) / (2 * num_eps)

        assert np.allclose(predictions.grad, num_grad, atol=1e-6)

    def test_backward_target_gradient_is_zero(self):
        predictions = Tensor(np.random.rand(2, 4), requires_grad=True)
        targets = Tensor(np.random.rand(2, 4) > 0.5, requires_grad=True)
        loss = DiceLoss()(predictions, targets)
        loss.backward()
        assert np.array_equal(targets.grad, np.zeros_like(targets.data, dtype=float))

    def test_forward_rejects_mismatched_shapes(self):
        predictions = Tensor(np.random.rand(2, 4))
        targets = Tensor(np.random.rand(2, 3))
        try:
            DiceLoss()(predictions, targets)
            assert False, "expected ValueError"
        except ValueError:
            pass
