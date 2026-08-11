from nabla.init import he, xavier, zeros
import numpy as np


class TestHe:
    def test_shape(self):
        w = he((4, 8))
        assert w.shape == (4, 8)

    def test_std_matches_formula_for_linear_shape(self):
        np.random.seed(0)
        fan_in, fan_out = 256, 128
        w = he((fan_in, fan_out))
        expected_std = np.sqrt(2.0 / fan_in)
        assert np.isclose(w.std(), expected_std, rtol=0.05)

    def test_std_matches_formula_for_conv_shape(self):
        np.random.seed(1)
        out_channels, in_channels, kh, kw = 16, 8, 3, 3
        w = he((out_channels, in_channels, kh, kw))
        expected_std = np.sqrt(2.0 / (in_channels * kh * kw))
        assert w.shape == (out_channels, in_channels, kh, kw)
        assert np.isclose(w.std(), expected_std, rtol=0.05)

    def test_not_deterministic_across_calls(self):
        np.random.seed(2)
        w1 = he((4, 4))
        w2 = he((4, 4))
        assert not np.array_equal(w1, w2)


class TestXavier:
    def test_shape(self):
        w = xavier((4, 8))
        assert w.shape == (4, 8)

    def test_values_within_expected_range_for_linear_shape(self):
        np.random.seed(3)
        fan_in, fan_out = 100, 50
        w = xavier((fan_in, fan_out))
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        assert w.min() >= -limit
        assert w.max() <= limit

    def test_values_within_expected_range_for_conv_shape(self):
        np.random.seed(4)
        out_channels, in_channels, kh, kw = 8, 4, 3, 3
        w = xavier((out_channels, in_channels, kh, kw))
        fan_in = in_channels * kh * kw
        fan_out = out_channels * kh * kw
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        assert w.shape == (out_channels, in_channels, kh, kw)
        assert w.min() >= -limit
        assert w.max() <= limit


class TestZeros:
    def test_shape(self):
        w = zeros((4, 8))
        assert w.shape == (4, 8)

    def test_all_zero(self):
        w = zeros((3, 3, 3, 3))
        assert np.all(w == 0)
