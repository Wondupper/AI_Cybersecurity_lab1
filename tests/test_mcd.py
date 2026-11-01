import numpy as np
import pytest

from mcd import mcd


def test_known_value():
    a, b = np.zeros((1, 24)), np.zeros((1, 24))
    b[0, 1] = 1.0
    assert mcd(a, b) == pytest.approx(6.1419, abs=1e-3)


def test_identical_is_zero_and_coef0_ignored():
    a = np.random.default_rng(0).normal(size=(5, 24))
    b = a.copy()
    b[:, 0] += 100.0
    assert mcd(a, b) == 0.0


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        mcd(np.zeros((2, 24)), np.zeros((3, 24)))
    with pytest.raises(ValueError):
        mcd(np.full((1, 24), np.nan), np.zeros((1, 24)))
