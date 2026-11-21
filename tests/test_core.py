import os
import sys

import numpy as np
from scipy.special import expit
from scipy.special import logsumexp as sp_logsumexp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import core  # noqa: E402

RNG = np.random.default_rng(42)


def test_labels():
    assert core.LABELS == ["lua_dao", "rac", "hop_le"]


def test_sigmoid_extreme_finite_and_matches_expit():
    z = np.array([-1000.0, -50.0, -1.0, 0.0, 1.0, 50.0, 1000.0])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        s = core.sigmoid(z)
    assert np.all(np.isfinite(s))
    np.testing.assert_allclose(s, expit(z), rtol=1e-12, atol=0)
    z = RNG.normal(scale=20, size=1000)
    np.testing.assert_allclose(core.sigmoid(z), expit(z), rtol=1e-12, atol=1e-300)


def test_log_sigmoid():
    z = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])
    out = core.log_sigmoid(z)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out[1:4], np.log(expit(z[1:4])), rtol=1e-12)
    assert out[0] == -1000.0 and out[-1] == 0.0


def test_logsumexp_large_logits():
    Z = RNG.normal(scale=1e4, size=(50, 3))
    for axis in (-1, 0):
        for keepdims in (False, True):
            np.testing.assert_allclose(
                core.logsumexp(Z, axis=axis, keepdims=keepdims),
                sp_logsumexp(Z, axis=axis, keepdims=keepdims), rtol=1e-12)


def test_logsumexp_all_neg_inf_row():
    Z = np.array([[-np.inf, -np.inf], [0.0, 0.0]])
    out = core.logsumexp(Z)
    assert out[0] == -np.inf
    np.testing.assert_allclose(out[1], np.log(2.0))


def test_softmax_no_nan_rows_sum_to_one():
    Z = RNG.normal(scale=1e4, size=(100, 3))
    P = core.softmax(Z)
    assert not np.any(np.isnan(P))
    np.testing.assert_allclose(P.sum(axis=1), 1.0, rtol=1e-12)
    P0 = core.softmax(Z, axis=0)
    np.testing.assert_allclose(P0.sum(axis=0), 1.0, rtol=1e-12)
