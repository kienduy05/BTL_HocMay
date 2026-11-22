import os
import pickle
import sys
import time

import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import calibration as cal  # noqa: E402
import calibration_metrics as cm  # noqa: E402
import core  # noqa: E402

SEED = 42
NAMES = ["platt", "isotonic", "temperature"]


def make_logits(n, K=3, scale=2.0, seed=SEED):
    """Logit thật z (n, K) và y ~ Cat(softmax(z))."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(scale=scale, size=(n, K))
    P = core.softmax(Z, axis=1)
    y = (rng.uniform(size=(n, 1)) > np.cumsum(P, axis=1)).sum(axis=1)
    return Z, y


# ---------------------------------------------------------------- PAVA
@pytest.mark.parametrize("weighted", [False, True])
def test_pava_matches_sklearn(weighted):
    rng = np.random.default_rng(SEED)
    for n in (1, 2, 10, 500):
        y = rng.normal(size=n) + np.linspace(0, 2, n)
        w = rng.uniform(0.1, 3.0, size=n) if weighted else None
        ours = cal.pava(y, w)
        sk = IsotonicRegression().fit(np.arange(n), y, sample_weight=w).predict(np.arange(n))
        np.testing.assert_allclose(ours, sk, rtol=1e-10, atol=1e-12)
        assert np.all(np.diff(ours) >= -1e-12)
        ww = np.ones(n) if w is None else w
        assert np.sum(ww * ours) == pytest.approx(np.sum(ww * y), rel=1e-10, abs=1e-10)


def test_pava_monotone_input_unchanged():
    y = np.array([0.0, 0.1, 0.1, 0.5, 2.0, 7.0])
    np.testing.assert_array_equal(cal.pava(y), y)


def test_pava_fast():
    y = np.random.default_rng(SEED).normal(size=100_000)
    t0 = time.perf_counter()
    cal.pava(y)
    assert time.perf_counter() - t0 < 1.0


def test_isotonic_binary_ties_and_extrapolation():
    rng = np.random.default_rng(SEED)
    s = rng.integers(0, 20, size=400).astype(float)        # nhiều điểm trùng
    y = (rng.uniform(size=400) < s / 20).astype(float)
    xk, yk = cal.fit_isotonic_binary(s, y)
    assert np.all(np.diff(xk) > 0) and np.all(np.diff(yk) >= -1e-12)
    sk = IsotonicRegression(out_of_bounds="clip").fit(s, y)
    grid = np.linspace(-5, 25, 61)
    np.testing.assert_allclose(cal.predict_isotonic_binary(grid, xk, yk), sk.predict(grid),
                               atol=1e-12)


# ---------------------------------------------------------------- Platt
def test_platt_unsmoothed_matches_sklearn():
    rng = np.random.default_rng(SEED)
    s = rng.normal(size=1000)
    y = (rng.uniform(size=1000) < core.sigmoid(1.5 * s - 0.5)).astype(float)
    a, b = cal.fit_platt_binary(s, y, smooth=False)
    sk = LogisticRegression(C=np.inf, tol=1e-12, max_iter=10_000).fit(s[:, None], y)
    assert a == pytest.approx(sk.coef_[0, 0], abs=1e-5)
    assert b == pytest.approx(sk.intercept_[0], abs=1e-5)


def _platt_grad(a, b, s, y):
    n_pos, n_neg = (y == 1).sum(), (y == 0).sum()
    t = np.where(y == 1, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))
    r = core.sigmoid(a * s + b) - t
    return np.array([np.mean(r * s), np.mean(r)])


def test_platt_smooth_gradient_and_separable():
    rng = np.random.default_rng(SEED)
    s = rng.normal(size=500)
    y = (rng.uniform(size=500) < core.sigmoid(2 * s)).astype(float)
    a, b = cal.fit_platt_binary(s, y, smooth=True)
    assert np.max(np.abs(_platt_grad(a, b, s, y))) < 1e-8
    # khả tách hoàn toàn: vẫn hữu hạn, gradient ≈ 0
    ys = (s > 0).astype(float)
    a, b = cal.fit_platt_binary(s, ys, smooth=True)
    assert np.isfinite(a) and np.isfinite(b) and a > 0
    assert np.max(np.abs(_platt_grad(a, b, s, ys))) < 1e-8
    # vắng hẳn lớp dương
    a, b = cal.fit_platt_binary(s, np.zeros(500), smooth=True)
    assert np.isfinite(a) and np.isfinite(b)


# ---------------------------------------------------------------- Temperature
def test_temperature_recovers_T():
    rng = np.random.default_rng(SEED)
    n = 20_000
    Z = rng.normal(scale=3.0, size=(n, 3))
    P = core.softmax(Z / 2.0, axis=1)
    y = (rng.uniform(size=(n, 1)) > np.cumsum(P, axis=1)).sum(axis=1)
    T = cal.fit_temperature(Z, y)
    assert abs(T - 2.0) < 0.1
    grid = np.linspace(0.1, 10, 400)
    nll_grid = min(cal.temperature_nll(Z, y, t) for t in grid)
    assert cal.temperature_nll(Z, y, T) <= nll_grid + 1e-12


def test_temperature_large_logits_no_nan():
    Z, y = make_logits(2000)
    T = cal.fit_temperature(Z * 1e3, y)
    assert np.isfinite(T) and T > 0
    P = cal.TemperatureCalibrator().fit(Z * 1e3, y).predict_proba(Z * 1e3)
    assert not np.any(np.isnan(P))


# ---------------------------------------------------------------- Calibrators
@pytest.mark.parametrize("name", NAMES + ["none"])
def test_calibrator_output_and_missing_class_and_pickle(name):
    Z, y = make_logits(600)
    for yy in (y, np.where(y == 2, 0, y)):                 # lần 2: thiếu hẳn lớp 2
        c = cal.get_calibrator(name, n_classes=3).fit(Z, yy)
        P = c.predict_proba(Z)
        assert P.shape == (600, 3)
        assert not np.any(np.isnan(P))
        assert np.all((P >= 0) & (P <= 1))
        np.testing.assert_allclose(P.sum(axis=1), 1.0, rtol=1e-12)
        c2 = pickle.loads(pickle.dumps(c))
        np.testing.assert_array_equal(c2.predict_proba(Z), P)


def test_calibrator_attributes_and_errors():
    Z, y = make_logits(300)
    p = cal.PlattCalibrator().fit(Z, y)
    assert p.a_.shape == (3,) and p.b_.shape == (3,)
    iso = cal.IsotonicCalibrator().fit(Z, y)
    assert len(iso.knots_) == 3
    t = cal.TemperatureCalibrator().fit(Z, y)
    assert isinstance(t.T_, float) and t.T_ > 0
    with pytest.raises(ValueError):
        cal.get_calibrator("abc")
    with pytest.raises(ValueError):
        cal.PlattCalibrator(n_classes=4).fit(Z, y)         # K không khớp
    with pytest.raises(ValueError):
        cal.PlattCalibrator().fit(Z, y + 1)                 # nhãn 3 ngoài [0, 3)


def test_calibrator_sample_weight():
    Z, y = make_logits(400)
    w = np.random.default_rng(SEED).uniform(0.5, 2.0, size=400)
    for name in NAMES:
        P = cal.get_calibrator(name).fit(Z, y, sample_weight=w).predict_proba(Z)
        np.testing.assert_allclose(P.sum(axis=1), 1.0, rtol=1e-12)
    # trọng số nguyên ≡ lặp mẫu (temperature)
    wi = np.random.default_rng(SEED).integers(1, 4, size=400)
    T_w = cal.fit_temperature(Z, y, sample_weight=wi)
    T_rep = cal.fit_temperature(np.repeat(Z, wi, axis=0), np.repeat(y, wi))
    assert T_w == pytest.approx(T_rep, rel=1e-8)


def test_temperature_keeps_argmax():
    Z, y = make_logits(1000)
    P = cal.TemperatureCalibrator().fit(3 * Z, y).predict_proba(3 * Z)
    np.testing.assert_array_equal(P.argmax(axis=1), Z.argmax(axis=1))


@pytest.mark.parametrize("name", ["platt", "isotonic"])
def test_columns_monotone_before_normalization(name):
    Z, y = make_logits(1000)
    c = cal.get_calibrator(name).fit(Z, y)
    if name == "platt":
        assert np.all(c.a_ > 0)
    for k in range(3):
        order = np.argsort(Z[:, k])
        col = c._column_proba(Z)[order, k]
        assert np.all(np.diff(col) >= -1e-12)


@pytest.mark.parametrize("name", NAMES)
def test_calibration_improves_overconfident(name):
    Z, y = make_logits(20_000, scale=1.0)
    S = 3.0 * Z                                             # cố ý quá tự tin
    half = Z.shape[0] // 2
    SA, yA, SB, yB = S[:half], y[:half], S[half:], y[half:]
    P0 = cal.get_calibrator("none").fit(SA, yA).predict_proba(SB)
    P1 = cal.get_calibrator(name).fit(SA, yA).predict_proba(SB)
    assert cm.ece(P1, yB) < cm.ece(P0, yB)
    assert cm.log_loss(P1, yB) < cm.log_loss(P0, yB)
