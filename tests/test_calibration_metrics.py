import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from sklearn.metrics import log_loss as sk_log_loss  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import calibration_metrics as cm  # noqa: E402

SEED = 42

# Ví dụ tính tay, n_bins = 5 (cạnh 0; 0,2; 0,4; 0,6; 0,8; 1)
P3 = np.array([[0.50, 0.30, 0.20],
               [0.20, 0.70, 0.10],
               [0.10, 0.10, 0.80],
               [0.30, 0.38, 0.32],
               [0.05, 0.05, 0.90]])
Y3 = np.array([0, 1, 0, 2, 2])


def test_hand_binary():
    p = np.array([0.1, 0.3, 0.3, 0.7, 0.9, 1.0])
    y = np.array([0, 0, 1, 1, 1, 0])
    # b1: |0−0,1|·1; b2: |0,5−0,3|·2; b4: |1−0,7|·1; b5: |0,5−0,95|·2  (chia 6)
    expected = (0.1 + 0.4 + 0.3 + 0.9) / 6
    assert cm.ece(p, y, n_bins=5, kind="binary") == pytest.approx(expected, abs=1e-12)


def test_hand_top_label():
    # conf = [0,5 ✓, 0,7 ✓, 0,8 ✗, 0,38 ✗, 0,9 ✓] → b2: 0,38; b3: 0,5; b4: |0,5−0,75|·2; b5: 0,1
    expected = (0.38 + 0.5 + 0.5 + 0.1) / 5
    assert cm.ece(P3, Y3, n_bins=5, kind="top_label") == pytest.approx(expected, abs=1e-12)


def test_hand_classwise():
    e0 = 0.65 / 5 + 0.3 / 5 + 0.5 / 5               # lớp 0
    e1 = 0.075 * 2 / 5 + 0.34 * 2 / 5 + 0.3 / 5       # lớp 1
    e2 = 0.15 * 2 / 5 + 0.68 / 5 + 0.8 / 5 + 0.1 / 5  # lớp 2
    expected = (e0 + e1 + e2) / 3
    assert cm.ece(P3, Y3, n_bins=5, kind="classwise") == pytest.approx(expected, abs=1e-12)
    for k, ek in enumerate([e0, e1, e2]):
        assert cm.ece(P3[:, k], Y3 == k, n_bins=5, kind="binary") == pytest.approx(ek, abs=1e-12)


@pytest.mark.parametrize("M", [5, 10, 15])
def test_bin_assignment_edges(M):
    def bin_of(p):
        c = cm.reliability_curve(np.array([p]), np.array([1]), n_bins=M, kind="binary")["count"]
        return int(np.argmax(c)) + 1  # 1-index

    assert bin_of(0.0) == 1
    assert bin_of(1.0) == M
    for m in range(1, M + 1):
        assert bin_of(m / M) == m
        assert bin_of(m / M - 1e-9) == m
        if m < M:
            assert bin_of(m / M + 1e-9) == m + 1


def test_perfectly_calibrated_small_ece():
    rng = np.random.default_rng(SEED)
    p = rng.uniform(size=10**6)
    y = (rng.uniform(size=p.size) < p).astype(int)
    assert cm.ece(p, y, kind="binary") < 0.01


def test_always_one_70_percent():
    rng = np.random.default_rng(SEED)
    n = 1000
    y = np.zeros(n, dtype=int)
    y[rng.choice(n, size=700, replace=False)] = 1
    assert cm.ece(np.ones(n), y, kind="binary") == pytest.approx(0.3, abs=1e-12)
    P = np.tile([1.0, 0.0, 0.0], (n, 1))
    assert cm.ece(P, 1 - y, kind="top_label") == pytest.approx(0.3, abs=1e-12)


@pytest.mark.parametrize("kind", ["binary", "top_label", "classwise"])
def test_ece_equals_sum_from_curve(kind):
    rng = np.random.default_rng(SEED)
    n, K = 500, 3
    P = rng.dirichlet(np.ones(K), size=n)
    y = rng.integers(0, K, size=n)
    if kind == "binary":
        P, y = P[:, 0], (y == 0).astype(int)

    def from_curve(c):
        keep = c["count"] > 0
        assert np.all(np.isnan(c["conf"][~keep])) and np.all(np.isnan(c["acc"][~keep]))
        assert c["edges"].shape == (16,) and c["count"].sum() == n
        return np.sum(c["count"][keep] / n * np.abs(c["acc"][keep] - c["conf"][keep]))

    curve = cm.reliability_curve(P, y, kind=kind)
    if kind == "classwise":
        assert isinstance(curve, list) and len(curve) == K
        expected = np.mean([from_curve(c) for c in curve])
    else:
        expected = from_curve(curve)
    assert cm.ece(P, y, kind=kind) == pytest.approx(expected, abs=1e-14)


def test_bad_shape_kind_raise():
    P = np.full((4, 3), 1 / 3)
    y = np.array([0, 1, 2, 0])
    with pytest.raises(ValueError):
        cm.ece(P[:, 0], y, kind="top_label")
    with pytest.raises(ValueError):
        cm.ece(P[:, 0], y, kind="classwise")
    with pytest.raises(ValueError):
        cm.ece(P, y, kind="binary")
    with pytest.raises(ValueError):
        cm.ece(P, y, kind="abc")
    with pytest.raises(ValueError):
        cm.ece(P, y[:3])
    with pytest.raises(ValueError):
        cm.ece(P[:, 0], np.array([0, 1, 2, 0]), kind="binary")   # nhãn binary ngoài {0, 1}
    with pytest.raises(ValueError):
        cm.ece(P * 4, y)                                          # xác suất > 1
    with pytest.raises(ValueError):
        cm.reliability_curve(P, y, n_bins=0)


def test_plot_reliability_agg():
    rng = np.random.default_rng(SEED)
    P = rng.dirichlet(np.ones(3), size=300)
    y = rng.integers(0, 3, size=300)
    n_figs = len(plt.get_fignums())
    ax = cm.plot_reliability(P, y, label="trước")
    ax2 = cm.plot_reliability(P ** 0.5 / (P ** 0.5).sum(1, keepdims=True), y, ax=ax, label="sau")
    assert ax2 is ax and isinstance(ax, matplotlib.axes.Axes)
    assert len(plt.get_fignums()) == n_figs + 1
    texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("trước" in t and "ECE" in t for t in texts)
    assert any("sau" in t for t in texts)
    _, ax3 = plt.subplots()
    cm.plot_reliability(P, y, kind="classwise", ax=ax3, show_hist=False)
    cm.plot_reliability(P[:, 0], y == 0, kind="binary", ax=ax3)
    plt.close("all")


def test_log_loss_matches_sklearn():
    rng = np.random.default_rng(SEED)
    P = rng.dirichlet(np.ones(3), size=200)
    y = rng.integers(0, 3, size=200)
    assert cm.log_loss(P, y) == pytest.approx(sk_log_loss(y, P, labels=[0, 1, 2]), rel=1e-10)
    p = rng.uniform(0.01, 0.99, size=200)
    yb = rng.integers(0, 2, size=200)
    assert cm.log_loss(p, yb) == pytest.approx(sk_log_loss(yb, p), rel=1e-10)
    # kẹp eps: xác suất 0 không ra inf
    assert np.isfinite(cm.log_loss(np.array([[1.0, 0.0, 0.0]]), np.array([1])))
