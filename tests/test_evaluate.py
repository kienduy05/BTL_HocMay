"""Kiểm tra evaluate.py (D5). Chỉ dữ liệu tổng hợp, seed 42."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from core import logsumexp, sigmoid, softmax  # noqa: E402
from evaluate import (adjust_prior, bootstrap_indices, bootstrap_metrics, cost_decision,  # noqa: E402
                      evaluate_probs, nb_log_ratio, paired_difference, recall_fpr_at_threshold,
                      resample_by_prior, scenario_prior, summarize, threshold_at_fpr, top_features)

C = np.array([[0, 5, 10], [1, 0, 1], [5, 3, 0]], dtype=float)


def _data(n=300, seed=42):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    Z = rng.normal(size=(n, 3)) + 2.0 * np.eye(3)[y]
    return softmax(Z, axis=1), y


# ---------------- quyết định theo chi phí (B5)
def test_cost_decision_01_cost_is_argmax():
    P, _ = _data()
    C01 = np.ones((3, 3)) - np.eye(3)
    assert np.array_equal(cost_decision(P, C01), P.argmax(1))


def test_cost_decision_binary_matches_b5_threshold():
    # K = 2, C = [[0, cFP], [cFN, 0]] (lớp 0 âm, lớp 1 dương): dương ⇔ p1 ≥ cFP/(cFP+cFN)
    cFP, cFN = 1.0, 5.0
    p1 = np.linspace(0.01, 0.99, 99)
    P = np.c_[1 - p1, p1]
    pred = cost_decision(P, np.array([[0, cFP], [cFN, 0]]))
    t = cFP / (cFP + cFN)
    assert np.array_equal(pred[np.abs(p1 - t) > 1e-9], (p1 >= t).astype(int)[np.abs(p1 - t) > 1e-9])


def test_cost_decision_hand_example():
    P = np.array([[0.2, 0.3, 0.5]])
    # chi phí kỳ vọng các cột: [0.3+2.5, 1+1.5, 2+0.3] = [2.8, 2.5, 2.3] → hop_le
    assert cost_decision(P, C)[0] == 2
    with pytest.raises(ValueError):
        cost_decision(P, np.eye(2))


# ---------------- ngưỡng FPR
def test_threshold_at_fpr_and_recall():
    score = np.array([0.1, 0.2, 0.3, 0.4, 0.9, 0.8, 0.7, 0.35])
    is_pos = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=bool)
    t, fpr, rec = threshold_at_fpr(score, is_pos, max_fpr=0.0)
    assert t == 0.7 and fpr == 0.0 and rec == 0.75
    assert recall_fpr_at_threshold(score, is_pos, t) == (0.75, 0.0)
    t2, fpr2, _ = threshold_at_fpr(score, is_pos, max_fpr=0.25)
    assert t2 == 0.35 and fpr2 == 0.25
    assert recall_fpr_at_threshold(score, is_pos, None) == (0.0, 0.0)


def test_threshold_none_when_only_inf():
    score = np.array([0.9, 0.9])
    t, fpr, rec = threshold_at_fpr(score, np.array([True, False]), max_fpr=0.0)
    assert t is None and fpr == 0.0 and rec == 0.0


# ---------------- evaluate_probs
def test_evaluate_probs_perfect():
    y = np.array([0, 1, 2, 0, 1, 2])
    P = np.full((6, 3), 1e-9)
    P[np.arange(6), y] = 1 - 2e-9
    m = evaluate_probs(P, y, C, threshold=0.5)
    assert m["f1_macro"] == 1.0 and m["chi_phi_tb"] == 0.0
    assert m["recall_lua_dao_at_fpr"] == 1.0 and m["fpr_at_threshold"] == 0.0
    assert m["ece_top_label"] < 1e-6 and m["log_loss"] < 1e-6
    assert m["recall_hop_le"] == 1.0 and m["bao_nham_lua_dao"] == 0.0


def test_evaluate_probs_cost_value():
    P, y = _data()
    m = evaluate_probs(P, y, C, threshold=0.5)
    pred = cost_decision(P, C)
    assert m["chi_phi_tb"] == pytest.approx(C[y, pred].mean())
    assert 0 <= m["ece_lua_dao"] <= 1 and np.isfinite(m["log_loss"])


# ---------------- bootstrap
def test_bootstrap_indices_seeded_and_shape():
    a, b = bootstrap_indices(50, 20, seed=42), bootstrap_indices(50, 20, seed=42)
    assert a.shape == (20, 50) and np.array_equal(a, b)
    assert a.min() >= 0 and a.max() < 50


def test_bootstrap_summary_contains_point_and_ci_order():
    P, y = _data()
    idx = bootstrap_indices(len(y), 200)
    boot = bootstrap_metrics(P, y, C, 0.5, idx)
    s = summarize(evaluate_probs(P, y, C, 0.5), boot)
    assert len(boot) == 200
    assert (s.ci_low <= s.ci_high).all()
    row = s.set_index("chi_so").loc["f1_macro"]
    assert row.ci_low <= row.goc <= row.ci_high
    assert row["std"] > 0


def test_paired_difference_identical_models_is_zero():
    P, y = _data()
    idx = bootstrap_indices(len(y), 50)
    boot = bootstrap_metrics(P, y, C, 0.5, idx)
    pt = evaluate_probs(P, y, C, 0.5)
    d = paired_difference(pt, pt, boot, boot)
    assert np.allclose(d[["hieu_goc", "mean", "ci_low", "ci_high"]].to_numpy(dtype=float), 0)
    assert d.ci_chua_0.all()


def test_paired_difference_detects_better_model():
    P, y = _data(n=400)
    noisy = softmax(np.log(P) + np.random.default_rng(0).normal(scale=3, size=P.shape), axis=1)
    idx = bootstrap_indices(len(y), 300)
    pa, pb = evaluate_probs(P, y, C, 0.5), evaluate_probs(noisy, y, C, 0.5)
    d = paired_difference(pa, pb, bootstrap_metrics(P, y, C, 0.5, idx),
                          bootstrap_metrics(noisy, y, C, 0.5, idx)).set_index("chi_so")
    assert d.loc["log_loss", "ci_high"] < 0 and not d.loc["log_loss", "ci_chua_0"]


# ---------------- điều chỉnh tỷ lệ lớp (B6)
def test_adjust_prior_binary_matches_b6_logit_shift():
    rng = np.random.default_rng(42)
    z = rng.normal(scale=3, size=100)
    p1 = sigmoid(z)
    P = np.c_[1 - p1, p1]
    pi, pi_new = 0.3, 0.05
    out = adjust_prior(P, [1 - pi, pi], [1 - pi_new, pi_new])
    expect = sigmoid(z + np.log(pi_new * (1 - pi) / (pi * (1 - pi_new))))
    assert np.allclose(out[:, 1], expect, atol=1e-12)


def test_adjust_prior_properties():
    P, _ = _data()
    pi = np.array([0.3, 0.4, 0.3])
    assert np.allclose(adjust_prior(P, pi, pi), P)
    out = adjust_prior(P, pi, [0.01, 0.6, 0.39])
    assert np.allclose(out.sum(1), 1) and np.all(out[:, 0] < P[:, 0])
    assert np.all(np.isfinite(adjust_prior(np.array([[1.0, 0.0, 0.0]]), pi, [0.01, 0.5, 0.49])))
    with pytest.raises(ValueError):
        adjust_prior(P, pi, [0.0, 0.5, 0.5])


def test_adjust_prior_softmax_equals_logit_shift():
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(20, 3))
    pi, pi_new = np.array([0.2, 0.5, 0.3]), np.array([0.05, 0.6, 0.35])
    assert np.allclose(adjust_prior(softmax(Z, axis=1), pi, pi_new),
                       softmax(Z + np.log(pi_new / pi), axis=1))


def test_scenario_prior_and_resample():
    pi = np.array([0.2866, 0.3792, 0.3342])
    s = scenario_prior(pi, 0.01)
    assert s[0] == 0.01 and s.sum() == pytest.approx(1)
    assert s[1] / s[2] == pytest.approx(pi[1] / pi[2])
    y = np.repeat([0, 1, 2], [30, 40, 50])
    idx = resample_by_prior(y, s, 200, np.random.default_rng(42))
    assert idx.size == 200
    counts = np.bincount(y[idx], minlength=3)
    assert counts[0] == 2 and counts.sum() == 200


# ---------------- đặc trưng
def test_nb_log_ratio_and_top_features():
    p = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8], [0.2, 0.7, 0.1]]).T   # (d=3, K=3) → cần (K, d)
    L = np.log(p.T)
    r = nb_log_ratio(L)
    assert r.shape == (3, 3)
    assert r[0, 0] == pytest.approx(np.log(0.7 / ((0.1 + 0.2) / 2)))
    df = top_features(r, np.array(["a", "b", "c"]), top=2)
    assert len(df) == 6
    assert df.query("lop == 'lua_dao' and hang == 1").dac_trung.item() == "a"
    assert np.allclose(logsumexp(L, axis=1), 0)
