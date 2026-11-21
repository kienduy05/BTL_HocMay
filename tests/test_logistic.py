import os
import sys

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import logistic  # noqa: E402


def make_data(n=200, d=3, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w_true = rng.normal(size=d)
    y = (rng.random(n) < logistic.predict_proba(X, w_true, 0.3)).astype(int)
    return X, y


def make_separable(seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(40, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    X += np.where(y[:, None] == 1, 0.5, -0.5)   # tạo lề rõ ràng
    return X, y


@pytest.mark.parametrize("lam", [0.0, 0.1])
def test_grad_hess_finite_difference(lam):
    X, y = make_data(n=50, d=3)
    rng = np.random.default_rng(42)
    w, b = rng.normal(size=3), 0.2
    _, g, H = logistic.loss_grad_hess(w, b, X, y, lam)
    theta = np.r_[w, b]
    eps = 1e-6
    g_fd = np.zeros(4)
    H_fd = np.zeros((4, 4))
    for j in range(4):
        e = np.zeros(4)
        e[j] = eps
        lp, gp, _ = logistic.loss_grad_hess((theta + e)[:3], (theta + e)[3], X, y, lam)
        lm, gm, _ = logistic.loss_grad_hess((theta - e)[:3], (theta - e)[3], X, y, lam)
        g_fd[j] = (lp - lm) / (2 * eps)
        H_fd[:, j] = (gp - gm) / (2 * eps)
    np.testing.assert_allclose(g, g_fd, atol=1e-7)
    np.testing.assert_allclose(H, H_fd, atol=1e-7)


@pytest.mark.parametrize("lam", [0.0, 0.05])
def test_matches_sklearn(lam):
    X, y = make_data()
    n = X.shape[0]
    w, b, hist = logistic.fit_irls(X, y, lam=lam)
    assert hist["grad_norm"][-1] < 1e-8
    # sklearn tính loss theo TỔNG: λ = 1/(n·C); λ = 0 ứng với không phạt (C = inf)
    C = np.inf if lam == 0 else 1.0 / (n * lam)
    sk = LogisticRegression(C=C, solver="lbfgs", tol=1e-12, max_iter=10000).fit(X, y)
    np.testing.assert_allclose(w, sk.coef_.ravel(), atol=1e-5)
    np.testing.assert_allclose(b, sk.intercept_[0], atol=1e-5)
    np.testing.assert_allclose(logistic.predict_proba(X, w, b),
                               sk.predict_proba(X)[:, 1], atol=1e-6)


def test_c1_two_steps():
    X = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])[:, None]
    y = np.array([0, 0, 1, 0, 1])
    w, b, hist = logistic.fit_irls(X, y, w_init=np.zeros(1), b_init=0.0, max_iter=2)
    for key in ("w", "b", "loss", "norm_w", "grad_norm"):
        assert len(hist[key]) == 3
        assert np.all(np.isfinite(np.asarray(hist[key], dtype=float)))
    np.testing.assert_allclose(hist["loss"][0], np.log(2.0))
    np.testing.assert_allclose(hist["w"][2], w)
    assert hist["b"][2] == b


def test_separable_no_penalty_norm_grows():
    X, y = make_separable()
    w, b, hist = logistic.fit_irls(X, y, lam=0.0, max_iter=30)
    norm_w = np.asarray(hist["norm_w"])
    assert np.all(np.isfinite(norm_w)) and np.all(np.isfinite(w)) and np.isfinite(b)
    assert np.all(np.isfinite(hist["loss"]))
    assert np.all(np.diff(norm_w) >= 0)
    assert norm_w[-1] > 10 * norm_w[1]


def test_separable_no_penalty_runs_to_max_iter_with_tol_zero():
    X, y = make_separable()
    w, b, hist = logistic.fit_irls(X, y, lam=0.0, tol=0.0, max_iter=100)
    assert len(hist["norm_w"]) == 101
    assert np.all(np.isfinite(hist["norm_w"])) and np.all(np.isfinite(hist["loss"]))
    assert np.all(np.diff(hist["norm_w"]) >= 0)


def test_separable_with_penalty_converges():
    X, y = make_separable()
    tol = 1e-8
    w, b, hist = logistic.fit_irls(X, y, lam=0.01, tol=tol)
    assert hist["grad_norm"][-1] < tol
    assert np.all(np.isfinite(w))
