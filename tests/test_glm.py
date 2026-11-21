import os
import sys

import numpy as np
import pytest
from sklearn.linear_model import PoissonRegressor, Ridge

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import glm  # noqa: E402
import logistic  # noqa: E402

FAMILIES = ["bernoulli", "poisson", "gaussian"]


def make_data(family, n=300, d=3, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w_true = 0.5 * rng.normal(size=d)
    eta = X @ w_true + 0.3
    if family == "bernoulli":
        y = (rng.random(n) < glm.predict(X, w_true, 0.3, "bernoulli")).astype(int)
    elif family == "poisson":
        y = rng.poisson(glm.predict(X, w_true, 0.3, "poisson"))
    else:
        y = eta + rng.normal(scale=0.5, size=n)
    return X, y


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("lam", [0.0, 0.1])
def test_grad_finite_difference(family, lam):
    X, y = make_data(family, n=60)
    rng = np.random.default_rng(42)
    w, b = 0.3 * rng.normal(size=3), 0.2
    _, g = glm.loss_grad(w, b, X, y, family, lam)
    theta = np.r_[w, b]
    eps = 1e-6
    g_fd = np.zeros(4)
    for j in range(4):
        e = np.zeros(4)
        e[j] = eps
        lp, _ = glm.loss_grad((theta + e)[:3], (theta + e)[3], X, y, family, lam)
        lm, _ = glm.loss_grad((theta - e)[:3], (theta - e)[3], X, y, family, lam)
        g_fd[j] = (lp - lm) / (2 * eps)
    np.testing.assert_allclose(g, g_fd, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("lam", [0.0, 0.1])
def test_converges_to_stationary_point(family, lam):
    X, y = make_data(family)
    w, b, hist = glm.fit_glm(X, y, family, lam=lam)
    assert hist["grad_norm"][-1] < 1e-8
    assert w.shape == (3,) and isinstance(b, float)
    assert np.all(np.diff(hist["loss"]) <= 1e-12)   # loss không tăng


@pytest.mark.parametrize("lam", [0.0, 0.1])
def test_bernoulli_matches_logistic(lam):
    X, y = make_data("bernoulli")
    w, b, _ = glm.fit_glm(X, y, "bernoulli", lam=lam)
    w_ref, b_ref, _ = logistic.fit_irls(X, y, lam=lam)
    np.testing.assert_allclose(w, w_ref, atol=1e-6)
    assert abs(b - b_ref) < 1e-6


@pytest.mark.parametrize("lam", [0.0, 0.1])
def test_poisson_matches_sklearn(lam):
    X, y = make_data("poisson")
    w, b, _ = glm.fit_glm(X, y, "poisson", lam=lam)
    # sklearn: (1/2n)·deviance + (α/2)‖w‖² = Loss của glm.py + hằng số, với α = λ
    ref = PoissonRegressor(alpha=lam, solver="newton-cholesky", tol=1e-12, max_iter=1000)
    ref.fit(X, y)
    np.testing.assert_allclose(w, ref.coef_, atol=1e-6)
    assert abs(b - ref.intercept_) < 1e-6


@pytest.mark.parametrize("family", FAMILIES)
def test_matches_statsmodels(family):
    """Đối chiếu với statsmodels.GLM (λ = 0). Bỏ qua nếu chưa cài statsmodels."""
    sm = pytest.importorskip("statsmodels.api")
    sm_family = {"bernoulli": sm.families.Binomial(),
                 "poisson": sm.families.Poisson(),
                 "gaussian": sm.families.Gaussian()}[family]
    X, y = make_data(family)
    w, b, _ = glm.fit_glm(X, y, family)
    Xc = sm.add_constant(X, prepend=True, has_constant="add")
    ref = sm.GLM(y, Xc, family=sm_family).fit(tol=1e-12, maxiter=200)
    np.testing.assert_allclose(w, ref.params[1:], atol=1e-6)
    assert abs(b - ref.params[0]) < 1e-6
    np.testing.assert_allclose(glm.predict(X, w, b, family), ref.fittedvalues, atol=1e-6)


def test_gaussian_matches_least_squares_one_step():
    X, y = make_data("gaussian")
    w, b, hist = glm.fit_glm(X, y, "gaussian")
    Xa = np.hstack([X, np.ones((X.shape[0], 1))])
    theta_ref = np.linalg.lstsq(Xa, y, rcond=None)[0]
    np.testing.assert_allclose(np.r_[w, b], theta_ref, atol=1e-10)
    assert len(hist["loss"]) == 2          # IRLS với liên kết đồng nhất: đúng 1 bước


def test_gaussian_l2_matches_ridge():
    X, y = make_data("gaussian")
    lam = 0.1
    w, b, _ = glm.fit_glm(X, y, "gaussian", lam=lam)
    # Ridge: ‖y − Xw − b‖² + α‖w‖²  ⇔  nhân Loss của glm.py với 2n, α = nλ
    ref = Ridge(alpha=X.shape[0] * lam).fit(X, y)
    np.testing.assert_allclose(w, ref.coef_, atol=1e-8)
    assert abs(b - ref.intercept_) < 1e-8


def test_poisson_far_init_no_overflow():
    X, y = make_data("poisson")
    w, b, hist = glm.fit_glm(X, y, "poisson", w_init=np.full(3, 5.0), b_init=10.0)
    assert np.all(np.isfinite(w)) and np.isfinite(b)
    assert hist["grad_norm"][-1] < 1e-8


def test_bernoulli_separable_no_nan():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(40, 2))
    y = (X[:, 0] > 0).astype(int)
    X[:, 0] += np.where(y == 1, 0.5, -0.5)
    w, b, hist = glm.fit_glm(X, y, "bernoulli", max_iter=30)
    assert np.all(np.isfinite(w)) and np.isfinite(b)
    assert np.all(np.isfinite(hist["loss"]))


def test_predict_shapes_and_range():
    X, y = make_data("bernoulli")
    w, b, _ = glm.fit_glm(X, y, "bernoulli")
    p = glm.predict(X, w, b, "bernoulli")
    assert p.shape == (X.shape[0],) and np.all((p >= 0) & (p <= 1))
    mu = glm.predict(X, w, b, "poisson")
    assert mu.shape == (X.shape[0],) and np.all(mu > 0)


def test_invalid_inputs():
    X = np.zeros((3, 1))
    with pytest.raises(ValueError):
        glm.fit_glm(X, [0, 1, 2], "bernoulli")
    with pytest.raises(ValueError):
        glm.fit_glm(X, [0, -1, 2], "poisson")
    with pytest.raises(ValueError):
        glm.fit_glm(X, [0.5, 1, 2], "poisson")
    with pytest.raises(ValueError):
        glm.fit_glm(X, [0, 1, 0], "gamma")
    with pytest.raises(ValueError):
        glm.fit_glm(np.zeros((2, 1)), [0, 1, 0], "bernoulli")


def test_custom_family_object():
    """Thêm họ mới bằng lớp con Family: dùng cùng fit_glm, không sửa code lõi."""
    X, y = make_data("gaussian")
    w1, b1, _ = glm.fit_glm(X, y, glm.Gaussian())
    w2, b2, _ = glm.fit_glm(X, y, "gaussian")
    np.testing.assert_allclose(w1, w2)
    assert b1 == b2
