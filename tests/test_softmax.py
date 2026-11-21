import os
import sys
import time
import tracemalloc

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import core  # noqa: E402
import softmax as sm  # noqa: E402

K = 3


def make_data(n=300, d=5, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    W = rng.normal(size=(d, K))
    P = core.softmax(X @ W + rng.normal(size=K))
    y = np.array([rng.choice(K, p=p) for p in P])
    return X, y


# ------------------------------------------------------------ gradient
@pytest.mark.parametrize("lam", [0.0, 0.1])
@pytest.mark.parametrize("weighted", [False, True])
def test_grad_finite_difference(lam, weighted):
    X, y = make_data(n=40, d=4)
    rng = np.random.default_rng(42)
    W, b = rng.normal(size=(4, K)), rng.normal(size=K)
    sw = rng.uniform(0.2, 3.0, size=40) if weighted else None
    _, dW, db = sm.loss_grad(W, b, X, y, lam, sample_weight=sw)
    eps = 1e-6
    dW_fd = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(K):
            E = np.zeros_like(W)
            E[i, j] = eps
            dW_fd[i, j] = (sm.loss_grad(W + E, b, X, y, lam, sw)[0]
                           - sm.loss_grad(W - E, b, X, y, lam, sw)[0]) / (2 * eps)
    db_fd = np.zeros(K)
    for j in range(K):
        e = np.zeros(K)
        e[j] = eps
        db_fd[j] = (sm.loss_grad(W, b + e, X, y, lam, sw)[0]
                    - sm.loss_grad(W, b - e, X, y, lam, sw)[0]) / (2 * eps)
    np.testing.assert_allclose(dW, dW_fd, atol=1e-8)
    np.testing.assert_allclose(db, db_fd, atol=1e-8)


def test_weighted_loss_is_weighted_mean():
    X, y = make_data(n=30, d=3)
    rng = np.random.default_rng(42)
    W, b = rng.normal(size=(3, K)), rng.normal(size=K)
    sw = rng.uniform(0.5, 2.0, size=30)
    Z = X @ W + b
    li = core.logsumexp(Z) - Z[np.arange(30), y]
    expected = np.sum(sw * li) / sw.sum() + 0.5 * 0.2 * np.sum(W ** 2)
    np.testing.assert_allclose(sm.loss_grad(W, b, X, y, 0.2, sw)[0], expected, rtol=1e-12)


# ------------------------------------------------------------ bộ giải
@pytest.mark.parametrize("lam", [1e-3, 1e-1])
def test_gd_matches_lbfgs(lam):
    X, y = make_data()
    _, _, h = sm.fit_gd_armijo(X, y, K, lam=lam, tol=1e-7, max_iter=20000)
    _, _, res = sm.fit_lbfgs(X, y, K, lam=lam)
    assert abs(h["loss"][-1] - res.fun) < 1e-5


def test_armijo_condition_and_monotone():
    X, y = make_data()
    c = 1e-4
    _, _, h = sm.fit_gd_armijo(X, y, K, lam=1e-2, c=c)
    loss, gn, step = map(np.asarray, (h["loss"], h["grad_norm"], h["step"]))
    assert len(step) == len(loss) - 1 == len(h["n_backtrack"])
    assert np.all(loss[1:] <= loss[:-1] - c * step * gn[:-1] ** 2)
    assert np.all(np.diff(loss) <= 0)


def test_step_rule_first_t0_then_double_previous():
    X, y = make_data()
    t0, beta = 1.0, 0.5
    _, _, h = sm.fit_gd_armijo(X, y, K, lam=1e-2, t0=t0, beta=beta, max_iter=50)
    step, nb = np.asarray(h["step"]), np.asarray(h["n_backtrack"])
    np.testing.assert_allclose(step[0], t0 * beta ** nb[0])
    np.testing.assert_allclose(step[1:], 2 * step[:-1] * beta ** nb[1:])


def test_sparse_equals_dense():
    X, y = make_data()
    X[np.abs(X) < 0.8] = 0.0
    Xs = sp.csr_matrix(X)
    Wd, bd, hd = sm.fit_gd_armijo(X, y, K, lam=1e-2)
    Ws, bs, hs = sm.fit_gd_armijo(Xs, y, K, lam=1e-2)
    assert len(hd["loss"]) == len(hs["loss"])
    np.testing.assert_allclose(Ws, Wd, rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(bs, bd, rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(sm.predict_proba(Xs, Ws, bs), sm.predict_proba(X, Wd, bd),
                               atol=1e-10)


@pytest.mark.parametrize("lam", [1e-3, 1e-1])
def test_matches_sklearn_multinomial(lam):
    X, y = make_data()
    n = X.shape[0]
    model = sm.SoftmaxRegression(lam=lam, tol=1e-8, max_iter=20000).fit(X, y)
    sk = LogisticRegression(C=1.0 / (n * lam), solver="lbfgs", tol=1e-12,
                            max_iter=10000).fit(X, y)
    np.testing.assert_allclose(model.predict_proba(X), sk.predict_proba(X), atol=1e-5)


def test_sample_weight_constant_equals_none():
    X, y = make_data()
    W1, b1, h1 = sm.fit_gd_armijo(X, y, K, lam=1e-2)
    W2, b2, h2 = sm.fit_gd_armijo(X, y, K, lam=1e-2, sample_weight=np.full(len(y), 2.0))
    np.testing.assert_allclose(W2, W1, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(b2, b1, rtol=1e-10, atol=1e-12)


def test_init_bias_is_log_frequency_plus_one():
    X, y = make_data()
    _, _, h = sm.fit_gd_armijo(X, y, K, max_iter=0)
    W, b, _ = sm.fit_gd_armijo(X, y, K, max_iter=0)
    counts = np.bincount(y, minlength=K) + 1.0
    np.testing.assert_allclose(b, np.log(counts / counts.sum()))
    assert np.all(W == 0) and len(h["loss"]) == 1


# ------------------------------------------------------------ lớp SoftmaxRegression
def test_missing_class():
    X, y = make_data()
    keep = y != 1
    X, y = X[keep], y[keep]
    model = sm.SoftmaxRegression(lam=1e-2, n_classes=3, max_iter=300).fit(X, y)
    P = model.predict_proba(X)
    assert P.shape == (len(y), 3)
    assert model.W_.shape == (X.shape[1], 3) and model.b_.shape == (3,)
    assert np.all(np.isfinite(P))
    np.testing.assert_allclose(P.sum(axis=1), 1.0)


def test_softmax_of_decision_function_equals_proba():
    X, y = make_data()
    model = sm.SoftmaxRegression(lam=1e-2).fit(X, y)
    Z = model.decision_function(X)
    assert Z.shape == (len(y), K)
    np.testing.assert_allclose(core.softmax(Z), model.predict_proba(X), rtol=1e-14)
    np.testing.assert_array_equal(model.predict(X), np.argmax(model.predict_proba(X), 1))


def test_save_load(tmp_path):
    X, y = make_data()
    model = sm.SoftmaxRegression(lam=1e-2).fit(X, y)
    path = tmp_path / "softmax.npz"
    model.save(path)
    loaded = sm.SoftmaxRegression.load(path)
    assert loaded.lam == model.lam and loaded.n_classes == model.n_classes
    np.testing.assert_array_equal(loaded.predict_proba(X), model.predict_proba(X))
    np.testing.assert_array_equal(loaded.W_, model.W_)


def test_warm_start_fewer_iterations():
    X, y = make_data()
    warm = sm.SoftmaxRegression(lam=1e-1, warm_start=True, max_iter=20000).fit(X, y)
    warm.lam = 5e-2
    warm.fit(X, y)
    cold = sm.SoftmaxRegression(lam=5e-2, max_iter=20000).fit(X, y)
    assert cold.history_["grad_norm"][-1] < cold.tol
    assert warm.history_["grad_norm"][-1] < warm.tol
    assert len(warm.history_["step"]) < len(cold.history_["step"])
    np.testing.assert_allclose(warm.predict_proba(X), cold.predict_proba(X), atol=1e-5)


def test_lbfgs_solver_in_class():
    X, y = make_data()
    a = sm.SoftmaxRegression(lam=1e-2, solver="lbfgs", tol=1e-10).fit(X, y)
    b = sm.SoftmaxRegression(lam=1e-2, tol=1e-8, max_iter=20000).fit(X, y)
    np.testing.assert_allclose(a.predict_proba(X), b.predict_proba(X), atol=1e-6)


def test_huge_logits_no_nan():
    X, y = make_data()
    Xb = X * 1e3
    model = sm.SoftmaxRegression(lam=1e-4, max_iter=200).fit(Xb, y)
    assert np.all(np.isfinite(model.history_["loss"]))
    assert np.all(np.isfinite(model.W_)) and np.all(np.isfinite(model.b_))
    P = model.predict_proba(Xb)
    assert not np.any(np.isnan(P))
    np.testing.assert_allclose(P.sum(axis=1), 1.0)
    # logit cỡ 1e5 từ mô hình bất kỳ
    rng = np.random.default_rng(42)
    P2 = sm.predict_proba(Xb * 10, rng.normal(size=(X.shape[1], K)), np.zeros(K))
    assert not np.any(np.isnan(P2))
    f, dW, db = sm.loss_grad(rng.normal(size=(X.shape[1], K)), np.zeros(K), Xb * 10, y, 0.0)
    assert np.isfinite(f) and np.all(np.isfinite(dW)) and np.all(np.isfinite(db))


def test_bad_labels_raise():
    X, y = make_data(n=20)
    with pytest.raises(ValueError):
        sm.SoftmaxRegression(n_classes=2).fit(X, y)


# ------------------------------------------------------------ hiệu năng
def test_performance_sparse_no_dense(monkeypatch):
    n, d = 2000, 100_000
    X = sp.random(n, d, density=0.001, format="csr", random_state=42)
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    X = sp.csr_matrix(sp.diags(1.0 / np.maximum(norms, 1e-12)) @ X)   # chuẩn hóa L2 như TF-IDF
    y = np.random.default_rng(42).integers(0, K, size=n)

    def forbid(*args, **kwargs):
        raise AssertionError("không được đổi X sparse sang dense")

    for cls in {type(X), sp.csr_matrix, sp.csc_matrix}:
        monkeypatch.setattr(cls, "toarray", forbid)
        monkeypatch.setattr(cls, "todense", forbid)

    t = time.perf_counter()
    model = sm.SoftmaxRegression(lam=1e-3).fit(X, y)
    elapsed = time.perf_counter() - t
    assert elapsed < 60
    assert model.W_.shape == (d, K)

    # đo bộ nhớ đỉnh riêng (tracemalloc làm chậm nhiều nên chỉ chạy ít vòng)
    tracemalloc.start()
    sm.SoftmaxRegression(lam=1e-3, max_iter=30).fit(X, y)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 0.05 * n * d * 8          # mảng dense (n, d) cần 1,6 GB
    # dừng do ‖g‖ < tol hoặc do loss gần như không giảm; loss phải sát nghiệm L-BFGS
    _, _, res = sm.fit_lbfgs(X, y, K, lam=1e-3)
    assert abs(model.history_["loss"][-1] - res.fun) < 1e-6
    assert len(model.history_["step"]) < model.max_iter


# ------------------------------------------------------------ theo dõi hội tụ, fit_path
def test_converged_flags_small_data():
    X, y = make_data()
    model = sm.SoftmaxRegression(lam=1e-2).fit(X, y)
    assert model.converged_ is True
    assert model.history_["stop_reason"] in ("grad_tol", "rel_loss")
    assert model.n_iter_ == len(model.history_["step"])


def test_not_converged_max_iter_warns():
    X, y = make_data()
    with pytest.warns(RuntimeWarning, match="max_iter"):
        model = sm.SoftmaxRegression(lam=1e-2, max_iter=5).fit(X, y)
    assert model.converged_ is False
    assert model.history_["stop_reason"] == "max_iter"
    assert model.n_iter_ == 5


def test_stop_reason_in_fit_gd_armijo():
    X, y = make_data()
    _, _, h = sm.fit_gd_armijo(X, y, K, lam=1e-2, tol=1e3)
    assert h["stop_reason"] == "grad_tol" and len(h["step"]) == 0
    _, _, h = sm.fit_gd_armijo(X, y, K, lam=1e-2, max_iter=3)
    assert h["stop_reason"] == "max_iter"


def test_lbfgs_sets_n_iter_and_converged():
    X, y = make_data()
    model = sm.SoftmaxRegression(lam=1e-2, solver="lbfgs", tol=1e-8).fit(X, y)
    assert isinstance(model.n_iter_, int) and model.n_iter_ > 0
    assert model.converged_ is True
    assert model.n_iter_ == model.history_.nit


LAMS_PATH = [1e-1, 1e-2, 1e-3]


def test_fit_path_matches_separate_fits():
    X, y = make_data()
    path = sm.fit_path(X, y, LAMS_PATH, tol=1e-8, max_iter=20000)
    for lam, model in zip(LAMS_PATH, path):
        single = sm.SoftmaxRegression(lam=lam, tol=1e-8, max_iter=20000).fit(X, y)
        assert model.converged_ and single.converged_
        f_path = sm.loss_grad(model.W_, model.b_, X, y, lam)[0]
        f_single = sm.loss_grad(single.W_, single.b_, X, y, lam)[0]
        assert abs(f_path - f_single) < 1e-6


def test_fit_path_fewer_total_iterations():
    X, y = make_data()
    path = sm.fit_path(X, y, LAMS_PATH, max_iter=20000)
    singles = [sm.SoftmaxRegression(lam=lam, max_iter=20000).fit(X, y) for lam in LAMS_PATH]
    assert sum(m.n_iter_ for m in path) < sum(m.n_iter_ for m in singles)


def test_fit_path_keeps_input_order():
    X, y = make_data()
    lams = [1e-3, 1e-1, 1e-2]
    path = sm.fit_path(X, y, lams, max_iter=20000)
    assert len(path) == 3
    for lam, model in zip(lams, path):
        assert model.lam == lam
        assert model.W_.shape == (X.shape[1], K)
        single = sm.SoftmaxRegression(lam=lam, max_iter=20000).fit(X, y)
        np.testing.assert_allclose(model.predict_proba(X), single.predict_proba(X), atol=1e-4)
