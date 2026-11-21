"""[A2] Hồi quy softmax có/không L2: hạ gradient + Armijo, đối chiếu L-BFGS (Kiên).

Đây là mô hình hồi quy logistic đa lớp THẬT của D4/D5.

Loss(W, b) = Σ_i s_i ℓ_i / Σ_i s_i + (λ/2)‖W‖²,
ℓ_i = logsumexp(z_i) - z_{i, y_i},  z_i = x_i W + b.
Hệ số chặn b không bị phạt. s_i = 1 nếu sample_weight=None.

X có thể là dense (n, d) hoặc scipy.sparse CSR; không bao giờ đổi X sang dense.
Nhãn y: mảng số nguyên (n,) trong {0, ..., K-1} theo core.LABELS; one-hot chỉ tạo bên trong.
"""

import numpy as np

from core import logsumexp, softmax


# ---------------------------------------------------------------- tiện ích nội bộ
def _check_y(y, n, K):
    y = np.asarray(y).ravel()
    if y.shape[0] != n:
        raise ValueError("X và y phải cùng số mẫu n")
    if not np.issubdtype(y.dtype, np.integer):
        if not np.all(np.equal(np.mod(y, 1), 0)):
            raise ValueError("y phải là nhãn số nguyên")
        y = y.astype(int)
    if y.size and (y.min() < 0 or y.max() >= K):
        raise ValueError(f"nhãn y phải nằm trong 0..{K - 1}")
    return y


def _norm_weights(sample_weight, n):
    """Trả về trọng số chuẩn hóa (n,), tổng bằng 1."""
    if sample_weight is None:
        return np.full(n, 1.0 / n)
    s = np.asarray(sample_weight, dtype=float).ravel()
    if s.shape[0] != n or np.any(s < 0) or s.sum() <= 0:
        raise ValueError("sample_weight phải có shape (n,), không âm, tổng > 0")
    return s / s.sum()


def _one_hot(y, K):
    Y = np.zeros((y.shape[0], K))
    Y[np.arange(y.shape[0]), y] = 1.0
    return Y


def _loss_grad(W, b, X, y, Y, s, lam):
    """Lõi tính toán: s đã chuẩn hóa (tổng 1), Y one-hot (n, K)."""
    Z = np.asarray(X @ W) + b                      # (n, K), X sparse vẫn cho ndarray
    lse = logsumexp(Z, axis=1)                     # (n,)
    loss = s @ (lse - Z[np.arange(Z.shape[0]), y]) + 0.5 * lam * np.sum(W * W)
    G = (softmax(Z, axis=1) - Y) * s[:, None]      # (n, K)
    dW = np.asarray(X.T @ G) + lam * W             # (d, K)
    db = G.sum(axis=0)                             # (K,)
    return float(loss), dW, db


# ---------------------------------------------------------------- hàm mức thấp
def loss_grad(W, b, X, y, lam, sample_weight=None):
    """Loss và gradient của hồi quy softmax.

    W: (d, K); b: (K,); X: (n, d) dense hoặc CSR; y: (n,) int; lam: λ;
    sample_weight: (n,) hoặc None.
    Trả về (loss float, dW (d, K), db (K,)).
    Chỉ dùng X @ W và X.T @ (...), không tạo ma trận dense (n, d) hay (d, d).
    """
    W = np.asarray(W, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    n, K = X.shape[0], W.shape[1]
    y = _check_y(y, n, K)
    s = _norm_weights(sample_weight, n)
    return _loss_grad(W, b, X, y, _one_hot(y, K), s, lam)


def _init_params(y, K, d, s):
    """W = 0, b = log tần suất lớp (có trọng số), cộng 1 vào mỗi lớp."""
    n = y.shape[0]
    counts = np.bincount(y, weights=s * n, minlength=K) + 1.0
    return np.zeros((d, K)), np.log(counts / counts.sum())


def fit_gd_armijo(X, y, K, lam=0.0, sample_weight=None, W_init=None, b_init=None,
                  t0=1.0, c=1e-4, beta=0.5, tol=1e-6, max_iter=2000):
    """Hạ gradient với tìm bước backtracking Armijo.

    X: (n, d) dense hoặc CSR; y: (n,) int; K: số lớp (không suy ra từ y).
    Trả về (W (d, K), b (K,), history).
    history: "loss", "grad_norm" (ghi ở mọi vòng kể cả vòng 0, dài m + 1),
             "step", "n_backtrack" (mỗi bước đã chấp nhận, dài m).
    Bước thử: vòng đầu t = t0, các vòng sau t = 2·(bước chấp nhận trước),
    nhân beta tới khi f(θ - t·g) ≤ f(θ) - c·t·‖g‖².
    Dừng khi ‖g‖ < tol, hoặc loss giảm tương đối < 1e-10 trong 10 vòng liền,
    hoặc không tìm được bước thỏa Armijo sau 100 lần lùi (giới hạn số học).
    """
    n, d = X.shape
    y = _check_y(y, n, K)
    s = _norm_weights(sample_weight, n)
    Y = _one_hot(y, K)
    W0, b0 = _init_params(y, K, d, s)
    W = W0 if W_init is None else np.array(W_init, dtype=float)
    b = b0 if b_init is None else np.array(b_init, dtype=float).ravel()

    # Xᵀ lưu sẵn (CSR nếu X thưa) để tính X.T @ G nhanh; không tạo mảng dense (n, d).
    XT = X.T.tocsr() if hasattr(X, "tocsr") else X.T
    rows = np.arange(n)

    def data_loss(Z):
        return s @ (logsumexp(Z, axis=1) - Z[rows, y])

    def grad(Z, W):
        G = (softmax(Z, axis=1) - Y) * s[:, None]     # (n, K)
        return np.asarray(XT @ G) + lam * W, G.sum(axis=0)

    Z = np.asarray(X @ W) + b                         # (n, K)
    wsq = np.vdot(W, W)
    f = data_loss(Z) + 0.5 * lam * wsq
    dW, db = grad(Z, W)
    gsq = np.vdot(dW, dW) + db @ db
    history = {"loss": [float(f)], "grad_norm": [float(np.sqrt(gsq))],
               "step": [], "n_backtrack": []}

    t_prev = None
    small = 0
    max_backtrack = 100
    for _ in range(max_iter):
        if np.sqrt(gsq) < tol:
            break
        # Trên đường θ - t·g: Z(t) = Z - t·(X dW + db), ‖W(t)‖² khai triển theo t,
        # nên mỗi lần thử bước chỉ tốn O(nK).
        dZ = np.asarray(X @ dW) + db
        w_dw, dw_sq = np.vdot(W, dW), np.vdot(dW, dW)
        t = t0 if t_prev is None else 2.0 * t_prev
        accepted = False
        for k in range(max_backtrack + 1):
            Z_new = Z - t * dZ
            wsq_new = wsq - 2.0 * t * w_dw + t * t * dw_sq
            f_new = data_loss(Z_new) + 0.5 * lam * wsq_new
            if f_new <= f - c * t * gsq:
                accepted = True
                break
            t *= beta
        if not accepted:
            break
        rel = (f - f_new) / max(abs(f), 1e-300)
        W = W - t * dW
        b = b - t * db
        Z, f, wsq, t_prev = Z_new, f_new, wsq_new, t
        dW, db = grad(Z, W)
        gsq = np.vdot(dW, dW) + db @ db
        history["loss"].append(float(f))
        history["grad_norm"].append(float(np.sqrt(gsq)))
        history["step"].append(t)
        history["n_backtrack"].append(k)
        small = small + 1 if rel < 1e-10 else 0
        if small >= 10:
            break
    return W, b, history


def fit_lbfgs(X, y, K, lam=0.0, sample_weight=None, tol=1e-8, max_iter=1000):
    """L-BFGS-B của scipy trên vector phẳng θ = (W.ravel(), b). CHỈ để đối chiếu.

    Trả về (W (d, K), b (K,), scipy_result).
    """
    from scipy.optimize import minimize

    n, d = X.shape
    y = _check_y(y, n, K)
    s = _norm_weights(sample_weight, n)
    Y = _one_hot(y, K)
    W0, b0 = _init_params(y, K, d, s)

    def fun(theta):
        W, b = theta[: d * K].reshape(d, K), theta[d * K:]
        f, dW, db = _loss_grad(W, b, X, y, Y, s, lam)
        return f, np.r_[dW.ravel(), db]

    res = minimize(fun, np.r_[W0.ravel(), b0], method="L-BFGS-B", jac=True,
                   options={"maxiter": max_iter, "gtol": tol, "ftol": 1e-15})
    return res.x[: d * K].reshape(d, K), res.x[d * K:], res


def logits(X, W, b):
    """Logit Z = X W + b. X: (n, d) dense hoặc CSR; W: (d, K); b: (K,). Trả về (n, K)."""
    return np.asarray(X @ W) + np.asarray(b).ravel()


def predict_proba(X, W, b):
    """Xác suất softmax (n, K), mỗi hàng tổng bằng 1."""
    return softmax(logits(X, W, b), axis=1)


# ---------------------------------------------------------------- lớp cho D4/D5
class SoftmaxRegression:
    """Hồi quy softmax đa lớp, API giống sklearn (fit / predict_proba / predict).

    Thuộc tính sau khi fit: W_ (d, K), b_ (K,), history_ (dict hoặc scipy result).
    """

    def __init__(self, lam=1e-4, n_classes=3, solver="gd_armijo", tol=1e-6,
                 max_iter=2000, warm_start=False):
        self.lam = lam
        self.n_classes = n_classes
        self.solver = solver
        self.tol = tol
        self.max_iter = max_iter
        self.warm_start = warm_start

    def fit(self, X, y, sample_weight=None):
        """X: (n, d) dense hoặc CSR; y: (n,) int; sample_weight: (n,) hoặc None. Trả về self."""
        K = int(self.n_classes)
        d = X.shape[1]
        if self.solver == "gd_armijo":
            W_init = b_init = None
            if self.warm_start and getattr(self, "W_", None) is not None \
                    and self.W_.shape == (d, K):
                W_init, b_init = self.W_, self.b_
            W, b, hist = fit_gd_armijo(X, y, K, lam=self.lam, sample_weight=sample_weight,
                                       W_init=W_init, b_init=b_init,
                                       tol=self.tol, max_iter=self.max_iter)
        elif self.solver == "lbfgs":
            W, b, hist = fit_lbfgs(X, y, K, lam=self.lam, sample_weight=sample_weight,
                                   tol=self.tol, max_iter=self.max_iter)
        else:
            raise ValueError("solver phải là 'gd_armijo' hoặc 'lbfgs'")
        self.W_, self.b_, self.history_ = W, b, hist
        return self

    def decision_function(self, X):
        """Logit (n, K) — đầu vào cho temperature scaling (A4)."""
        return logits(X, self.W_, self.b_)

    def predict_proba(self, X):
        """Xác suất (n, K), mỗi hàng tổng bằng 1."""
        return softmax(self.decision_function(X), axis=1)

    def predict(self, X):
        """Nhãn số nguyên (n,) theo core.LABELS."""
        return np.argmax(self.decision_function(X), axis=1)

    def save(self, path):
        """Lưu W, b, lam, n_classes vào file .npz."""
        np.savez(path, W=self.W_, b=self.b_, lam=self.lam, n_classes=self.n_classes)

    @classmethod
    def load(cls, path):
        """Nạp mô hình từ .npz do save() tạo."""
        with np.load(path) as f:
            model = cls(lam=float(f["lam"]), n_classes=int(f["n_classes"]))
            model.W_ = f["W"].copy()
            model.b_ = f["b"].copy()
        model.history_ = None
        return model
