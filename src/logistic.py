"""[A2] Hồi quy logistic nhị phân bằng Newton/IRLS — dùng cho C1, C3 (Kiên).

KHÔNG dùng ở phần D (phần D dùng softmax.py).
Nhãn y ∈ {0, 1}. Chỉ hỗ trợ X dense.

Loss(w, b) = (1/n) Σ_i [-y_i log σ(z_i) - (1 - y_i) log σ(-z_i)] + (λ/2)‖w‖²,
z_i = x_i·w + b. Hệ số chặn b không bị phạt.
"""

import numpy as np

from core import log_sigmoid, sigmoid


def _check_Xy(X, y):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(y, dtype=float).ravel()
    if X.shape[0] != y.shape[0]:
        raise ValueError("X và y phải cùng số mẫu n")
    return X, y


def loss_grad_hess(w, b, X, y, lam):
    """Loss, gradient, Hessian theo vector tham số gộp θ = (w, b).

    w: (d,); b: float; X: (n, d); y: (n,) ∈ {0, 1}; lam: λ ≥ 0.
    Trả về:
        loss: float
        grad: (d + 1,) — d thành phần đầu theo w, thành phần cuối theo b
        hess: (d + 1, d + 1)
    Loss tính bằng log_sigmoid (không tính log(p)).
    """
    X, y = _check_Xy(X, y)
    n, d = X.shape
    w = np.asarray(w, dtype=float).ravel()
    z = X @ w + b
    loss = -np.mean(y * log_sigmoid(z) + (1.0 - y) * log_sigmoid(-z)) + 0.5 * lam * (w @ w)

    # p - y viết lại thành (1-y)σ(z) - yσ(-z), 1 - p = σ(-z): tránh mất chính xác
    # khi σ(z) làm tròn thành 1 (quan trọng cho dữ liệu khả tách, C3).
    p, q = sigmoid(z), sigmoid(-z)
    r = (1.0 - y) * p - y * q                       # (n,) = p - y
    Xa = np.hstack([X, np.ones((n, 1))])            # (n, d+1)
    reg = np.r_[np.full(d, lam), 0.0]               # b không bị phạt
    grad = Xa.T @ r / n + reg * np.r_[w, 0.0]
    s = p * q                                       # (n,) = p(1 - p)
    hess = (Xa * s[:, None]).T @ Xa / n + np.diag(reg)
    return float(loss), grad, hess


def _newton_step(H, g):
    """Giải H·Δ = -g. H suy biến hoặc kết quả không hữu hạn thì dùng lstsq."""
    try:
        delta = np.linalg.solve(H, -g)
        if np.all(np.isfinite(delta)):
            return delta
    except np.linalg.LinAlgError:
        pass
    delta = np.linalg.lstsq(H, -g, rcond=None)[0]
    return np.where(np.isfinite(delta), delta, 0.0)


def fit_irls(X, y, lam=0.0, w_init=None, b_init=0.0, tol=1e-8, max_iter=100):
    """Newton/IRLS cho hồi quy logistic nhị phân.

    X: (n, d) dense; y: (n,) ∈ {0, 1}; lam: λ.
    Trả về (w (d,), b float, history).
    history: dict các list "w", "b", "loss", "norm_w", "grad_norm",
    ghi lại ở MỌI vòng kể cả vòng 0 (khởi tạo), nên có tối đa max_iter + 1 phần tử.
    Dừng khi ‖grad‖ < tol hoặc đã đủ max_iter bước.
    H suy biến (dữ liệu khả tách, λ = 0): dùng lstsq, không crash, không sinh nan.
    """
    X, y = _check_Xy(X, y)
    d = X.shape[1]
    w = np.zeros(d) if w_init is None else np.asarray(w_init, dtype=float).ravel().copy()
    b = float(b_init)
    history = {"w": [], "b": [], "loss": [], "norm_w": [], "grad_norm": []}

    def record(w, b):
        loss, g, H = loss_grad_hess(w, b, X, y, lam)
        gn = float(np.linalg.norm(g))
        history["w"].append(w.copy())
        history["b"].append(b)
        history["loss"].append(loss)
        history["norm_w"].append(float(np.linalg.norm(w)))
        history["grad_norm"].append(gn)
        return g, H, gn

    g, H, gn = record(w, b)
    for _ in range(max_iter):
        if gn < tol:
            break
        delta = _newton_step(H, g)
        w = w + delta[:d]
        b = float(b + delta[d])
        g, H, gn = record(w, b)
    return w, b, history


def predict_proba(X, w, b):
    """P(y = 1 | x). X: (n, d); w: (d,); b: float. Trả về (n,)."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    return sigmoid(X @ np.asarray(w, dtype=float).ravel() + b)
