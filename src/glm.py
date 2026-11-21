"""[A3] Khung GLM: IRLS chung cho họ Bernoulli, Poisson, Gauss (Đức).

Mô hình: η_i = x_i·w + b,  μ_i = g⁻¹(η_i),  Var(y_i) ∝ V(μ_i).
Mỗi họ phân phối là một lớp con của `Family`, khai báo:
    inverse_link(η)  μ = g⁻¹(η)
    dmu_deta(η)      dμ/dη
    variance(μ)      V(μ)
    nll(y, η)        −log p(y | η), bỏ hằng số không phụ thuộc η
Thêm họ mới (ví dụ Gamma) chỉ cần viết thêm một lớp con, không sửa `fit_glm`.

Loss(w, b) = (1/n) Σ_i nll(y_i, η_i) + (λ/2)‖w‖²   (b không bị phạt; φ = 1),
cùng quy ước với logistic.py.

IRLS (Fisher scoring) cho θ = (w, b), X̃ = [X, 1]:
    W_i = (dμ/dη)² / V(μ),   a_i = (dμ/dη) / V(μ)
    (X̃ᵀ diag(W) X̃ / n + λR) θ_mới = X̃ᵀ [W ⊙ η + a ⊙ (y − μ)] / n
Vế phải là X̃ᵀ W z / n với z = η + (y − μ)/(dμ/dη) (biến phản hồi làm việc),
viết lại để không chia cho dμ/dη ≈ 0. Với liên kết chính tắc (cả ba họ ở đây),
a ≡ 1 và W = V(μ): IRLS trùng với Newton. Có giảm nửa bước khi loss tăng.
"""

import numpy as np

from core import log_sigmoid, sigmoid

_ETA_MAX = 700.0  # chặn η trước exp của liên kết log để không tràn số


class Family:
    """Lớp cơ sở cho một họ GLM. η, μ, y: mảng (n,). Trả về mảng (n,)."""

    name = "base"

    def inverse_link(self, eta):
        raise NotImplementedError

    def link(self, mu):
        raise NotImplementedError

    def dmu_deta(self, eta):
        raise NotImplementedError

    def variance(self, mu):
        raise NotImplementedError

    def nll(self, y, eta):
        raise NotImplementedError

    def check_y(self, y):
        """Kiểm tra miền giá trị của y; sai thì raise ValueError."""

    def residual(self, y, eta):
        """y − μ, (n,)."""
        return y - self.inverse_link(eta)

    def irls_weights(self, eta):
        """Trả về (W, a), mỗi mảng (n,): W = (dμ/dη)²/V(μ), a = (dμ/dη)/V(μ)."""
        mu, dmu = self.inverse_link(eta), self.dmu_deta(eta)
        v = self.variance(mu)
        return dmu * dmu / v, dmu / v


class Bernoulli(Family):
    """y ∈ {0, 1}, liên kết logit: μ = σ(η), V(μ) = μ(1 − μ)."""

    name = "bernoulli"

    def inverse_link(self, eta):
        return sigmoid(eta)

    def link(self, mu):
        return np.log(mu) - np.log1p(-mu)

    def dmu_deta(self, eta):
        return sigmoid(eta) * sigmoid(-eta)

    def variance(self, mu):
        return mu * (1.0 - mu)

    def nll(self, y, eta):
        return -(y * log_sigmoid(eta) + (1.0 - y) * log_sigmoid(-eta))

    def check_y(self, y):
        if not np.all((y == 0) | (y == 1)):
            raise ValueError("Bernoulli: y phải thuộc {0, 1}")

    def residual(self, y, eta):
        # y − σ(η) = y·σ(−η) − (1 − y)·σ(η): không mất chính xác khi σ(η) ≈ 1
        return y * sigmoid(-eta) - (1.0 - y) * sigmoid(eta)

    def irls_weights(self, eta):
        # chính tắc: W = σ(η)σ(−η), a = 1 (không chia 0/0 khi σ bão hòa)
        return self.dmu_deta(eta), np.ones_like(eta)


class Poisson(Family):
    """y ∈ {0, 1, 2, ...}, liên kết log: μ = e^η, V(μ) = μ."""

    name = "poisson"

    def inverse_link(self, eta):
        return np.exp(np.minimum(eta, _ETA_MAX))

    def link(self, mu):
        return np.log(mu)

    def dmu_deta(self, eta):
        return self.inverse_link(eta)

    def variance(self, mu):
        return mu

    def nll(self, y, eta):
        # μ − y·η  (bỏ log y!)
        return self.inverse_link(eta) - y * eta

    def check_y(self, y):
        if np.any(y < 0) or not np.allclose(y, np.round(y)):
            raise ValueError("Poisson: y phải là số nguyên không âm")

    def irls_weights(self, eta):
        return self.inverse_link(eta), np.ones_like(eta)


class Gaussian(Family):
    """y ∈ ℝ, liên kết đồng nhất: μ = η, V(μ) = 1 (φ = 1)."""

    name = "gaussian"

    def inverse_link(self, eta):
        return eta

    def link(self, mu):
        return mu

    def dmu_deta(self, eta):
        return np.ones_like(eta)

    def variance(self, mu):
        return np.ones_like(mu)

    def nll(self, y, eta):
        return 0.5 * (y - eta) ** 2


FAMILIES = {"bernoulli": Bernoulli, "poisson": Poisson, "gaussian": Gaussian}


def get_family(family):
    """family: tên ('bernoulli' | 'poisson' | 'gaussian') hoặc đối tượng Family."""
    if isinstance(family, Family):
        return family
    try:
        return FAMILIES[str(family).lower()]()
    except KeyError:
        raise ValueError(f"Họ không hỗ trợ: {family!r}; chọn trong {list(FAMILIES)}")


def _check_Xy(X, y, fam):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(y, dtype=float).ravel()
    if X.shape[0] != y.shape[0]:
        raise ValueError("X và y phải cùng số mẫu n")
    fam.check_y(y)
    return X, y


def loss_grad(w, b, X, y, family, lam=0.0):
    """Loss và gradient theo θ = (w, b).

    w: (d,); b: float; X: (n, d); y: (n,); family: tên hoặc Family; lam: λ ≥ 0.
    Trả về loss: float, grad: (d + 1,) — d thành phần đầu theo w, cuối theo b.
    grad = −X̃ᵀ[a ⊙ (y − μ)]/n + λ(w, 0) (Fisher = Newton với liên kết chính tắc).
    """
    fam = get_family(family)
    X, y = _check_Xy(X, y, fam)
    n, d = X.shape
    w = np.asarray(w, dtype=float).ravel()
    eta = X @ w + b
    loss = np.mean(fam.nll(y, eta)) + 0.5 * lam * (w @ w)
    _, a = fam.irls_weights(eta)
    s = a * fam.residual(y, eta)                       # (n,)
    grad = -np.r_[X.T @ s, s.sum()] / n + lam * np.r_[w, 0.0]
    return float(loss), grad


def _solve(A, rhs):
    """Giải A·θ = rhs; A suy biến hoặc kết quả không hữu hạn thì dùng lstsq."""
    try:
        theta = np.linalg.solve(A, rhs)
        if np.all(np.isfinite(theta)):
            return theta
    except np.linalg.LinAlgError:
        pass
    theta = np.linalg.lstsq(A, rhs, rcond=None)[0]
    return np.where(np.isfinite(theta), theta, 0.0)


def fit_glm(X, y, family="bernoulli", lam=0.0, w_init=None, b_init=None,
            tol=1e-8, max_iter=100, max_halving=30):
    """IRLS cho GLM họ `family`, có L2 tùy chọn.

    X: (n, d) dense; y: (n,); family: 'bernoulli' | 'poisson' | 'gaussian' hoặc Family;
    lam: λ ≥ 0 (không phạt b).
    w_init: (d,) hoặc None (= 0); b_init: float hoặc None (= g(ȳ), chặn vào miền hợp lệ).
    Trả về (w (d,), b float, history).
    history: dict các list "w", "b", "loss", "grad_norm", ghi ở MỌI vòng kể cả vòng 0.
    Dừng khi ‖grad‖ < tol hoặc đủ max_iter bước. Mỗi bước, nếu loss tăng thì giảm
    nửa bước tối đa `max_halving` lần.
    """
    fam = get_family(family)
    X, y = _check_Xy(X, y, fam)
    n, d = X.shape
    w = np.zeros(d) if w_init is None else np.asarray(w_init, dtype=float).ravel().copy()
    if b_init is None:
        ybar = float(np.mean(y))
        if fam.name == "bernoulli":
            ybar = float(np.clip(ybar, 1e-3, 1 - 1e-3))
        elif fam.name == "poisson":
            ybar = max(ybar, 1e-3)
        b = float(fam.link(ybar))
    else:
        b = float(b_init)

    Xa = np.hstack([X, np.ones((n, 1))])               # (n, d+1)
    R = np.diag(np.r_[np.full(d, lam), 0.0])           # (d+1, d+1)
    history = {"w": [], "b": [], "loss": [], "grad_norm": []}

    def record(w, b):
        loss, g = loss_grad(w, b, X, y, fam, lam)
        gn = float(np.linalg.norm(g))
        history["w"].append(w.copy())
        history["b"].append(b)
        history["loss"].append(loss)
        history["grad_norm"].append(gn)
        return loss, gn

    loss, gn = record(w, b)
    theta = np.r_[w, b]
    for _ in range(max_iter):
        if gn < tol:
            break
        eta = Xa @ theta
        W, a = fam.irls_weights(eta)
        rhs = Xa.T @ (W * eta + a * fam.residual(y, eta)) / n
        A = (Xa * W[:, None]).T @ Xa / n + R
        theta_new = _solve(A, rhs)

        # giảm nửa bước nếu loss tăng (an toàn cho Poisson khi khởi tạo xa)
        step = theta_new - theta
        for _ in range(max_halving):
            cand = theta + step
            new_loss, _ = loss_grad(cand[:d], cand[d], X, y, fam, lam)
            if np.isfinite(new_loss) and new_loss <= loss + 1e-12:
                break
            step = step / 2.0
        theta = theta + step
        w, b = theta[:d].copy(), float(theta[d])
        loss, gn = record(w, b)
    return w, b, history


def predict(X, w, b, family="bernoulli"):
    """Kỳ vọng μ = g⁻¹(Xw + b). X: (n, d); w: (d,); b: float. Trả về (n,)."""
    fam = get_family(family)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    return fam.inverse_link(X @ np.asarray(w, dtype=float).ravel() + b)
