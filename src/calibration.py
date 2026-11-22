"""[A4] Hiệu chuẩn hậu nghiệm: Platt scaling, hồi quy đẳng trương (PAVA), temperature scaling (Nhật).

Tự cài bằng NumPy (không dùng sklearn.calibration / sklearn.isotonic / scipy.optimize).
Mọi exp trên logit đi qua core.sigmoid / core.log_sigmoid / core.logsumexp / core.softmax.

Quy ước (RULES.md):
- Bộ hiệu chuẩn nhận "điểm" scores (n, K) và trả xác suất (n, K), mỗi hàng tổng = 1:
    softmax → SoftmaxRegression.decision_function (logit)
    Naive Bayes → predict_log_proba
    SVM tuyến tính → decision_function
- K lấy từ tham số n_classes, không suy ra từ y (val_a có thể thiếu lớp).
- Platt và PAVA đa lớp: one-vs-rest theo từng cột, kẹp vào [1e-6, 1 − 1e-6], rồi chuẩn hóa hàng.
- Chỉ fit trên val_a; chọn bộ hiệu chuẩn trên val_b.
"""

import numpy as np

from core import log_sigmoid, logsumexp, sigmoid, softmax

CLIP = 1e-6               # kẹp xác suất từng cột (Platt, Isotonic) trước khi chuẩn hóa hàng
BETA_MIN, BETA_MAX = 1e-4, 1e4   # β = 1/T giữ trong khoảng này (T ∈ [1e-4, 1e4])


# ---------------------------------------------------------------------------
# Kiểm tra đầu vào
# ---------------------------------------------------------------------------
def _check_weights(sample_weight, n):
    """sample_weight: None hoặc (n,) ≥ 0, tổng > 0. Trả về (n,) float."""
    if sample_weight is None:
        return np.ones(n)
    w = np.asarray(sample_weight, dtype=float)
    if w.shape != (n,):
        raise ValueError(f"sample_weight phải có shape ({n},), nhận {w.shape}")
    if np.any(~np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("sample_weight phải hữu hạn, ≥ 0 và có tổng > 0")
    return w


def _check_scores_labels(scores, y, n_classes):
    """scores: (n, K) hữu hạn, K == n_classes; y: (n,) số nguyên trong [0, K) hoặc None."""
    S = np.asarray(scores, dtype=float)
    if S.ndim != 2 or S.shape[1] != n_classes:
        raise ValueError(f"scores phải có shape (n, {n_classes}), nhận {S.shape}")
    if not np.all(np.isfinite(S)):
        raise ValueError("scores chứa nan/inf")
    if y is None:
        return S, None
    y = np.asarray(y)
    if y.shape != (S.shape[0],):
        raise ValueError(f"y phải có shape ({S.shape[0]},), nhận {y.shape}")
    if not np.issubdtype(y.dtype, np.integer):
        if not np.all(np.mod(y, 1) == 0):
            raise ValueError("y phải là mảng số nguyên")
        y = y.astype(int)
    if y.size and (y.min() < 0 or y.max() >= n_classes):
        raise ValueError(f"y phải nằm trong [0, {n_classes})")
    return S, y


def _clip_normalize(P):
    """P: (n, K) xác suất từng cột (OvR) → kẹp [CLIP, 1 − CLIP] rồi chuẩn hóa hàng. Trả về (n, K)."""
    P = np.clip(P, CLIP, 1.0 - CLIP)
    return P / P.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# PAVA / hồi quy đẳng trương
# ---------------------------------------------------------------------------
def pava(y, sample_weight=None):
    """Hồi quy đẳng trương bằng thuật toán PAVA (ngăn xếp các khối), O(n).

    Nghiệm của min Σ wᵢ(ŷᵢ − yᵢ)² với ŷ₁ ≤ ŷ₂ ≤ … ≤ ŷₙ (y đã xếp theo điểm).
    y: (n,) float; sample_weight: (n,) ≥ 0 hoặc None. Trả về ŷ: (n,).
    Khối có tổng trọng số 0 lấy trung bình không trọng số.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError(f"y phải là mảng 1 chiều, nhận shape {y.shape}")
    n = y.size
    if n == 0:
        return y.copy()
    w = _check_weights(sample_weight, n) if sample_weight is not None else np.ones(n)

    means, weights, sizes = [], [], []
    for yi, wi in zip(y.tolist(), w.tolist()):
        m, wt, c = yi, wi, 1
        # gộp ngược khi khối trước lớn hơn khối hiện tại (vi phạm tính không giảm)
        while means and means[-1] > m:
            pm, pw, pc = means.pop(), weights.pop(), sizes.pop()
            tw = pw + wt
            m = (pm * pw + m * wt) / tw if tw > 0 else (pm * pc + m * c) / (pc + c)
            wt, c = tw, c + pc
        means.append(m)
        weights.append(wt)
        sizes.append(c)
    return np.repeat(np.asarray(means), np.asarray(sizes))


def fit_isotonic_binary(s, y, sample_weight=None):
    """Hồi quy đẳng trương của y ∈ [0, 1] theo điểm s.

    s: (n,) điểm; y: (n,) nhãn {0, 1} (hoặc giá trị trong [0, 1]); sample_weight: (n,) hoặc None.
    Sắp s tăng dần, gộp các s bằng nhau (trung bình có trọng số), chạy pava.
    Trả về (x_knots, y_knots): hai mảng (m,), x_knots tăng ngặt, y_knots không giảm.
    """
    s = np.asarray(s, dtype=float)
    y = np.asarray(y, dtype=float)
    if s.ndim != 1 or s.shape != y.shape or s.size == 0:
        raise ValueError("s, y phải là mảng 1 chiều cùng shape (n,), n ≥ 1")
    w = _check_weights(sample_weight, s.size)
    order = np.argsort(s, kind="stable")
    s, y, w = s[order], y[order], w[order]
    x_knots, start = np.unique(s, return_index=True)
    w_sum = np.add.reduceat(w, start)
    wy_sum = np.add.reduceat(w * y, start)
    cnt = np.diff(np.append(start, s.size))
    y_sum = np.add.reduceat(y, start)
    with np.errstate(invalid="ignore", divide="ignore"):
        y_grp = np.where(w_sum > 0, wy_sum / w_sum, y_sum / cnt)
    return x_knots, pava(y_grp, w_sum)


def predict_isotonic_binary(s, x_knots, y_knots):
    """Nội suy tuyến tính giữa các nút; ngoài khoảng [x_knots[0], x_knots[-1]] lấy giá trị biên.

    s: (n,); x_knots, y_knots: (m,). Trả về (n,).
    """
    return np.interp(np.asarray(s, dtype=float), x_knots, y_knots)


# ---------------------------------------------------------------------------
# Platt scaling
# ---------------------------------------------------------------------------
def _platt_loss(a, b, s, t, w):
    """Trung bình có trọng số của cross-entropy với nhãn (mềm) t, f = a·s + b."""
    f = a * s + b
    return -np.sum(w * (t * log_sigmoid(f) + (1.0 - t) * log_sigmoid(-f))) / w.sum()


def fit_platt_binary(s, y, smooth=True, sample_weight=None, tol=1e-10, max_iter=100):
    """Platt scaling nhị phân: p = σ(a·s + b).

    s: (n,) điểm; y: (n,) ∈ {0, 1}; sample_weight: (n,) hoặc None.
    smooth=True: nhãn mềm Platt (1999) t₊ = (N₊ + 1)/(N₊ + 2), t₋ = 1/(N₋ + 2),
      N₊, N₋ = SỐ mẫu dương/âm (không trọng số). Nghiệm luôn hữu hạn, kể cả khi
      dữ liệu khả tách hoặc vắng một lớp.
    Tối ưu loss (trung bình có trọng số) bằng Newton 2 biến + lùi bước để loss không tăng.
    Dừng khi ‖∇‖_∞ < tol hoặc hết max_iter. Trả về (a, b) kiểu float.
    """
    s = np.asarray(s, dtype=float)
    y = np.asarray(y, dtype=float)
    if s.ndim != 1 or s.shape != y.shape or s.size == 0:
        raise ValueError("s, y phải là mảng 1 chiều cùng shape (n,), n ≥ 1")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("y phải thuộc {0, 1}")
    w = _check_weights(sample_weight, s.size)
    W = w.sum()

    if smooth:
        n_pos = float(np.sum(y == 1))
        n_neg = float(np.sum(y == 0))
        t = np.where(y == 1, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0))
    else:
        t = y

    # khởi tạo: a = 0, b = logit(trung bình nhãn) → nghiệm đúng khi s không mang thông tin
    t_bar = np.clip(np.sum(w * t) / W, 1e-12, 1.0 - 1e-12)
    a, b = 0.0, float(np.log(t_bar) - np.log1p(-t_bar))
    loss = _platt_loss(a, b, s, t, w)

    for _ in range(max_iter):
        f = a * s + b
        p = sigmoid(f)
        r = w * (p - t) / W
        g = np.array([np.sum(r * s), np.sum(r)])
        if np.max(np.abs(g)) < tol:
            break
        h = w * p * (1.0 - p) / W
        H = np.array([[np.sum(h * s * s), np.sum(h * s)],
                      [np.sum(h * s), np.sum(h)]])
        H += 1e-12 * (np.trace(H) + 1e-12) * np.eye(2)   # tránh suy biến (vd. s hằng)
        d = -np.linalg.solve(H, g)

        step, improved = 1.0, False
        for _ in range(60):
            a_new, b_new = a + step * d[0], b + step * d[1]
            loss_new = _platt_loss(a_new, b_new, s, t, w)
            if np.isfinite(loss_new) and loss_new <= loss:
                improved = True
                break
            step *= 0.5
        if not improved:
            break
        a, b, loss = a_new, b_new, loss_new
    return float(a), float(b)


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------
def _temperature_nll_derivs(beta, Z, y, w):
    """NLL(β) = Σ wᵢ [logsumexp(β zᵢ) − β z_{i,yᵢ}] / Σ w và đạo hàm bậc 1, 2 theo β.

    Z: (n, K); y: (n,) int; w: (n,). Với p = softmax(β z):
    NLL' = E_p[z] − z_y,  NLL'' = Var_p[z] ≥ 0 (lấy trung bình có trọng số).
    """
    W = w.sum()
    BZ = beta * Z
    lse = logsumexp(BZ, axis=1)
    zy = Z[np.arange(Z.shape[0]), y]
    nll = np.sum(w * (lse - beta * zy)) / W
    P = np.exp(BZ - lse[:, None])               # = softmax(βZ), đối số exp ≤ 0
    Ez = np.sum(P * Z, axis=1)
    var = np.sum(P * (Z - Ez[:, None]) ** 2, axis=1)
    g = np.sum(w * (Ez - zy)) / W
    h = np.sum(w * var) / W
    return nll, g, h


def temperature_nll(logits, y, T, sample_weight=None):
    """NLL trung bình (có trọng số) của softmax(logits / T). logits: (n, K); y: (n,) int. Trả về float."""
    Z = np.asarray(logits, dtype=float)
    y = np.asarray(y, dtype=int)
    w = _check_weights(sample_weight, Z.shape[0])
    return _temperature_nll_derivs(1.0 / float(T), Z, y, w)[0]


def fit_temperature(logits, y, sample_weight=None, tol=1e-10, max_iter=100):
    """Temperature scaling: T = argmin_T NLL(softmax(logits / T), y).

    logits: (n, K); y: (n,) int trong [0, K); sample_weight: (n,) hoặc None.
    Tối ưu theo β = 1/T (NLL lồi theo β) bằng Newton 1 chiều, đạo hàm giải tích;
    lùi bước (chia đôi) khi loss tăng hoặc β ≤ 0; β kẹp trong [1e-4, 1e4].
    Dừng khi |NLL'(β)| < tol hoặc hết max_iter. Trả về T = 1/β (float > 0).
    """
    Z = np.asarray(logits, dtype=float)
    if Z.ndim != 2:
        raise ValueError(f"logits phải có shape (n, K), nhận {Z.shape}")
    Z, y = _check_scores_labels(Z, y, Z.shape[1])
    w = _check_weights(sample_weight, Z.shape[0])

    beta = 1.0
    nll, g, h = _temperature_nll_derivs(beta, Z, y, w)
    for _ in range(max_iter):
        if abs(g) < tol:
            break
        d = -g / h if h > 1e-300 else -np.sign(g) * beta
        step, improved = 1.0, False
        for _ in range(60):
            b_new = beta + step * d
            if b_new > 0:
                b_new = min(max(b_new, BETA_MIN), BETA_MAX)
                nll_new, g_new, h_new = _temperature_nll_derivs(b_new, Z, y, w)
                if np.isfinite(nll_new) and nll_new <= nll:
                    improved = True
                    break
            step *= 0.5
        if not improved or b_new == beta:
            break
        beta, nll, g, h = b_new, nll_new, g_new, h_new
    return float(1.0 / beta)


# ---------------------------------------------------------------------------
# Lớp dùng cho D4/D5 (pickle được)
# ---------------------------------------------------------------------------
class NoCalibrator:
    """"none": softmax(scores), dùng để so "trước hiệu chuẩn" (SVM: chỉ để tham khảo)."""

    name = "none"

    def __init__(self, n_classes=3):
        self.n_classes = n_classes

    def fit(self, scores, y, sample_weight=None):
        """Không học gì; chỉ kiểm tra shape. Trả về self."""
        _check_scores_labels(scores, y, self.n_classes)
        return self

    def predict_proba(self, scores):
        """scores: (n, K) → softmax(scores): (n, K)."""
        S, _ = _check_scores_labels(scores, None, self.n_classes)
        return softmax(S, axis=1)


class PlattCalibrator:
    """Platt scaling one-vs-rest: p_k = σ(a_k · s_k + b_k), kẹp rồi chuẩn hóa hàng.

    Thuộc tính sau fit: a_ (K,), b_ (K,).
    """

    name = "platt"

    def __init__(self, n_classes=3, smooth=True):
        self.n_classes = n_classes
        self.smooth = smooth

    def fit(self, scores, y, sample_weight=None):
        """scores: (n, K); y: (n,) int theo core.LABELS; sample_weight: (n,) hoặc None. Trả về self."""
        S, y = _check_scores_labels(scores, y, self.n_classes)
        K = self.n_classes
        self.a_, self.b_ = np.zeros(K), np.zeros(K)
        for k in range(K):
            self.a_[k], self.b_[k] = fit_platt_binary(
                S[:, k], (y == k).astype(float), smooth=self.smooth, sample_weight=sample_weight)
        return self

    def _column_proba(self, S):
        """Xác suất OvR từng cột, CHƯA kẹp/chuẩn hóa: (n, K)."""
        return sigmoid(S * self.a_ + self.b_)

    def predict_proba(self, scores):
        """scores: (n, K) → (n, K), mỗi hàng tổng = 1."""
        S, _ = _check_scores_labels(scores, None, self.n_classes)
        return _clip_normalize(self._column_proba(S))


class IsotonicCalibrator:
    """Hồi quy đẳng trương (PAVA) one-vs-rest theo từng cột, kẹp rồi chuẩn hóa hàng.

    Thuộc tính sau fit: knots_ — list K cặp (x_knots, y_knots).
    """

    name = "isotonic"

    def __init__(self, n_classes=3):
        self.n_classes = n_classes

    def fit(self, scores, y, sample_weight=None):
        """scores: (n, K); y: (n,) int theo core.LABELS; sample_weight: (n,) hoặc None. Trả về self."""
        S, y = _check_scores_labels(scores, y, self.n_classes)
        self.knots_ = [fit_isotonic_binary(S[:, k], (y == k).astype(float), sample_weight)
                       for k in range(self.n_classes)]
        return self

    def _column_proba(self, S):
        """Xác suất OvR từng cột, CHƯA kẹp/chuẩn hóa: (n, K)."""
        return np.column_stack([predict_isotonic_binary(S[:, k], xk, yk)
                                for k, (xk, yk) in enumerate(self.knots_)])

    def predict_proba(self, scores):
        """scores: (n, K) → (n, K), mỗi hàng tổng = 1."""
        S, _ = _check_scores_labels(scores, None, self.n_classes)
        return _clip_normalize(self._column_proba(S))


class TemperatureCalibrator:
    """Temperature scaling: softmax(scores / T). Không đổi argmax.

    Thuộc tính sau fit: T_ (float > 0).
    """

    name = "temperature"

    def __init__(self, n_classes=3):
        self.n_classes = n_classes

    def fit(self, scores, y, sample_weight=None):
        """scores: (n, K) logit; y: (n,) int theo core.LABELS; sample_weight: (n,) hoặc None. Trả về self."""
        S, y = _check_scores_labels(scores, y, self.n_classes)
        self.T_ = fit_temperature(S, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, scores):
        """scores: (n, K) → softmax(scores / T): (n, K)."""
        S, _ = _check_scores_labels(scores, None, self.n_classes)
        return softmax(S / self.T_, axis=1)


_CALIBRATORS = {
    "none": NoCalibrator,
    "platt": PlattCalibrator,
    "isotonic": IsotonicCalibrator,
    "temperature": TemperatureCalibrator,
}


def get_calibrator(name, n_classes=3):
    """Tạo bộ hiệu chuẩn chưa fit. name ∈ {"none", "platt", "isotonic", "temperature"}."""
    if name not in _CALIBRATORS:
        raise ValueError(f"name phải thuộc {sorted(_CALIBRATORS)}, nhận {name!r}")
    return _CALIBRATORS[name](n_classes=n_classes)
