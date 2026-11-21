"""[A2] Hằng số LABELS, sigmoid ổn định, log-sum-exp, softmax — dùng chung cho mọi file (Kiên).

Quy ước: mọi phép exp trên logit phải đi qua các hàm trong file này
(softmax là chỗ duy nhất exp trực tiếp trên logit đã trừ log-sum-exp).
"""

import numpy as np

# Thứ tự nhãn: lua_dao = 0, rac = 1, hop_le = 2
LABELS = ["lua_dao", "rac", "hop_le"]


def log_sigmoid(z):
    """log σ(z), ổn định số.

    z: số thực hoặc mảng bất kỳ shape. Trả về cùng shape.
    log σ(z) = -log(1 + e^{-z}) = -logaddexp(0, -z).
    """
    return -np.logaddexp(0.0, -np.asarray(z, dtype=float))


def sigmoid(z):
    """σ(z) = 1 / (1 + e^{-z}), không tràn số tại z = ±1000.

    z: số thực hoặc mảng bất kỳ shape. Trả về cùng shape, giá trị trong [0, 1].
    Tính bằng exp(log σ(z)): đối số của exp luôn ≤ 0 nên không tràn.
    """
    return np.exp(log_sigmoid(z))


def logsumexp(Z, axis=-1, keepdims=False):
    """log Σ_k exp(Z_k) theo trục `axis`, ổn định số (trừ max trước khi exp).

    Z: mảng (..., K). Trả về mảng bỏ trục `axis` (hoặc giữ kích thước 1 nếu keepdims).
    Hàng toàn -inf cho kết quả -inf (không ra nan).
    """
    Z = np.asarray(Z, dtype=float)
    m = np.max(Z, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        out = np.log(np.sum(np.exp(Z - m), axis=axis, keepdims=True)) + m
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


def softmax(Z, axis=-1):
    """softmax(Z) = exp(Z - logsumexp(Z)) theo trục `axis`.

    Z: mảng (..., K) logit. Trả về cùng shape, tổng theo `axis` bằng 1.
    """
    Z = np.asarray(Z, dtype=float)
    return np.exp(Z - logsumexp(Z, axis=axis, keepdims=True))


if __name__ == "__main__":
    print("1+1=2")
