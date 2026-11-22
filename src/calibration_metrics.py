"""[A4] Sai số hiệu chuẩn kỳ vọng (ECE), biểu đồ độ tin cậy, log-loss — dùng cho C2, D5 (Nhật).

    ECE = Σ_{m=1..M} |B_m|/n · |acc(B_m) − conf(B_m)|

Quy ước (RULES.md):
- M = 15 bin đều trên [0, 1]; bin m = ((m−1)/M, m/M], riêng bin 1 gồm cả 0. Bin rỗng đóng góp 0.
- kind="binary":    probs (n,) xác suất lớp dương; conf = p, acc = tỷ lệ y = 1 trong bin.
- kind="top_label": probs (n, K); conf = max_k p_k, acc = tỷ lệ argmax đúng.
- kind="classwise": probs (n, K); trung bình theo K lớp của ECE binary trên cột k với nhãn (y == k).
- "ECE riêng lớp lừa đảo" ở D5 = ece(probs[:, 0], labels == 0, kind="binary").
"""

import numpy as np

KINDS = ("binary", "top_label", "classwise")


# ---------------------------------------------------------------------------
# Kiểm tra đầu vào + gán bin
# ---------------------------------------------------------------------------
def _check(probs, labels, n_bins, kind):
    """Trả về (P, y) đã kiểm tra. Sai shape/kind/giá trị → ValueError."""
    if kind not in KINDS:
        raise ValueError(f"kind phải thuộc {KINDS}, nhận {kind!r}")
    if int(n_bins) != n_bins or n_bins < 1:
        raise ValueError(f"n_bins phải là số nguyên ≥ 1, nhận {n_bins}")
    P = np.asarray(probs, dtype=float)
    y = np.asarray(labels)
    if kind == "binary" and P.ndim != 1:
        raise ValueError(f"kind='binary' cần probs shape (n,), nhận {P.shape}")
    if kind != "binary" and P.ndim != 2:
        raise ValueError(f"kind={kind!r} cần probs shape (n, K), nhận {P.shape}")
    n = P.shape[0]
    if n == 0:
        raise ValueError("probs rỗng")
    if y.shape != (n,):
        raise ValueError(f"labels phải có shape ({n},), nhận {y.shape}")
    if np.any(~np.isfinite(P)) or P.min() < -1e-12 or P.max() > 1 + 1e-12:
        raise ValueError("probs phải nằm trong [0, 1] và không nan")
    P = np.clip(P, 0.0, 1.0)
    if y.dtype == bool:
        y = y.astype(int)
    if not np.all(np.mod(y, 1) == 0):
        raise ValueError("labels phải là số nguyên")
    y = y.astype(int)
    hi = 2 if kind == "binary" else P.shape[1]
    if y.min() < 0 or y.max() >= hi:
        raise ValueError(f"labels phải nằm trong [0, {hi})")
    return P, y


def _bin_index(conf, n_bins):
    """conf: (n,) trong [0, 1] → chỉ số bin 0..M−1; bin m (1-index) = ((m−1)/M, m/M], 0 vào bin 1."""
    edges = np.arange(n_bins + 1) / n_bins
    idx = np.searchsorted(edges, conf, side="left") - 1
    return np.clip(idx, 0, n_bins - 1), edges


def _curve(conf, correct, n_bins):
    """conf: (n,) độ tin cậy; correct: (n,) 0/1. Trả về dict edges (M+1,), conf, acc, count (M,).

    Bin rỗng có conf = acc = nan, count = 0.
    """
    idx, edges = _bin_index(conf, n_bins)
    count = np.bincount(idx, minlength=n_bins)
    s_conf = np.bincount(idx, weights=conf, minlength=n_bins)
    s_acc = np.bincount(idx, weights=correct.astype(float), minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.where(count > 0, s_conf / count, np.nan)
        a = np.where(count > 0, s_acc / count, np.nan)
    return {"edges": edges, "conf": c, "acc": a, "count": count}


def _ece_from_curve(curve):
    """Σ count/n · |acc − conf|, bỏ bin rỗng. curve: dict của _curve."""
    count = curve["count"]
    keep = count > 0
    n = count.sum()
    return float(np.sum(count[keep] / n * np.abs(curve["acc"][keep] - curve["conf"][keep])))


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
def reliability_curve(probs, labels, n_bins=15, kind="top_label"):
    """Số liệu biểu đồ độ tin cậy.

    probs: (n,) (binary) hoặc (n, K) (top_label, classwise); labels: (n,) int.
    Trả về dict {"edges": (M+1,), "conf": (M,), "acc": (M,), "count": (M,)};
    bin rỗng có conf/acc = nan. kind="classwise": list K dict (lớp k: cột k, nhãn y == k).
    """
    P, y = _check(probs, labels, n_bins, kind)
    if kind == "binary":
        return _curve(P, y == 1, n_bins)
    if kind == "top_label":
        return _curve(P.max(axis=1), P.argmax(axis=1) == y, n_bins)
    return [_curve(P[:, k], y == k, n_bins) for k in range(P.shape[1])]


def ece(probs, labels, n_bins=15, kind="top_label") -> float:
    """Sai số hiệu chuẩn kỳ vọng.

    probs: (n,) xác suất lớp dương hoặc (n, K); labels: (n,);
    kind: 'binary' | 'top_label' | 'classwise'. Sai shape so với kind → ValueError.
    Trả về float ≥ 0.
    """
    curve = reliability_curve(probs, labels, n_bins=n_bins, kind=kind)
    if kind == "classwise":
        return float(np.mean([_ece_from_curve(c) for c in curve]))
    return _ece_from_curve(curve)


def log_loss(probs, labels, eps=1e-15) -> float:
    """Trung bình −log p_{y} (p kẹp vào [eps, 1 − eps]).

    probs: (n, K) với labels (n,) int trong [0, K); hoặc (n,) xác suất lớp dương với labels ∈ {0, 1}.
    Trả về float.
    """
    P = np.asarray(probs, dtype=float)
    y = np.asarray(labels).astype(int)
    if P.ndim == 1:
        P = np.column_stack([1.0 - P, P])
    if P.ndim != 2 or y.shape != (P.shape[0],):
        raise ValueError(f"shape không khớp: probs {np.shape(probs)}, labels {np.shape(labels)}")
    if y.min() < 0 or y.max() >= P.shape[1]:
        raise ValueError(f"labels phải nằm trong [0, {P.shape[1]})")
    p = np.clip(P[np.arange(P.shape[0]), y], eps, 1.0 - eps)
    return float(-np.mean(np.log(p)))


def plot_reliability(probs, labels, n_bins=15, kind="top_label", ax=None, label=None,
                     show_hist=True):
    """Vẽ biểu đồ độ tin cậy lên ax (tạo mới nếu None); không gọi plt.show().

    Vẽ đường chéo y = x (một lần cho mỗi ax), đường acc theo conf của các bin không rỗng,
    chú thích "<label> (ECE = …)". show_hist: cột số mẫu mỗi bin ở trục y phụ (bên phải).
    Gọi nhiều lần trên cùng ax để vẽ trước/sau hiệu chuẩn.
    kind="classwise": mỗi lớp một đường, chú thích ghi ECE classwise.
    probs, labels: như ece. Trả về matplotlib Axes chính.
    """
    import matplotlib.pyplot as plt

    curve = reliability_curve(probs, labels, n_bins=n_bins, kind=kind)
    e = ece(probs, labels, n_bins=n_bins, kind=kind)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    if not getattr(ax, "_rel_diag", False):
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Hoàn hảo")
        ax._rel_diag = True
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Độ tin cậy (conf)")
        ax.set_ylabel("Độ chính xác (acc)")

    text = f"{label + ' ' if label else ''}(ECE = {e:.4f})"
    curves = curve if kind == "classwise" else [curve]
    color = None
    for k, c in enumerate(curves):
        keep = c["count"] > 0
        lbl = text if k == 0 else None
        ls = ["-", "--", ":", "-."][k % 4] if kind == "classwise" else "-"
        line, = ax.plot(c["conf"][keep], c["acc"][keep], marker="o", markersize=4,
                        linestyle=ls, color=color, label=lbl)
        color = line.get_color()

    if show_hist:
        ax2 = getattr(ax, "_rel_hist_ax", None)
        if ax2 is None:
            ax2 = ax.twinx()
            ax2.set_ylabel("Số mẫu mỗi bin")
            ax._rel_hist_ax = ax2
            ax.set_zorder(ax2.get_zorder() + 1)
            ax.patch.set_visible(False)
        edges = curves[0]["edges"]
        count = np.sum([c["count"] for c in curves], axis=0)
        ax2.bar(edges[:-1], count, width=np.diff(edges), align="edge",
                alpha=0.2, color=color, edgecolor=color)

    ax.legend(loc="upper left", fontsize="small")
    return ax
