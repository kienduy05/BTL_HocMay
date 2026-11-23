"""[D5] Đánh giá trên tập test: quyết định theo chi phí (B5), recall lừa đảo tại ngưỡng FPR cố định,
ECE, log-loss, bootstrap (mean ± std, KTC 95%), bootstrap ghép cặp, điều chỉnh tỷ lệ lớp (B6) (Nhật).

Quy ước (RULES.md):
- Nhãn y: mảng số nguyên (n,) theo core.LABELS (lua_dao = 0, rac = 1, hop_le = 2).
- probs: (n, K), mỗi hàng tổng = 1 (xác suất đã hoặc chưa hiệu chuẩn).
- Ma trận chi phí C[thật, đoán] (K, K).
- Bootstrap: 1.000 lần, seed 42, chỉ lấy mẫu lại dự đoán đã có — không huấn luyện lại,
  không chọn lại ngưỡng hay bộ hiệu chuẩn.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from calibration_metrics import ece, log_loss
from core import LABELS, logsumexp, softmax

SEED = 42
FRAUD = 0  # chỉ số lớp lua_dao trong LABELS
METRICS = ["f1_macro", "chi_phi_tb", "recall_lua_dao_at_fpr", "fpr_at_threshold",
           "ece_top_label", "ece_lua_dao", "log_loss"]


# ---------------------------------------------------------------------------
# Quyết định và chỉ số
# ---------------------------------------------------------------------------
def cost_decision(probs, C):
    """Quy tắc Bayes theo chi phí (B5, dạng K lớp): ŷ = argmin_j Σ_i C[i, j]·p(i|x).

    probs: (n, K); C: (K, K) = C[thật, đoán]. Trả về ŷ (n,) int.
    """
    P = np.asarray(probs, dtype=float)
    C = np.asarray(C, dtype=float)
    if P.ndim != 2 or C.shape != (P.shape[1], P.shape[1]):
        raise ValueError(f"probs (n, K) và C (K, K) không khớp: {P.shape}, {C.shape}")
    return np.argmin(P @ C, axis=1)


def recall_fpr_at_threshold(score, is_pos, threshold):
    """Báo dương khi score ≥ threshold (threshold None = không bao giờ báo).

    score: (n,); is_pos: (n,) bool. Trả về (recall, fpr); lớp nào vắng mặt thì giá trị tương ứng = nan.
    """
    score = np.asarray(score, dtype=float)
    is_pos = np.asarray(is_pos, dtype=bool)
    flag = np.zeros_like(is_pos) if threshold is None else score >= threshold
    recall = float(flag[is_pos].mean()) if is_pos.any() else np.nan
    fpr = float(flag[~is_pos].mean()) if (~is_pos).any() else np.nan
    return recall, fpr


def threshold_at_fpr(score, is_pos, max_fpr=0.01):
    """Ngưỡng nhỏ nhất trong {các điểm khác nhau} ∪ {+∞} sao cho FPR = P(score ≥ t | âm) ≤ max_fpr.

    Cùng quy tắc D4 dùng để chốt ngưỡng trên val_b. score: (n,); is_pos: (n,) bool.
    Trả về (t, fpr, recall); t = None nếu chỉ +∞ thỏa.
    """
    score = np.asarray(score, dtype=float)
    is_pos = np.asarray(is_pos, dtype=bool)
    neg, pos = score[~is_pos], score[is_pos]
    for t in np.append(np.unique(score), np.inf):
        fpr = float(np.mean(neg >= t))
        if fpr <= max_fpr:
            rec = float(np.mean(pos >= t)) if pos.size else np.nan
            return (None if np.isinf(t) else float(t)), fpr, rec


def evaluate_probs(probs, y, C, threshold, n_bins=15):
    """Mọi chỉ số D5 cho một bộ xác suất.

    probs: (n, K); y: (n,) int; C: (K, K); threshold: ngưỡng trên p(lua_dao) (hoặc None).
    Trả về dict:
      f1_macro (theo quy tắc chi phí), chi_phi_tb = mean C[y, ŷ],
      recall_lua_dao_at_fpr, fpr_at_threshold (lừa đảo so với phần còn lại, ngưỡng cố định),
      ece_top_label, ece_lua_dao (binary trên cột 0), log_loss,
      recall_<lớp> theo quy tắc chi phí cho từng lớp,
      bao_nham_lua_dao = tỉ lệ tin không lừa đảo bị quy tắc chi phí đoán là lừa đảo.
    """
    P = np.asarray(probs, dtype=float)
    y = np.asarray(y).astype(int)
    C = np.asarray(C, dtype=float)
    K = P.shape[1]
    pred = cost_decision(P, C)
    is_fraud = y == FRAUD
    rec, fpr = recall_fpr_at_threshold(P[:, FRAUD], is_fraud, threshold)
    out = {
        "f1_macro": float(f1_score(y, pred, average="macro", labels=list(range(K)), zero_division=0)),
        "chi_phi_tb": float(C[y, pred].mean()),
        "recall_lua_dao_at_fpr": rec,
        "fpr_at_threshold": fpr,
        "ece_top_label": ece(P, y, n_bins=n_bins, kind="top_label"),
        "ece_lua_dao": ece(P[:, FRAUD], is_fraud.astype(int), n_bins=n_bins, kind="binary"),
        "log_loss": log_loss(P, y),
    }
    for k in range(K):
        m = y == k
        out[f"recall_{LABELS[k]}"] = float(np.mean(pred[m] == k)) if m.any() else np.nan
    out["bao_nham_lua_dao"] = float(np.mean(pred[~is_fraud] == FRAUD)) if (~is_fraud).any() else np.nan
    return out


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def bootstrap_indices(n, n_boot=1000, seed=SEED):
    """Chỉ số lấy mẫu lại có hoàn lại theo tin nhắn: (n_boot, n) int. Dùng chung cho mọi mô hình
    để bootstrap ghép cặp."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n))


def bootstrap_metrics(probs, y, C, threshold, idx, n_bins=15):
    """Tính evaluate_probs trên từng mẫu bootstrap. idx: (B, n). Trả về DataFrame (B dòng × chỉ số)."""
    P = np.asarray(probs, dtype=float)
    y = np.asarray(y).astype(int)
    return pd.DataFrame([evaluate_probs(P[i], y[i], C, threshold, n_bins) for i in idx])


def summarize(point, boot, alpha=0.05):
    """point: dict giá trị gốc; boot: DataFrame (B × chỉ số).

    Trả về DataFrame: chi_so, goc, mean, std (ddof = 1), ci_low, ci_high (phân vị 2,5–97,5),
    n_hop_le (số mẫu bootstrap không nan). nan (ví dụ mẫu thiếu lớp) bị bỏ qua.
    """
    rows = []
    for m in boot.columns:
        v = boot[m].to_numpy(dtype=float)
        v = v[~np.isnan(v)]
        rows.append({
            "chi_so": m, "goc": point[m],
            "mean": float(v.mean()) if v.size else np.nan,
            "std": float(v.std(ddof=1)) if v.size > 1 else np.nan,
            "ci_low": float(np.percentile(v, 100 * alpha / 2)) if v.size else np.nan,
            "ci_high": float(np.percentile(v, 100 * (1 - alpha / 2))) if v.size else np.nan,
            "n_hop_le": int(v.size),
        })
    return pd.DataFrame(rows)


def paired_difference(point_a, point_b, boot_a, boot_b, alpha=0.05):
    """Bootstrap ghép cặp: boot_a, boot_b tính trên CÙNG chỉ số mẫu. Hiệu = a − b.

    Trả về DataFrame: chi_so, hieu_goc, mean, std, ci_low, ci_high, ci_chua_0 (bool).
    Chỉ kết luận a khác b khi KTC của hiệu không chứa 0.
    """
    point = {m: point_a[m] - point_b[m] for m in boot_a.columns}
    s = summarize(point, boot_a - boot_b, alpha).rename(columns={"goc": "hieu_goc"})
    s["ci_chua_0"] = (s.ci_low <= 0) & (s.ci_high >= 0)
    return s


# ---------------------------------------------------------------------------
# Điều chỉnh tỷ lệ lớp (B6, dạng K lớp)
# ---------------------------------------------------------------------------
def adjust_prior(probs, pi_train, pi_new, eps=1e-300):
    """p′(y|x) ∝ p(y|x)·π′_y/π_y (B6), tính ở thang log rồi core.softmax.

    probs: (n, K); pi_train, pi_new: (K,) dương. Trả về (n, K), mỗi hàng tổng = 1.
    Với K = 2: logit p′₁ = logit p₁ + log[π′(1−π)/(π(1−π′))].
    """
    P = np.asarray(probs, dtype=float)
    pi_train = np.asarray(pi_train, dtype=float)
    pi_new = np.asarray(pi_new, dtype=float)
    if np.any(pi_train <= 0) or np.any(pi_new <= 0):
        raise ValueError("pi_train và pi_new phải dương")
    logp = np.log(np.clip(P, eps, 1.0)) + np.log(pi_new) - np.log(pi_train)
    return softmax(logp, axis=1)


def scenario_prior(pi_train, p_fraud):
    """π′ cho kịch bản lừa đảo chiếm p_fraud; hai lớp còn lại giữ tỉ lệ với nhau như train. (K,)."""
    pi_train = np.asarray(pi_train, dtype=float)
    rest = pi_train.copy()
    rest[FRAUD] = 0.0
    out = (1.0 - p_fraud) * rest / rest.sum()
    out[FRAUD] = p_fraud
    return out


def resample_by_prior(y, pi_new, n, rng):
    """Lấy mẫu lại (có hoàn lại, trong từng lớp) sao cho số tin lớp k = round(n·π′_k).

    y: (N,) int; pi_new: (K,); n: cỡ mẫu. Phần làm tròn dồn vào lớp có π′ lớn nhất. Trả về (n,) chỉ số.
    """
    y = np.asarray(y).astype(int)
    pi_new = np.asarray(pi_new, dtype=float)
    counts = np.round(n * pi_new).astype(int)
    counts[np.argmax(pi_new)] += n - counts.sum()
    parts = []
    for k, c in enumerate(counts):
        pool = np.flatnonzero(y == k)
        if c > 0:
            if pool.size == 0:
                raise ValueError(f"lớp {k} không có mẫu để lấy")
            parts.append(rng.choice(pool, size=c, replace=True))
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Đặc trưng có trọng số lớn
# ---------------------------------------------------------------------------
def nb_log_ratio(feature_log_prob):
    """Tỷ số log-xác suất của Naive Bayes, lớp k so với các lớp còn lại.

    feature_log_prob: (K, d) = log p(w | k). Trả về (d, K):
    r[w, k] = log p(w|k) − log( (1/(K−1)) Σ_{j≠k} p(w|j) ).
    """
    L = np.asarray(feature_log_prob, dtype=float)
    K = L.shape[0]
    out = np.empty((L.shape[1], K))
    for k in range(K):
        others = np.delete(L, k, axis=0)
        out[:, k] = L[k] - (logsumexp(others, axis=0) - np.log(K - 1))
    return out


def top_features(weights, names, top=10):
    """Đặc trưng có trọng số lớn nhất cho từng lớp.

    weights: (d, K); names: (d,) tên đặc trưng. Trả về DataFrame: lop, hang (1..top), dac_trung, trong_so.
    """
    Wt = np.asarray(weights, dtype=float)
    names = np.asarray(names)
    rows = []
    for k in range(Wt.shape[1]):
        order = np.argsort(-Wt[:, k], kind="stable")[:top]
        for r, j in enumerate(order, 1):
            rows.append({"lop": LABELS[k], "hang": r, "dac_trung": names[j], "trong_so": float(Wt[j, k])})
    return pd.DataFrame(rows)
