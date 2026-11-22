"""[D3] Tiền xử lý tin nhắn (Đức).

D3a – chạy TRƯỚC khi chia dữ liệu (không học gì từ dữ liệu, chỉ dùng quy tắc + từ điển cố định):
    normalize_text()  : NFC, dấu thanh, biến thể cố ý, thẻ [LINK]/[TIỀN]/[SĐT], teencode
    tokenize()        : tách từ bằng pyvi (cho n-gram từ)
    deduplicate()     : loại trùng tuyệt đối + gần trùng (Jaccard trên 5-gram ký tự), giữ tin cũ nhất
    clean()           : ghép các bước trên cho một DataFrame
D3b – chạy SAU khi Kiên chia dữ liệu (D4a):
    fit_tfidf()       : TF-IDF n-gram từ + n-gram ký tự, CHỈ fit trên train
    transform_tfidf() : áp vectorizer đã fit cho train/val_a/val_b/test

Chạy: python src/preprocess.py  (messages_masked.csv + labels_final.csv → data/labeled/clean.csv)

Quy ước thẻ: mask.py sinh [NUMBER] [MONEY] [TIME] [DATE] [OTP] [CODE] [NAME] [POINT] [EMAIL].
Ở đây [MONEY] → [TIỀN]; link (mask.py không che) → [LINK]; số điện thoại / số tiền còn sót sau
mask.py → [SĐT] / [TIỀN]. Hai thẻ cờ thêm: [LINK_LẠ] (đuôi tên miền rẻ tiền hoặc tên miền giả
thương hiệu có gạch nối), [KÝ_TỰ_LẠ] (nhiều ký tự chèn giữa chữ kiểu "Li`xi~phO;ng").
Đã chốt trong RULES.md và PROGRESS.md (2026-11-22).

Không in nội dung tin nhắn ra log (RULES.md). Không truy cập bất kỳ đường link nào.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import LABELS  # noqa: E402

# ---------------------------------------------------------------------------
# Thẻ và placeholder
# ---------------------------------------------------------------------------
# Các thẻ được bảo vệ bằng placeholder toàn chữ cái để các bước sau (bỏ dấu câu, đổi chữ số,
# teencode, pyvi) không làm vỡ thẻ. Cuối cùng placeholder được trả về dạng "[THẺ]".
TAGS = ["NUMBER", "TIỀN", "TIME", "DATE", "OTP", "CODE", "NAME", "POINT", "EMAIL",
        "LINK", "LINK_LẠ", "SĐT", "KÝ_TỰ_LẠ"]
_PH = {t: f"zzqtag{chr(97 + i)}zz" for i, t in enumerate(TAGS)}
_PH_BACK = {v: f"[{k}]" for k, v in _PH.items()}
_PH_RE = re.compile("|".join(_PH_BACK))
MASK_TAG_MAP = {"number": "NUMBER", "money": "TIỀN", "time": "TIME", "date": "DATE",
                "otp": "OTP", "code": "CODE", "name": "NAME", "point": "POINT", "email": "EMAIL",
                # thẻ đã đổi tên (cho phép chạy lại trên văn bản đã làm sạch)
                "tiền": "TIỀN", "link": "LINK", "link_lạ": "LINK_LẠ", "sđt": "SĐT",
                "ký_tự_lạ": "KÝ_TỰ_LẠ"}
_MASK_TAG_RE = re.compile(r"\[\s*(" + "|".join(map(re.escape, MASK_TAG_MAP)) + r")\s*\]")


def _ph(tag: str) -> str:
    return f" {_PH[tag]} "


# ---------------------------------------------------------------------------
# Từ điển cố định
# ---------------------------------------------------------------------------
# Tên miền cấp cao: danh sách cố định. Nhóm "mơ hồ" trùng với từ tiếng Việt không dấu
# ("de" = để, "me" = mẹ, "co" = có...) chỉ được nhận là link khi có ≥ 2 dấu chấm hoặc có đường dẫn "/".
TLD_COMMON = ["com", "vn", "net", "org", "gov", "edu", "info", "biz", "asia", "top", "vip",
              "xyz", "cc", "icu", "club", "online", "shop", "live", "bid", "link", "pro", "site",
              "buzz", "fun", "ml", "tech", "store", "space", "life", "band", "run", "ru", "health",
              "casino", "bond", "win", "tk", "app", "io", "page", "store", "website", "onl"]
TLD_AMBIGUOUS = ["de", "me", "co", "to", "ly", "at", "us", "sg", "nz", "la", "so"]
TLD_SUSPICIOUS = {"top", "vip", "xyz", "cc", "icu", "club", "online", "shop", "live", "bid",
                  "link", "pro", "site", "buzz", "fun", "ml", "tech", "store", "space", "life",
                  "band", "run", "ru", "health", "casino", "bond", "win", "tk", "website", "onl"}

# Thương hiệu hay bị giả mạo: dùng để gộp biến thể (V1etc0mbank, vlETCOMBANK, Zal0Pay...).
BRANDS = ["vietcombank", "techcombank", "vietinbank", "agribank", "sacombank", "mbbank",
          "vpbank", "tpbank", "bidv", "acb", "vib", "shb", "hdbank", "shopee", "lazada", "tiki",
          "momo", "zalopay", "zalo", "viettel", "vinaphone", "mobifone", "vietnamobile",
          "facebook", "tiktok", "telegram", "google", "apple", "garena", "grab"]

# Teencode / viết tắt: chỉ thay khi là NGUYÊN một từ.
TEENCODE = {
    "k": "không", "ko": "không", "kg": "không", "khg": "không", "hok": "không", "hk": "không",
    "đc": "được", "dc": "được",
    "đk": "đăng ký", "dk": "đăng ký", "đky": "đăng ký", "dky": "đăng ký", "đký": "đăng ký",
    "tk": "tài khoản", "tkhoan": "tài khoản", "ck": "chuyển khoản", "stk": "số tài khoản",
    "sđt": "số điện thoại", "sdt": "số điện thoại", "đt": "điện thoại",
    "hnay": "hôm nay", "mng": "mọi người", "trg": "trong", "vs": "với", "mk": "mình",
    "mik": "mình", "cty": "công ty", "gd": "giao dịch", "nv": "nhân viên", "lh": "liên hệ",
    "qk": "quý khách", "ctv": "cộng tác viên", "nhìu": "nhiều", "nhiu": "nhiều",
}

# Chữ số / ký hiệu thay chữ cái, chỉ áp khi nằm GIỮA hai chữ cái (V1etc0mbank, kh0a, d!eu, kh@ch).
LEET = {"0": "o", "1": "i", "3": "e", "4": "a", "6": "b", "@": "a", "!": "i", "$": "s"}

_TONES = "̣̀́̃̉"  # huyền, sắc, ngã, hỏi, nặng (dạng tổ hợp)
_LETTER = r"[^\W\d_]"

# ---------------------------------------------------------------------------
# Biểu thức chính quy
# ---------------------------------------------------------------------------
_TLD_ALL = "|".join(sorted(set(TLD_COMMON) | set(TLD_AMBIGUOUS), key=len, reverse=True))
_URL_SCHEME_RE = re.compile(r"(?:https?|hxxps?)\s*[:.]?\s*/\s*/\s*[^\s]*")
_URL_WWW_RE = re.compile(r"\bwww\.[^\s]+")
_URL_BARE_RE = re.compile(
    r"(?<![\w@\[])((?:[a-z0-9][a-z0-9_\-]*\.)+)(" + _TLD_ALL + r")\b(/[^\s]*)?")
_PHONE_RE = re.compile(r"(?<![\w])(?:\+?84|0)(?:[\s.\-]?\d){8,10}(?![\w])")
_MONEY_RE = re.compile(
    r"(?<![\w])\d+(?:[.,]\d{3})*(?:[.,]\d+)?\s?"
    r"(?:vnđ|vnd|đồng|dong|đ|₫|k|tr|triệu|trieu|tỷ|ty|nghìn|nghin|ngàn|ngan)(?![\w])")
_SEP_LETTERS_RE = re.compile(
    r"(?<![\w])(?:" + _LETTER + r"\s?[.\-_/*]\s?){2,}" + _LETTER + r"(?![\w])")
_ODD_PUNCT_RE = re.compile(_LETTER + r"[`~^;*$|\"'!<>{}:,]+(?=" + _LETTER + ")")
_LEET_MID_RE = re.compile(r"(?<=" + _LETTER + r")[0134@!$6](?=" + _LETTER + ")")
_LEET_END_RE = re.compile(r"(?<=" + _LETTER + r"{2})0(?![\w])")
_REPEAT_RE = re.compile(r"(" + _LETTER + r")\1{2,}")
_CAMEL_RE = re.compile(r"(?<=[a-zà-ỹđ])(?=[A-ZÀ-ỸĐ])")
_TONE_RE = re.compile(r"(?<!q)([ou])([aey])([" + _TONES + r"])(?![^\W\d_]|[̀-ͯ])")
_PUNCT_RE = re.compile(r"[^\w\s%]|_")
_SPACE_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Các bước chuẩn hóa nhỏ (đều nhận và trả về str)
# ---------------------------------------------------------------------------
def normalize_unicode(text: str) -> str:
    """NFKC (đưa chữ toán học 𝙉𝙝, chữ full-width về chữ thường) rồi NFC; bỏ ký tự ẩn.

    text: str → str.
    """
    text = unicodedata.normalize("NFKC", text)
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat in ("Cf", "Co", "Cs", "So", "Sk") or ch in "ㅤᅟᅠ":
            out.append(" ")
        elif cat == "Cc":
            out.append(" ")
        else:
            out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def normalize_tone(text: str) -> str:
    """Thống nhất vị trí dấu thanh kiểu mới cho vần oa/oe/uy cuối âm tiết: hoà → hòa, thuý → thúy.

    Không đổi khi sau vần còn phụ âm/nguyên âm (hoàng, khuyến) hoặc khi 'u' thuộc 'qu' (quý).
    text: str (nên đã viết thường) → str dạng NFC.
    """
    d = unicodedata.normalize("NFD", text)
    d = _TONE_RE.sub(lambda m: m.group(1) + m.group(3) + m.group(2), d)
    return unicodedata.normalize("NFC", d)


def strip_accents(text: str) -> str:
    """Bỏ dấu tiếng Việt (đ → d). text: str → str."""
    d = unicodedata.normalize("NFD", text)
    d = "".join(ch for ch in d if unicodedata.category(ch) != "Mn")
    return d.replace("đ", "d").replace("Đ", "D")


def _fold(token: str) -> str:
    """Dạng gộp để so khớp thương hiệu: đổi leet, l/j → i, bỏ dấu và . _ -."""
    t = strip_accents(token.lower())
    t = "".join(LEET.get(c, c) for c in t)
    t = t.replace("l", "i").replace("j", "i")
    return re.sub(r"[._\-]", "", t)


_BRAND_FOLD = {_fold(b): b for b in BRANDS}


def _replace_urls(text: str, mark_suspicious: bool) -> str:
    def tag(host: str) -> str:
        host = host.strip("./:")
        tld = host.rsplit(".", 1)[-1] if "." in host else ""
        lab = host.rsplit(".", 1)[0]
        fake_brand = "-" in lab and any(b in lab.replace("-", "") for b in BRANDS)
        odd = mark_suspicious and (tld in TLD_SUSPICIOUS or fake_brand)
        return _ph("LINK") + (_ph("LINK_LẠ") if odd else "")

    def host_of(url: str) -> str:
        url = re.sub(r"^(?:https?|hxxps?)\s*[:.]?\s*/\s*/\s*", "", url)
        url = re.sub(r"^www\.", "", url)
        return re.split(r"[/?#\s]", url, maxsplit=1)[0]

    text = _URL_SCHEME_RE.sub(lambda m: tag(host_of(m.group(0))), text)
    text = _URL_WWW_RE.sub(lambda m: tag(host_of(m.group(0))), text)

    def bare(m: re.Match) -> str:
        labels, tld, path = m.group(1), m.group(2), m.group(3)
        if tld in TLD_AMBIGUOUS and labels.count(".") < 2 and not path:
            return m.group(0)
        return tag(labels + tld)

    return _URL_BARE_RE.sub(bare, text)


def _join_separated(m: re.Match) -> str:
    return re.sub(r"[\s.\-_/*]", "", m.group(0))


def _brand_token(tok: str) -> str:
    core = tok.strip("[](){}<>.,;:!?\"'`~^*")
    if len(core) < 3:
        return tok
    b = _BRAND_FOLD.get(_fold(core))
    return tok.replace(core, b) if b else tok


def normalize_text(text: str, mark_suspicious_link: bool = True) -> str:
    """Làm sạch một tin nhắn (D3a). Chỉ dùng quy tắc + từ điển cố định, không học từ dữ liệu.

    Các bước: Unicode → tách chữ dính kiểu CamelCase (từ dài ≥ 15 ký tự) → viết thường → dấu thanh
    → thẻ mask.py → [LINK] → [SĐT]/[TIỀN] còn sót → ghép chữ bị chèn ký tự (n.g.â.n → ngân)
    → cờ [KÝ_TỰ_LẠ] → chữ số thay chữ (kh0a → khoa) → gộp tên thương hiệu → bỏ dấu câu
    → rút gọn ký tự lặp (≥ 3 lần) → teencode.
    text: str → str (có dấu, viết thường, thẻ dạng [THẺ]).
    """
    return _restore(_normalize_ph(text, mark_suspicious_link))


def _normalize_ph(text: str, mark_suspicious_link: bool = True) -> str:
    """Như normalize_text nhưng giữ placeholder (dùng nội bộ cho bỏ dấu và pyvi)."""
    if not isinstance(text, str):
        text = "" if text is None or (isinstance(text, float) and np.isnan(text)) else str(text)
    t = normalize_unicode(text)
    t = " ".join(_CAMEL_RE.sub(" ", w) if len(w) >= 15 else w for w in t.split())
    t = t.lower()
    t = normalize_tone(t)
    t = _MASK_TAG_RE.sub(lambda m: _ph(MASK_TAG_MAP[m.group(1)]), t)
    t = _replace_urls(t, mark_suspicious_link)
    t = _PHONE_RE.sub(lambda m: _ph("SĐT"), t)
    t = _MONEY_RE.sub(lambda m: _ph("TIỀN"), t)
    t = _SEP_LETTERS_RE.sub(_join_separated, t)
    odd = len(_ODD_PUNCT_RE.findall(t)) >= 4
    t = _LEET_MID_RE.sub(lambda m: LEET[m.group(0)], t)
    t = _LEET_END_RE.sub("o", t)
    t = " ".join(_brand_token(w) for w in t.split())
    t = _PUNCT_RE.sub(" ", t)
    t = _REPEAT_RE.sub(r"\1", t)
    words = []
    for w in t.split():
        words.extend(TEENCODE.get(w, w).split())
    if odd:
        words.append(_PH["KÝ_TỰ_LẠ"])
    return " ".join(words)


def _restore(text_ph: str) -> str:
    return _SPACE_RE.sub(" ", _PH_RE.sub(lambda m: _PH_BACK[m.group(0)], text_ph)).strip()


def tokenize(text_ph: str) -> str:
    """Tách từ bằng pyvi; từ ghép nối bằng '_'. text_ph: str (có placeholder) → str."""
    from pyvi import ViTokenizer
    if not text_ph:
        return ""
    tok = ViTokenizer.tokenize(text_ph)
    # pyvi đôi khi nối placeholder với từ bên cạnh ("hi_zzqtag…") → tách lại cho thẻ đứng riêng
    tok = re.sub(r"_?(zzqtag[a-z]zz)_?", r" \1 ", tok)
    return _restore(tok)


def process_text(text: str, mark_suspicious_link: bool = True) -> dict:
    """Trả về 3 dạng của một tin: dict với khóa
    'text_clean' (có dấu), 'text_khongdau' (bỏ dấu, cho n-gram ký tự), 'text_tok' (pyvi, cho n-gram từ).
    """
    ph = _normalize_ph(text, mark_suspicious_link)
    return {"text_clean": _restore(ph),
            "text_khongdau": _restore(strip_accents(ph)),
            "text_tok": tokenize(ph)}


# ---------------------------------------------------------------------------
# Ngày tháng
# ---------------------------------------------------------------------------
def parse_dates(s: pd.Series) -> pd.Series:
    """Chuẩn ngày về datetime: nhận M/D/YYYY (file Drive) hoặc YYYY-MM-DD; rỗng → NaT.

    s: Series (n,) str → Series (n,) datetime64.
    """
    s = s.fillna("").astype(str).str.strip()
    a = pd.to_datetime(s, format="%m/%d/%Y", errors="coerce")
    b = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    return a.fillna(b)


# ---------------------------------------------------------------------------
# Loại trùng
# ---------------------------------------------------------------------------
def char_shingles(texts, n: int = 5) -> sparse.csr_matrix:
    """Ma trận nhị phân shingle n-gram ký tự (chỉ dùng để loại trùng, không phải đặc trưng).

    texts: list (m,) str → csr (m, V), V = số shingle khác nhau trong chính danh sách này.
    """
    vocab: dict[str, int] = {}
    rows, cols = [], []
    for i, t in enumerate(texts):
        t = f" {t} "
        grams = {t[j:j + n] for j in range(max(len(t) - n + 1, 1))}
        for g in grams:
            cols.append(vocab.setdefault(g, len(vocab)))
            rows.append(i)
    data = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(texts), max(len(vocab), 1)))


def jaccard_matrix(X: sparse.csr_matrix) -> sparse.csr_matrix:
    """Jaccard |A∩B| / |A∪B| giữa mọi cặp hàng của ma trận nhị phân.

    X: csr (m, V) → csr (m, m), chỉ lưu các cặp có giao khác rỗng.
    """
    inter = (X @ X.T).tocoo()
    size = np.asarray(X.sum(axis=1)).ravel()
    union = size[inter.row] + size[inter.col] - inter.data
    jac = np.divide(inter.data, union, out=np.zeros_like(inter.data), where=union > 0)
    return sparse.csr_matrix((jac, (inter.row, inter.col)), shape=inter.shape)


def near_dup_groups(texts, threshold: float = 0.8, n: int = 5) -> np.ndarray:
    """Gom các tin gần trùng thành nhóm (thành phần liên thông của đồ thị Jaccard ≥ threshold).

    texts: list (m,) str → mảng (m,) int, mã nhóm 0..G-1 (tin không trùng ai có nhóm riêng).
    Dùng để split.py chia theo nhóm, tránh các tin cùng mẫu rơi vào cả train lẫn test.
    """
    from scipy.sparse.csgraph import connected_components
    if len(texts) == 0:
        return np.zeros(0, dtype=int)
    J = jaccard_matrix(char_shingles(list(texts), n=n))
    A = J.multiply(J >= threshold).tocsr()
    _, lab = connected_components(A, directed=False)
    return lab.astype(int)


def deduplicate(df: pd.DataFrame, text_col: str = "text_khongdau", date_col: str = "date",
                threshold: float | None = None, n: int = 5):
    """Loại trùng, giữ tin có date sớm nhất.

    threshold=None (mặc định): chỉ loại tin TRÙNG HẲN (text_col giống hệt sau chuẩn hóa).
    threshold=t: loại thêm tin gần trùng có Jaccard 5-gram ký tự với một tin ĐÃ GIỮ ≥ t.
    Duyệt tin theo (date tăng dần, NaT xếp cuối, rồi thứ tự gốc).
    df: DataFrame (m dòng) có cột text_col, date_col, 'id', 'label'.
    Trả về (kept, removed): kept (m', cột như df); removed (m − m') với thêm cột
    'kept_id', 'jaccard', 'same_label', 'exact'.
    """
    order = df.assign(_nat=df[date_col].isna(), _pos=np.arange(len(df))) \
              .sort_values(["_nat", date_col, "_pos"], kind="stable").index
    d = df.loc[order].reset_index(drop=True)
    keep = np.zeros(len(d), dtype=bool)
    rec = []
    if threshold is None:
        first: dict[str, int] = {}
        for i, t in enumerate(d[text_col].tolist()):
            if t in first:
                rec.append({"idx": i, "kept_idx": first[t], "jaccard": 1.0})
            else:
                first[t] = i
                keep[i] = True
        rng = []
    else:
        J = jaccard_matrix(char_shingles(d[text_col].tolist(), n=n))
        rng = range(len(d))
    for i in rng:
        row = J.getrow(i)
        js, vs = row.indices, row.data
        mask = (js < i) & keep[js] & (vs >= threshold)
        if mask.any():
            k = js[mask][np.argmax(vs[mask])]
            rec.append({"idx": i, "kept_idx": k, "jaccard": float(J[i, k])})
        else:
            keep[i] = True
    kept = d[keep].reset_index(drop=True)
    removed = d.loc[[r["idx"] for r in rec]].reset_index(drop=True)
    if rec:
        kid = [r["kept_idx"] for r in rec]
        removed["kept_id"] = d.loc[kid, "id"].to_numpy()
        removed["jaccard"] = [r["jaccard"] for r in rec]
        removed["same_label"] = removed["label"].to_numpy() == d.loc[kid, "label"].to_numpy()
        removed["exact"] = removed[text_col].to_numpy() == d.loc[kid, text_col].to_numpy()
    else:
        for c in ("kept_id", "jaccard", "same_label", "exact"):
            removed[c] = pd.Series(dtype=object)
    return kept, removed


# ---------------------------------------------------------------------------
# D3a: clean()
# ---------------------------------------------------------------------------
def load_inputs(labels_path, masked_path=None) -> tuple[pd.DataFrame, dict]:
    """Đọc labels_final.csv (+ messages_masked.csv nếu có) và ghép theo id.

    Lấy text trong labels_final nếu file có cột text (bản đã sửa tay mới hơn), ngược lại lấy từ
    messages_masked. Chỉ giữ tin reviewed = 1 khi có cột reviewed.
    Trả về (df (n,) cột id, text, label, date, nguon[, reviewed]; thống kê dict).
    """
    rd = dict(encoding="utf-8-sig", dtype=str, keep_default_na=False)
    lab = pd.read_csv(labels_path, **rd)
    info = {"n_labels": len(lab)}
    if masked_path is not None and Path(masked_path).exists():
        msk = pd.read_csv(masked_path, **rd).drop(columns=["label"], errors="ignore")
        df = lab.merge(msk, on="id", how="left", suffixes=("", "_masked"))
        info["n_missing_in_masked"] = int(df["text_masked"].isna().sum()) if "text_masked" in df else 0
        if "text" in lab.columns and "text_masked" in df:
            info["n_text_from_labels"] = int((df["text"] != df["text_masked"]).sum())
        elif "text_masked" in df:
            df["text"] = df["text_masked"]
        for c in ("date", "nguon"):
            if c not in lab.columns and f"{c}_masked" in df:
                df[c] = df[f"{c}_masked"]
        df = df.drop(columns=[c for c in df.columns if c.endswith("_masked")])
    else:
        df = lab
    if "reviewed" in df.columns:
        rv = df["reviewed"].fillna("0").astype(str).str.strip()
        info["n_not_reviewed"] = int((rv != "1").sum())
        df = df[rv == "1"]
    if "text" not in df.columns:
        raise ValueError("Không tìm thấy cột text trong labels_final hoặc messages_masked")
    bad = ~df["label"].isin(LABELS)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} nhãn không thuộc {LABELS}")
    return df.reset_index(drop=True), info


def clean(df: pd.DataFrame, threshold: float | None = None, group_threshold: float = 0.8,
          mark_suspicious_link: bool = True):
    """D3a: chuẩn hóa văn bản + loại trùng. Không học gì từ dữ liệu.

    Mặc định chỉ xóa tin trùng hẳn; tin gần trùng được GIỮ và gán chung mã 'nhom_gan_trung'
    (Jaccard ≥ group_threshold) để bước chia dữ liệu xếp cả nhóm vào cùng một tập.
    df: DataFrame (n dòng) có cột id, text, label (chuỗi trong core.LABELS), date, nguon.
    Trả về (clean_df, removed_df, stats):
      clean_df (n' dòng): id, date (YYYY-MM-DD hoặc rỗng), nguon, label, label_so (int: lua_dao=0,
                          rac=1, hop_le=2), nhom_gan_trung (int), text_clean, text_khongdau,
                          text_tok — sắp theo date;
      removed_df: các tin bị loại kèm kept_id, jaccard, same_label, exact (không có cột text);
      stats: dict số lượng (không chứa nội dung tin).
    """
    df = df.copy()
    proc = [process_text(t, mark_suspicious_link) for t in df["text"].tolist()]
    for c in ("text_clean", "text_khongdau", "text_tok"):
        df[c] = [p[c] for p in proc]
    df["date"] = parse_dates(df["date"]) if "date" in df else pd.NaT
    empty = df["text_khongdau"].str.strip() == ""
    df = df[~empty]
    kept, removed = deduplicate(df, threshold=threshold)
    kept["label_so"] = kept["label"].map({l: i for i, l in enumerate(LABELS)}).astype(int)
    kept["date"] = kept["date"].dt.strftime("%Y-%m-%d").fillna("")
    grp = near_dup_groups(kept["text_khongdau"].tolist(), threshold=group_threshold)
    kept["nhom_gan_trung"] = grp
    gsize = pd.Series(grp).map(pd.Series(grp).value_counts()).to_numpy()
    glab = kept.groupby("nhom_gan_trung")["label"].nunique()
    cols = ["id", "date", "nguon", "label", "label_so", "nhom_gan_trung",
            "text_clean", "text_khongdau", "text_tok"]
    kept = kept[[c for c in cols if c in kept.columns]]
    rcols = ["id", "label", "kept_id", "jaccard", "same_label", "exact"]
    removed = removed[[c for c in rcols if c in removed.columns]]
    stats = {
        "n_in": int(len(empty)), "n_empty": int(empty.sum()), "n_out": int(len(kept)),
        "n_removed_exact": int(removed["exact"].sum()) if len(removed) else 0,
        "n_removed_near": int((~removed["exact"].astype(bool)).sum()) if len(removed) else 0,
        "n_conflict_label": int((~removed["same_label"].astype(bool)).sum()) if len(removed) else 0,
        "n_no_date": int((kept["date"] == "").sum()),
        "threshold": "chi_trung_han" if threshold is None else threshold,
        "group_threshold": group_threshold,
        "n_groups": int(len(np.unique(grp))),
        "n_in_near_dup_group": int((gsize > 1).sum()),
        "n_group_mixed_label": int((glab > 1).sum()),
    }
    for l in LABELS:
        stats[f"in_{l}"] = int((df["label"] == l).sum())
        stats[f"removed_{l}"] = int((removed["label"] == l).sum())
        stats[f"out_{l}"] = int((kept["label"] == l).sum())
    return kept, removed, stats


# ---------------------------------------------------------------------------
# D3b: TF-IDF (chỉ fit trên train)
# ---------------------------------------------------------------------------
def fit_tfidf(train: pd.DataFrame, word_col: str = "text_tok", char_col: str = "text_khongdau",
              word_ngram=(1, 2), char_ngram=(2, 5), min_df: int = 2, max_features=None):
    """Fit TF-IDF n-gram từ và n-gram ký tự TRÊN TẬP TRAIN DUY NHẤT.

    train: DataFrame (n_train dòng) có word_col (văn bản đã tách từ) và char_col (văn bản chưa tách).
    Trả về (vec_word, vec_char): hai TfidfVectorizer đã fit (lưu thành models/tfidf_word.pkl,
    models/tfidf_char.pkl ở D4).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec_word = TfidfVectorizer(analyzer="word", tokenizer=str.split, token_pattern=None,
                               lowercase=False, ngram_range=tuple(word_ngram), min_df=min_df,
                               max_features=max_features, sublinear_tf=True)
    vec_char = TfidfVectorizer(analyzer="char_wb", lowercase=False, ngram_range=tuple(char_ngram),
                               min_df=min_df, max_features=max_features, sublinear_tf=True)
    vec_word.fit(train[word_col].fillna("").tolist())
    vec_char.fit(train[char_col].fillna("").tolist())
    return vec_word, vec_char


def transform_tfidf(vec_word, vec_char, df: pd.DataFrame, word_col: str = "text_tok",
                    char_col: str = "text_khongdau") -> sparse.csr_matrix:
    """Áp hai vectorizer đã fit. df (n dòng) → csr X (n, d_word + d_char)."""
    Xw = vec_word.transform(df[word_col].fillna("").tolist())
    Xc = vec_char.transform(df[char_col].fillna("").tolist())
    return sparse.hstack([Xw, Xc], format="csr")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="D3a: messages_masked + labels_final → clean.csv")
    p.add_argument("--labels", default=root / "data/labeled/labels_final.csv")
    p.add_argument("--masked", default=root / "data/masked/messages_masked.csv")
    p.add_argument("--out", default=root / "data/labeled/clean.csv")
    p.add_argument("--removed", default=root / "data/labeled/dedup_removed.csv")
    p.add_argument("--stats", default=root / "results/D3_thong_ke.csv")
    p.add_argument("--threshold", type=float, default=None,
                   help="bỏ trống = chỉ xóa trùng hẳn; đặt số (vd 0.8) để xóa cả gần trùng")
    p.add_argument("--group-threshold", type=float, default=0.8)
    a = p.parse_args(argv)

    df, info = load_inputs(a.labels, a.masked)
    kept, removed, stats = clean(df, threshold=a.threshold, group_threshold=a.group_threshold)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(a.out, index=False, encoding="utf-8-sig")
    removed.to_csv(a.removed, index=False, encoding="utf-8-sig")
    Path(a.stats).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{**info, **stats}]).T.rename(columns={0: "gia_tri"}) \
        .to_csv(a.stats, encoding="utf-8-sig", index_label="chi_so")
    # Chỉ in số lượng, không in nội dung tin nhắn.
    for k, v in {**info, **stats}.items():
        print(f"{k:>22}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
