"""[D4a] Kiểm tra data/splits/ ĐÃ CHIA (không chia lại, không in nội dung tin).

Cách chia (src/split.py): theo thời gian, PHÂN TẦNG THEO LỚP — trong từng lớp, 15% tin mới nhất
→ test, 7,5% kế trước → val_b, 7,5% kế trước nữa → val_a, còn lại → train; mỗi nhóm
nhom_gan_trung đi trọn vào một tập (xếp theo date sớm nhất của nhóm); tin không date chỉ ở train.

Mọi file chỉ đọc các cột id, label, date, nhom_gan_trung (usecols) — không đọc cột văn bản.
"""

import os
import sys

import pandas as pd
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from core import LABELS  # noqa: E402

SPLITS = ["train", "val_a", "val_b", "test"]
TARGET = {"train": 0.70, "val_a": 0.075, "val_b": 0.075, "test": 0.15}
COLS = ["id", "label", "date", "nhom_gan_trung"]
SPLIT_DIR = os.path.join(ROOT, "data", "splits")
CLEAN = os.path.join(ROOT, "data", "labeled", "clean.csv")

pytestmark = pytest.mark.skipif(
    not all(os.path.exists(os.path.join(SPLIT_DIR, f"{s}.csv")) for s in SPLITS),
    reason="chưa có data/splits/*.csv")


def _parse_dates(s):
    """M/D/YYYY (file Drive) hoặc YYYY-MM-DD; rỗng → NaT. s: Series (n,) → Series (n,)."""
    s = s.fillna("").astype(str).str.strip()
    a = pd.to_datetime(s, format="%m/%d/%Y", errors="coerce")
    b = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    return a.fillna(b)


@pytest.fixture(scope="module")
def df():
    """Ghép 4 tập (chỉ cột COLS) + cột split, _date, ngay_nhom (date sớm nhất của nhóm)."""
    parts = []
    for s in SPLITS:
        p = pd.read_csv(os.path.join(SPLIT_DIR, f"{s}.csv"), encoding="utf-8-sig", usecols=COLS)
        p["split"] = s
        parts.append(p)
    d = pd.concat(parts, ignore_index=True)
    d["_date"] = _parse_dates(d["date"])
    d["ngay_nhom"] = d.groupby("nhom_gan_trung")["_date"].transform("min")
    return d


def test_labels_valid(df):
    assert set(df["label"]) <= set(LABELS)
    for s in SPLITS:
        assert set(df.loc[df.split == s, "label"]) == set(LABELS), f"{s} thiếu lớp"


def test_disjoint(df):
    assert df["id"].notna().all()
    assert not df["id"].duplicated().any(), "có id nằm ở nhiều tập (hoặc lặp trong một tập)"


def test_union_equals_clean(df):
    if not os.path.exists(CLEAN):
        pytest.skip("chưa có data/labeled/clean.csv trên máy này")
    c = pd.read_csv(CLEAN, encoding="utf-8-sig", usecols=["id", "label"])
    assert len(c) == len(df)
    assert set(c["id"]) == set(df["id"])
    m = c.merge(df[["id", "label"]], on="id", suffixes=("_clean", "_split"))
    assert (m.label_clean == m.label_split).all(), "nhãn trong splits khác clean.csv"


def test_group_in_one_split(df):
    n_split = df.groupby("nhom_gan_trung")["split"].nunique()
    assert (n_split == 1).all(), f"{(n_split > 1).sum()} nhóm gần trùng nằm ở > 1 tập"


def test_undated_only_train(df):
    undated = df[df["_date"].isna()]
    assert (undated["split"] == "train").all()
    # date không parse được mà chuỗi khác rỗng → lỗi định dạng
    raw = df["date"].fillna("").astype(str).str.strip()
    assert (raw[df["_date"].isna()] == "").all(), "có date sai định dạng"


@pytest.mark.parametrize("lop", LABELS)
def test_time_order_per_class(df, lop):
    d = df[(df.label == lop) & df["ngay_nhom"].notna()]
    for a, b in zip(SPLITS[:-1], SPLITS[1:]):
        last_a = d.loc[d.split == a, "ngay_nhom"].max()
        first_b = d.loc[d.split == b, "ngay_nhom"].min()
        assert first_b >= last_a, f"{lop}: {b} bắt đầu {first_b.date()} < {a} kết thúc {last_a.date()}"


@pytest.mark.parametrize("lop", LABELS)
def test_ratio_per_class(df, lop):
    d = df[(df.label == lop) & df["_date"].notna()]
    frac = d["split"].value_counts(normalize=True)
    for s, t in TARGET.items():
        assert abs(frac.get(s, 0.0) - t) <= 0.02, f"{lop}/{s}: {frac.get(s, 0.0):.3f} vs {t}"
