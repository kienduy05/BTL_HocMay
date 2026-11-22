"""[D4] Chia clean.csv thành train / val_a / val_b / test theo thời gian (70/7,5/7,5/15),
PHÂN TẦNG THEO LỚP: trong từng lớp, 15% tin mới nhất -> test, 7,5% kế trước -> val_b,
7,5% kế trước nữa -> val_a, phần cũ hơn -> train.
  Lý do: date là ngày thu thập theo đợt; tin lừa đảo dồn vào 21-25/7, nên chia theo thời gian
  trên toàn bộ dữ liệu làm val_a, val_b không có tin lừa đảo nào và test chỉ có 5.
- Chỉ tin có date được xếp theo thời gian; mỗi nhóm gần trùng (Jaccard 5-gram ký tự >= 0,8)
  đi trọn vào một tập, xếp theo date sớm nhất trong nhóm.
- Tin không có ngày (SYN, tong_hop) chỉ vào train -> val_a, val_b, test chỉ gồm tin thật.
- Không in nội dung tin."""
import sys, numpy as np, pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

src = sys.argv[1] if len(sys.argv) > 1 else "data/labeled/clean.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "data/splits"
T = 0.8
d = pd.read_csv(src, encoding="utf-8-sig")
d["_date"] = pd.to_datetime(d["date"], format="%m/%d/%Y", errors="coerce")

# Nhóm gần trùng
X = CountVectorizer(analyzer="char", ngram_range=(5, 5), binary=True, lowercase=False).fit_transform(d.text_khongdau.fillna(""))
I = (X @ X.T).tocoo(); sz = np.asarray(X.sum(1)).ravel()
J = I.data / (sz[I.row] + sz[I.col] - I.data)
k = (J >= T) & (I.row != I.col)
A = coo_matrix((np.ones(k.sum()), (I.row[k], I.col[k])), shape=(len(d), len(d)))
_, lab = connected_components(A, directed=False)
d["nhom_gan_trung"] = lab

# Xếp nhóm theo date sớm nhất; nhóm không có ngày nào -> train
assert (d.groupby("nhom_gan_trung").label.nunique() == 1).all(), "nhóm gần trùng lẫn nhãn"
g = d.groupby("nhom_gan_trung").agg(ngay=("_date", "min"), n=("id", "size"), id0=("id", "min"), label=("label", "first"))
moc = [0.70, 0.775, 0.85]
phan = {}
for lop, gl in g[g.ngay.notna()].groupby("label"):
    gl = gl.sort_values(["ngay", "id0"])
    N = gl.n.sum(); cum = gl.n.cumsum() - gl.n           # số tin của lớp đã xếp trước nhóm
    phan.update(dict(zip(gl.index, np.select([cum < moc[0]*N, cum < moc[1]*N, cum < moc[2]*N],
                                             ["train", "val_a", "val_b"], "test"))))
d["split"] = d.nhom_gan_trung.map(phan).fillna("train")   # nhóm không có ngày -> train

cols = [c for c in d.columns if c not in ("_date", "split")]
for s in ["train", "val_a", "val_b", "test"]:
    p = d[d.split == s].sort_values(["_date", "id"], na_position="last")
    p[cols].to_csv(f"{out}/{s}.csv", index=False, encoding="utf-8-sig")

# Thống kê (không in nội dung tin)
print(pd.crosstab(d.split, d.label, margins=True).reindex(["train", "val_a", "val_b", "test", "All"]))
print(d.groupby(["label", "split"])._date.agg(["min", "max"]))
print("Tin không ngày (chỉ train):", d._date.isna().sum(), "| nhóm gần trùng >=2 tin:", (g.n >= 2).sum())
print("Nhóm nằm ở >1 tập:", (d.groupby("nhom_gan_trung").split.nunique() > 1).sum())
