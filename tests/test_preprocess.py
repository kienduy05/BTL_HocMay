"""Kiểm tra src/preprocess.py (D3) — toàn bộ dữ liệu là câu tự bịa, không dùng tin thật."""

import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import preprocess as pp  # noqa: E402


def clean_of(t, **kw):
    return pp.normalize_text(t, **kw)


# ---------------------------------------------------------------- Unicode, dấu thanh
def test_nfc_va_nfd_cho_cung_ket_qua():
    s = "Ngân hàng thông báo"
    assert clean_of(unicodedata.normalize("NFD", s)) == clean_of(s) == "ngân hàng thông báo"


@pytest.mark.parametrize("cu, moi", [("hoà", "hòa"), ("thuý", "thúy"), ("loè", "lòe"),
                                     ("hoạ", "họa"), ("HOÀ", "hòa")])
def test_dau_thanh_kieu_moi(cu, moi):
    assert clean_of(cu) == moi


@pytest.mark.parametrize("giu", ["hoàng", "khuyến", "quý", "quà", "khuỷu"])
def test_dau_thanh_khong_doi_khi_khong_can(giu):
    assert clean_of(giu) == giu


def test_chu_toan_hoc_va_ky_tu_an():
    assert clean_of("𝙉𝙝𝙖𝙣 qu​a") == "nhan qu a"


# ---------------------------------------------------------------- thẻ
def test_doi_the_mask_py():
    out = clean_of("Nhan [MONEY] luc [TIME] ma [OTP] cua [NAME]")
    assert out == "nhan [TIỀN] luc [TIME] ma [OTP] cua [NAME]"


@pytest.mark.parametrize("url", ["https://abc.example.com/x?y=1", "http:// abc.com.vn/",
                                 "www.abc.net", "abc.com.vn", "zalo.me/[NUMBER]", "x.y.me"])
def test_link(url):
    assert "[LINK]" in clean_of(f"vao {url} ngay").split()


def test_link_la_theo_duoi_va_ten_mien_gia():
    assert "[LINK_LẠ]" in clean_of("vao abcxyz.top ngay")
    assert "[LINK_LẠ]" in clean_of("vao vietcombank.vn-khac.com ngay")
    assert "[LINK_LẠ]" not in clean_of("vao viettel.vn/goi ngay")
    assert "[LINK_LẠ]" not in clean_of("vao abc.top", mark_suspicious_link=False)


@pytest.mark.parametrize("cau", ["hom nay.de biet them", "ban co.to lam", "1.5gb data",
                                 "tp.hcm hom nay"])
def test_khong_nham_link(cau):
    assert "[LINK]" not in clean_of(cau)


def test_sdt_va_tien_con_sot():
    out = clean_of("goi 0912 345 678 hoac +84912345678 nhan 100k, 2tr, 500.000đ")
    assert out.split().count("[SĐT]") == 2
    assert out.split().count("[TIỀN]") == 3


def test_khong_coi_ma_goi_cuoc_la_tien():
    out = clean_of("soan ST5K gui 191 duoc 5GB")
    assert "[TIỀN]" not in out and "st5k" in out and "5gb" in out


# ---------------------------------------------------------------- biến thể cố ý
def test_ky_tu_chen_giua_chu():
    assert clean_of("n.g.â.n h.à.n.g") == "ngân hàng"
    assert clean_of("T-D-T-C tang qua") == "tdtc tang qua"


def test_hai_chu_cai_khong_bi_ghep():
    assert clean_of("a/c lh") == "a c liên hệ"


@pytest.mark.parametrize("vao, ra", [("kh0a", "khoa"), ("th0ng ba0", "thong bao"),
                                     ("d!eu k!en", "dieu kien"), ("kh@ch", "khach")])
def test_chu_so_thay_chu(vao, ra):
    assert clean_of(vao) == ra


@pytest.mark.parametrize("vao, ra", [("V1etc0mbank", "vietcombank"), ("vlETCOMBANK", "vietcombank"),
                                     ("Sh0pee", "shopee"), ("M0M0", "momo"),
                                     ("[Agri_bank]", "agribank"), ("Zal0Pay", "zalopay"),
                                     ("Vjettel", "viettel")])
def test_gop_ten_thuong_hieu(vao, ra):
    assert clean_of(vao) == ra


def test_khong_doi_ma_so():
    assert clean_of("goi 5G10 xem tv360 ngay 12/3") == "goi 5g10 xem tv360 ngay 12 3"


def test_teencode_chi_doi_ca_tu():
    assert clean_of("k đc ck vào stk") == "không được chuyển khoản vào số tài khoản"
    assert clean_of("kho hàng") == "kho hàng"


def test_lap_ky_tu_va_chu_dinh():
    assert clean_of("haaaay quaaa") == "hay qua"
    assert clean_of("NhanNgayMotTramNghin") == "nhan ngay mot tram nghin"
    assert clean_of("ViettelPay") == "viettelpay"  # từ ngắn: không tách


def test_co_ky_tu_la():
    assert "[KÝ_TỰ_LẠ]" in clean_of("Li`xi~ph;on,g Ta:ng`Q;ua")
    assert "[KÝ_TỰ_LẠ]" not in clean_of("Xin chào, mình là An: cảm ơn")


# ---------------------------------------------------------------- tách từ, bỏ dấu
def test_process_text_ba_dang():
    p = pp.process_text("Chuyển khoản [MONEY] vào tài khoản")
    assert p["text_clean"] == "chuyển khoản [TIỀN] vào tài khoản"
    assert p["text_khongdau"] == "chuyen khoan [TIỀN] vao tai khoan"
    assert "chuyển_khoản" in p["text_tok"].split()
    assert "[TIỀN]" in p["text_tok"].split()


def test_the_dung_rieng_sau_tach_tu():
    tok = pp.process_text("Li`xi~ph;on,g Ta:ng`Q;ua hi")["text_tok"].split()
    assert "[KÝ_TỰ_LẠ]" in tok


def test_khong_dau_va_co_dau_gap_nhau():
    a = pp.process_text("Tài khoản bị khóa")["text_khongdau"]
    b = pp.process_text("Tai khoan bi khoa")["text_khongdau"]
    assert a == b


def test_dau_vao_rong():
    assert pp.process_text("") == {"text_clean": "", "text_khongdau": "", "text_tok": ""}
    assert pp.normalize_text(None) == ""


# ---------------------------------------------------------------- ngày, Jaccard
def test_parse_dates():
    d = pp.parse_dates(pd.Series(["5/28/2026", "2026-01-02", "", "abc"]))
    assert d.iloc[0] == pd.Timestamp("2026-05-28")
    assert d.iloc[1] == pd.Timestamp("2026-01-02")
    assert d.iloc[2:].isna().all()


def test_jaccard_khop_dinh_nghia():
    texts = ["abcdefgh", "abcdefgx", "zzzzzzzz"]
    J = pp.jaccard_matrix(pp.char_shingles(texts, n=3)).toarray()
    sh = [{f" {t} "[i:i + 3] for i in range(len(t) + 2 - 3 + 1)} for t in texts]
    for i in range(3):
        for j in range(3):
            exp = len(sh[i] & sh[j]) / len(sh[i] | sh[j])
            assert J[i, j] == pytest.approx(exp)
    assert J.shape == (3, 3) and np.allclose(np.diag(J), 1)


# ---------------------------------------------------------------- loại trùng, clean
def _df():
    return pd.DataFrame({
        "id": ["a", "b", "c", "d", "e"],
        "text": ["Ban da trung thuong [MONEY] tu chuong trinh tri an, lien he ngay de nhan qua",
                 "Bạn đã trúng thưởng [MONEY] từ chương trình tri ân, liên hệ ngay để nhận quà",
                 "Ban da trung thuong [MONEY] tu chuong trinh tri an, lien he ngay de nhan qua!",
                 "Mai đi học nhớ mang áo mưa nhé",
                 "Hóa đơn tiền điện tháng này là [MONEY], hạn thanh toán [DATE]"],
        "label": ["lua_dao", "lua_dao", "rac", "hop_le", "hop_le"],
        "date": ["3/2/2026", "1/15/2026", "", "2/1/2026", "2/2/2026"],
        "nguon": ["x"] * 5,
    })


def test_clean_giu_tin_som_nhat_va_bao_xung_dot():
    kept, removed, st = pp.clean(_df(), threshold=0.8)
    assert set(kept["id"]) == {"b", "d", "e"}          # b có ngày sớm nhất trong nhóm a,b,c
    assert set(removed["id"]) == {"a", "c"}
    assert (removed["kept_id"] == "b").all()
    assert st["n_conflict_label"] == 1                  # c (rac) trùng b (lua_dao)
    assert st["n_removed_exact"] == 2                   # sau chuẩn hóa không dấu là trùng hẳn
    assert st["out_lua_dao"] + st["out_rac"] + st["out_hop_le"] == st["n_out"] == 3
    assert list(kept["date"]) == sorted(kept["date"])   # sắp theo ngày
    assert kept["label_so"].tolist() == [pp.LABELS.index(l) for l in kept["label"]]
    assert "text" not in removed.columns                # không xuất nội dung tin bị loại


def test_tin_khong_ngay_bi_xep_sau():
    df = _df().iloc[[0, 2]].reset_index(drop=True)      # a có ngày, c không có
    kept, removed, _ = pp.clean(df)
    assert kept["id"].tolist() == ["a"] and removed["id"].tolist() == ["c"]


def test_nguong_1_chi_loai_trung_tuyet_doi():
    df = pd.DataFrame({"id": ["a", "b", "c"], "label": ["rac"] * 3, "nguon": ["x"] * 3,
                       "date": ["1/1/2026"] * 3,
                       "text": ["Khuyen mai lon thang nay", "khuyến mại lớn tháng này",
                                "Khuyen mai lon thang nay nhe"]})
    kept, removed, _ = pp.clean(df, threshold=1.0)
    assert kept["id"].tolist() == ["a", "c"]


def test_load_inputs(tmp_path):
    lab = tmp_path / "labels.csv"
    msk = tmp_path / "masked.csv"
    pd.DataFrame({"id": ["a", "b"], "label": ["rac", "hop_le"]}).to_csv(lab, index=False)
    pd.DataFrame({"id": ["a", "b"], "text": ["t1", "t2"], "date": ["1/1/2026", "1/2/2026"],
                  "nguon": ["x", "y"], "reviewed": ["1", "0"], "label": ["", ""]}) \
        .to_csv(msk, index=False)
    df, info = pp.load_inputs(lab, msk)
    assert df["id"].tolist() == ["a"] and df["text"].tolist() == ["t1"]
    assert info["n_not_reviewed"] == 1


def test_load_inputs_nhan_sai(tmp_path):
    lab = tmp_path / "labels.csv"
    pd.DataFrame({"id": ["a"], "text": ["t"], "label": ["spam"], "date": [""], "nguon": ["x"]}) \
        .to_csv(lab, index=False)
    with pytest.raises(ValueError):
        pp.load_inputs(lab)


def test_main_ghi_file_va_khong_in_noi_dung(tmp_path, capsys):
    lab = tmp_path / "labels_final.csv"
    _df().to_csv(lab, index=False)
    out, rem, st = tmp_path / "clean.csv", tmp_path / "rm.csv", tmp_path / "st.csv"
    assert pp.main(["--labels", str(lab), "--masked", str(tmp_path / "khong_co.csv"),
                    "--out", str(out), "--removed", str(rem), "--stats", str(st)]) == 0
    assert len(pd.read_csv(out)) == 3 and rem.exists() and st.exists()
    printed = capsys.readouterr().out
    assert "trúng thưởng" not in printed and "trung thuong" not in printed


# ---------------------------------------------------------------- TF-IDF
def test_fit_tfidf_chi_hoc_tu_train():
    tr = pd.DataFrame([pp.process_text(t) for t in
                       ["tài khoản bị khóa vào link", "tài khoản bị khóa ngay", "mai đi học nhé",
                        "mai đi học sớm"]])
    te = pd.DataFrame([pp.process_text("xổ số độc đắc tuyệt vời")])
    vw, vc = pp.fit_tfidf(tr, min_df=1)
    vocab_w = set(vw.vocabulary_)
    assert "xổ_số" not in vocab_w and "độc_đắc" not in vocab_w
    assert all(g in " ".join(tr["text_khongdau"]) or " " in g for g in vc.vocabulary_)
    X = pp.transform_tfidf(vw, vc, te)
    assert X.shape == (1, len(vw.vocabulary_) + len(vc.vocabulary_))
    Xtr = pp.transform_tfidf(vw, vc, tr)
    assert Xtr.shape[0] == 4 and np.isfinite(Xtr.data).all()


def test_fit_tfidf_giu_the():
    tr = pd.DataFrame([pp.process_text(t) for t in ["nhận [MONEY] tại abc.top", "nhận [MONEY] ngay"]])
    vw, _ = pp.fit_tfidf(tr, min_df=1)
    assert "[TIỀN]" in vw.vocabulary_ and "[LINK]" in vw.vocabulary_


# ---------------------------------------------------------------- mặc định: chỉ xóa trùng hẳn
def test_mac_dinh_chi_xoa_trung_han_va_giu_gan_trung():
    df = pd.DataFrame({"id": ["a", "b", "c", "d"], "label": ["rac", "rac", "rac", "hop_le"],
                       "nguon": ["x"] * 4, "date": ["1/2/2026", "1/1/2026", "1/3/2026", "1/4/2026"],
                       "text": ["Khuyen mai lon thang nay cho ban",
                                "Khuyến mại lớn tháng này cho bạn!",      # trùng hẳn sau chuẩn hóa
                                "Khuyen mai lon thang nay cho ban nhe",   # chỉ gần trùng → giữ
                                "Mai di hoc nho mang ao mua"]})
    kept, removed, st = pp.clean(df)
    assert set(kept["id"]) == {"b", "c", "d"}            # b sớm nhất trong cặp trùng hẳn a–b
    assert removed["id"].tolist() == ["a"] and removed["exact"].all()
    g = dict(zip(kept["id"], kept["nhom_gan_trung"]))
    assert g["b"] == g["c"] != g["d"]                    # b, c cùng nhóm gần trùng
    assert st["n_in_near_dup_group"] == 2 and st["n_removed_near"] == 0


def test_near_dup_groups():
    g = pp.near_dup_groups(["abcdefghij", "abcdefghix", "zzzzzzzzzz"], threshold=0.5)
    assert g.shape == (3,) and g[0] == g[1] != g[2]
    assert pp.near_dup_groups([]).shape == (0,)
