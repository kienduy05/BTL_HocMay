# Đề tài 2: Hồi quy logistic, GLM và hiệu chuẩn xác suất
**Ứng dụng:** Phát hiện tin nhắn lừa đảo tiếng Việt (3 lớp: lừa đảo, quảng cáo rác, hợp lệ)

> [!IMPORTANT]
> **Hướng dẫn thiết lập khi clone repo lần đầu:**
> ```bash
> # 1. Tạo môi trường ảo .venv
> python -m venv .venv
>
> # 2. Kích hoạt môi trường ảo
> .\.venv\Scripts\activate      # Windows (PowerShell / CMD)
> # source .venv/bin/activate   # macOS / Linux
>
> # 3. Cài đặt các thư viện
> pip install -r requirements.txt
>
> # 4. Kiểm tra môi trường (phải in ra: 1+1=2)
> python src/core.py
> ```

## 1. Thành viên

| Thành viên | Vai trò |
|---|---|
| Kiên | Nhóm trưởng, tạo repo, duyệt Pull Request, ghép báo cáo |
| Đức | Thành viên |
| Nhật | Thành viên |

## 2. Phân công

| Phần | Kiên (làm trước) | Đức | Nhật |
|---|---|---|---|
| **A – Cài đặt** | A2: IRLS nhị phân, softmax (có L2) + Armijo, đối chiếu L-BFGS, log-sum-exp | A1: dẫn công thức gradient/Hessian; A3: khung GLM | A4: Platt, PAVA, temperature scaling, ECE, biểu đồ độ tin cậy |
| **B – Chứng minh** | B1, B2 | B3, B4 | B5, B6 |
| **C – Bài tập** | C1, C3 | C2 | C4 (chứng minh + kiểm chứng), C5 |
| **D – Ứng dụng** | D1: chốt cột dữ liệu, `mask.py`, mẫu văn bản đồng ý, rà soát thủ công lô của Nhật; D2: phân xử + viết quy trình phân xử; D4: chia dữ liệu, huấn luyện, hiệu chuẩn, ma trận chi phí + biện luận | D1: rà soát thủ công lô của Kiên; D2: hướng dẫn gán nhãn + gán toàn bộ; D3: tiền xử lý | D1: rà soát thủ công lô của Đức; D2: gán chéo 20% + Cohen's κ; D5: đánh giá |
| **Báo cáo** | A2, B1–B2, C1, C3, D1, D4; ghép, hoàn thiện | A1, A3, B3–B4, C2, D2 (hướng dẫn), D3 | A4, B5–B6, C4, C5, D2 (κ), D5 |

Cả nhóm cùng thu thập tin nhắn ở D1.

## 3. Cấu trúc thư mục

```
de-tai-2/
├── README.md
├── RULES.md             # quy tắc cho AI + quy ước dữ liệu (nhãn, shape, cột CSV, seed)              (Kiên)
├── PROGRESS.md          # tiến độ, việc bị chặn, quyết định đã chốt — cập nhật mỗi khi gộp PR        (Kiên)
├── CLAUDE.md            # chỉ dẫn cho Claude Code: tự nạp RULES.md, PROGRESS.md;
│                        # bảng chỉ mục việc → mục README cần đọc                                    (Kiên)
├── requirements.txt     # thư viện + phiên bản, cả nhóm cài giống nhau (có pyvi hoặc underthesea)  (Kiên)
├── .gitignore           # bỏ qua .venv/ và nội dung data/, models/, consent/ (giữ thư mục qua .gitkeep)  (Kiên)
│
├── src/
│   ├── core.py                 # [A2] hằng số LABELS, sigmoid ổn định, log-sum-exp (dùng chung cho mọi file)  (Kiên)
│   ├── logistic.py             # [A2] Newton/IRLS nhị phân — dùng cho C1, C3                          (Kiên)
│   ├── softmax.py              # [A2] softmax có/không L2 + hạ gradient + Armijo, đối chiếu L-BFGS — dùng cho D4  (Kiên)
│   ├── glm.py                  # [A3] IRLS chung cho Bernoulli, Poisson, Gauss                        (Đức)
│   ├── calibration.py          # [A4] Platt scaling, PAVA, temperature scaling — dùng cho D4          (Nhật)
│   ├── calibration_metrics.py  # [A4] ECE (nhị phân, top-label, theo lớp), biểu đồ độ tin cậy — dùng cho C2, D5  (Nhật)
│   ├── scoring.py              # [C2] điểm Brier, phân rã Murphy (độ tin cậy, độ phân giải, độ bất định)  (Đức)
│   ├── mask.py                 # [D1] đọc file trong data/raw/, che SĐT, số tài khoản, họ tên, mã giao dịch,
│   │                           #      thêm dòng vào data/masked/messages_masked.csv                 (Kiên)
│   ├── preprocess.py           # [D3] clean(): NFC, chuẩn hóa dấu thanh, viết tắt, biến thể cố ý,
│   │                           #      thay thẻ, tách từ, loại trùng và gần trùng (chạy TRƯỚC khi chia)
│   │                           #      fit_tfidf(): TF-IDF n-gram từ + ký tự, chỉ học trên train   (Đức)
│   ├── split.py                # [D4] chia theo thời gian 70/15/15, tập val tách đôi val_a/val_b    (Kiên)
│   └── evaluate.py             # [D5] F1 macro, recall lừa đảo tại FPR cố định, quyết định theo chi phí,
│                               #      điều chỉnh tỷ lệ lớp (B6)                                     (Nhật)
│
├── tests/
│   ├── test_core.py                 # kiểm tra core.py                 (Kiên)
│   ├── test_logistic.py             # kiểm tra logistic.py             (Kiên)
│   ├── test_softmax.py              # kiểm tra softmax.py              (Kiên)
│   ├── test_mask.py                 # kiểm tra mask.py — nhiều dạng SĐT/STK/tên  (Kiên)
│   ├── test_split.py                # kiểm tra split.py — không chồng lấn, đúng thứ tự thời gian  (Kiên)
│   ├── test_glm.py                  # kiểm tra glm.py                  (Đức)
│   ├── test_scoring.py              # kiểm tra scoring.py              (Đức)
│   ├── test_preprocess.py           # kiểm tra preprocess.py           (Đức)
│   ├── test_calibration.py          # kiểm tra calibration.py          (Nhật)
│   ├── test_calibration_metrics.py  # kiểm tra calibration_metrics.py  (Nhật)
│   └── test_evaluate.py             # kiểm tra evaluate.py             (Nhật)
│
├── notebooks/
│   ├── C1_newton_tay.ipynb          # [C1] kiểm chứng số liệu tính tay        (Kiên)
│   ├── C2_ece_brier.ipynb           # [C2] dùng scoring.py + calibration_metrics.py  (Đức)
│   ├── C3_kha_tach.ipynb            # [C3]                                    (Kiên)
│   ├── C4_trong_so_lop.ipynb        # [C4] kiểm chứng số học                   (Nhật)
│   ├── C5_roc_pr.ipynb              # [C5] kèm phần giải thích                 (Nhật)
│   ├── D1_thong_ke.ipynb            # [D1] thống kê lớp × nguồn × tháng        (Kiên)
│   ├── D2_cohen_kappa.ipynb         # [D2] chọn mẫu 20%, tính κ                (Nhật)
│   ├── D4_huan_luyen.ipynb          # [D4]                                    (Kiên)
│   └── D5_danh_gia.ipynb            # [D5]                                    (Nhật)
│
├── proofs/
│   ├── A1_dan_cong_thuc.md          # [A1] (Đức)
│   ├── B1_B2.md                     # [B1, B2] (Kiên)
│   ├── B3_B4.md                     # [B3, B4] (Đức)
│   ├── B5_B6.md                     # [B5, B6] + mở rộng cho 3 lớp (Nhật)
│   └── C4.md                        # [C4] chứng minh dịch logit log(a1/a0) (Nhật)
│
├── docs/
│   ├── mau_dong_y.md                # [D1] mẫu văn bản đồng ý cho tình nguyện viên (Kiên)
│   ├── huong_dan_gan_nhan.md        # [D2] định nghĩa + ví dụ cho từng lớp (Đức)
│   ├── quy_trinh_phan_xu.md         # [D2] quy trình phân xử tin lệch nhãn (Kiên)
│   └── ma_tran_chi_phi.md           # [D4] ma trận chi phí 3×3 + biện luận (Kiên, Nhật duyệt)
│
├── consent/                         # KHÔNG đẩy lên Git — văn bản đồng ý đã ký, chỉ Kiên giữ
│
├── data/                            # KHÔNG đẩy lên Git
│   ├── raw/                         # [D1] tin gốc CHƯA che, mỗi lô một file — chỉ trên máy Kiên
│   │   └── <tên>_<ngày>.csv         #      cột: text, date, source_type, source_ref (xem RULES.md)
│   │                                #      xóa sau khi lô đã che + rà soát xong
│   ├── masked/
│   │   └── messages_masked.csv      # [D1] đầu ra của mask.py — đã che, Kiên đưa lên Drive (xem mục 4)
│   ├── labeled/
│   │   ├── crosscheck_ids.csv       # [D2] id của 20% tin gán chéo (chọn ngẫu nhiên phân tầng)
│   │   ├── labels_duc.csv           # [D2] nhãn của Đức (toàn bộ)
│   │   ├── labels_nhat.csv          # [D2] nhãn của Nhật (20% gán chéo)
│   │   ├── labels_final.csv         # [D2] nhãn cuối sau phân xử
│   │   └── clean.csv                # [D3] đầu ra của clean()
│   └── splits/
│       ├── train.csv                # [D4] 70% cũ nhất
│       ├── val_a.csv                # [D4] 7,5% tiếp theo — fit bộ hiệu chuẩn
│       ├── val_b.csv                # [D4] 7,5% tiếp theo — chọn bộ hiệu chuẩn, chọn ngưỡng
│       └── test.csv                 # [D4] 15% mới nhất
│
├── models/                          # [D4] KHÔNG đẩy lên Git — D5 đọc lại từ đây
│   ├── tfidf_word.pkl               # TF-IDF n-gram từ (học trên train)
│   ├── tfidf_char.pkl               # TF-IDF n-gram ký tự (học trên train)
│   ├── softmax.npz                  # trọng số W, b của softmax tự viết (A2)
│   ├── naive_bayes.pkl              # Naive Bayes đa thức (sklearn)
│   ├── svm.pkl                      # SVM tuyến tính (sklearn)
│   ├── calibrators.pkl              # bộ hiệu chuẩn tốt nhất cho từng mô hình (A4)
│   └── threshold.json               # ma trận chi phí, ngưỡng tại FPR cố định, tỷ lệ lớp train π
│
├── results/                         # bảng, biểu đồ đưa vào báo cáo
│   ├── C1_newton.md                 # [C1] kết quả 2 bước Newton
│   ├── C2_ece_brier.csv             # [C2] ECE, Brier, 3 thành phần Murphy, độ lệch đẳng thức
│   ├── C3_norm_w.png                # [C3] quỹ đạo ‖w‖ theo vòng lặp
│   ├── C4_logit_shift.csv           # [C4] độ dịch logit theo trọng số lớp
│   ├── C5_roc_pr.png                # [C5] ROC và PR theo tỷ lệ lớp
│   ├── D1_thong_ke.csv              # [D1] phân bố lớp × nguồn × tập dữ liệu
│   ├── D2_kappa.txt                 # [D2] Cohen's κ (vòng thử + chính thức)
│   ├── D5_metrics.csv               # [D5] F1 macro, recall lừa đảo @FPR, ECE, log-loss của 3 mô hình
│   ├── D5_confusion_matrix.png      # [D5] ma trận nhầm lẫn
│   ├── D5_reliability.png           # [D5] biểu đồ độ tin cậy trước/sau hiệu chuẩn
│   ├── D5_drift.csv                 # [D5] so sánh kết quả val_b và test
│   ├── D5_prior_shift.csv           # [D5] kết quả trước/sau điều chỉnh tỷ lệ lớp theo các kịch bản π′
│   ├── D5_top_features.csv          # [D5] đặc trưng có trọng số lớn
│   └── D5_errors.csv                # [D5] tin bị đoán sai (đã che) để phân tích
│
└── report/
    └── bao_cao.pdf                  # báo cáo cuối (Kiên ghép)
```

Mỗi thư mục có sẵn một file rỗng `.gitkeep` để Git giữ lại thư mục khi còn trống. Với `data/`, `models/`, `consent/`, Git chỉ lưu `.gitkeep`, mọi file khác bên trong đều bị `.gitignore` bỏ qua.

## 4. Quy ước chung

> Kiên chốt quy ước khi tạo repo, **trước khi cả nhóm nhập lô tin đầu tiên**.

**Quy ước dữ liệu** (nhãn, shape, ma trận chi phí, cột file raw, cột `messages_masked.csv`, khung hàm mẫu, seed): xem `RULES.md`, mục "Quy ước dữ liệu". Muốn đổi quy ước thì sửa `RULES.md` và ghi vào `PROGRESS.md`.

**Dữ liệu dùng chung:** thư mục Google Drive riêng của nhóm (chỉ 3 thành viên), chứa `data/masked/`, `data/labeled/`, `data/splits/`, `models/`.

## 5. Quy trình làm việc với Git

1. Kiên tạo repo và khung code, gồm tên hàm và mô tả đầu vào/đầu ra.
2. Mỗi người làm trên nhánh riêng: `feature/<tên-việc>`, ví dụ `feature/glm`.
3. Xong việc thì tạo Pull Request; Kiên duyệt rồi gộp vào `main`.
4. Không ai đẩy thẳng lên `main`.
5. Sau khi tạo repo, Kiên tạo thử một file trong `data/raw/` và chạy `git status` để chắc chắn file đó không hiện ra.
6. Gộp PR xong thì cập nhật `PROGRESS.md`; mọi quyết định thay đổi so với kế hoạch ghi vào mục "Quyết định đã chốt".
7. Nhóm dùng Claude Code: `CLAUDE.md` tự nạp `RULES.md` và `PROGRESS.md` mỗi phiên. Khi giao việc, nói rõ đang làm phần nào (ví dụ "làm preprocess.py") để Claude đọc đúng mục README. Phiên dài thì gõ `/memory` để kiểm tra các file đã được nạp.

## 6. Quy trình phần D (theo thứ tự)

```
D1 Thu thập (mục tiêu 3.500–4.000 tin để sau lọc còn ≥ 3.000, đủ 3 lớp)
   → gửi file lô cho Kiên → Kiên lưu vào data/raw/ → che bằng mask.py
   → đưa messages_masked.csv lên Drive → rà soát thủ công → Kiên xóa file raw
 ↓
D2 Vòng thử: Đức + Nhật cùng gán 100 tin → tính κ → sửa hướng dẫn nếu cần
   Vòng chính: Đức gán toàn bộ; Nhật gán độc lập 20% (ngẫu nhiên phân tầng); tính κ; Kiên phân xử
 ↓
D3a Làm sạch: NFC, dấu thanh, viết tắt, biến thể cố ý, thẻ [LINK]/[TIỀN]/[SĐT], tách từ,
    loại trùng và gần trùng (giữ bản cũ nhất)
 ↓
D4a Chia theo thời gian: train 70% | val_a 7,5% | val_b 7,5% | test 15%
 ↓
D3b TF-IDF n-gram từ + n-gram ký tự: chỉ học trên train, rồi áp sang val/test
 ↓
D4b Huấn luyện softmax (A2), Naive Bayes, SVM — siêu tham số chọn bằng kiểm định chéo
    theo thời gian TRONG train
    Hiệu chuẩn: fit Platt / PAVA / temperature trên val_a; chọn bộ tốt nhất trên val_b
    (SVM bắt buộc qua hiệu chuẩn mới có xác suất)
    Quyết định: ma trận chi phí 3×3; ngưỡng tại FPR cố định chốt trên val_b
 ↓
D5 Đánh giá trên tập test (chỉ chạy MỘT lần cuối)
```

## 7. Đặc tả từng bước phần D

### D1 – Thu thập và che thông tin
- Nguồn: bản tin cảnh báo công khai của cơ quan chức năng, báo chí, tin do tình nguyện viên đóng góp **có văn bản đồng ý** (theo `docs/mau_dong_y.md`, bản đã ký lưu ở `consent/`).
- Mục tiêu 3.500–4.000 tin, vì bước loại trùng và gần trùng ở D3 sẽ làm giảm số lượng; sau lọc phải còn ≥ 3.000 tin và đủ cả 3 lớp.
- Người thu thập nhập tin vào `<tên>_<ngày>.csv` theo đúng các cột trong `RULES.md` (mục "Quy ước dữ liệu"), gửi trực tiếp cho Kiên (chuyển file 1–1, không qua nhóm chat, không qua công cụ AI), rồi xóa bản trên máy mình. Kiên là người duy nhất giữ `data/raw/` và chạy `mask.py`.
- Kiên chạy `mask.py` để che SĐT, số tài khoản/số thẻ, họ tên, mã giao dịch/OTP; kết quả được thêm vào `messages_masked.csv` với `reviewed = 0`, rồi Kiên đưa bản mới lên Drive.
- Rà soát thủ công trên bản đã che ở Drive: người thu thập không tự rà soát lô của mình (Kiên ↔ Nhật ↔ Đức xoay vòng). Phát hiện sót thì che tay, đặt `reviewed = 1`, và bổ sung mẫu vào `test_mask.py`. Rà soát xong thì Kiên **xóa file raw** của lô đó.
- **Không truy cập đường dẫn trong tin nhắn.**
- `D1_thong_ke.ipynb` báo cáo phân bố lớp × nguồn × tháng, để phát hiện sớm việc một nguồn chiếm hết một giai đoạn thời gian.

### D2 – Gán nhãn
- `huong_dan_gan_nhan.md`: định nghĩa, ít nhất 5 ví dụ và các trường hợp ranh giới (lừa đảo ↔ rác, rác ↔ hợp lệ) cho từng lớp.
- 20% gán chéo chọn ngẫu nhiên phân tầng theo nguồn và tháng; Nhật gán **không xem** nhãn của Đức.
- Ngưỡng chấp nhận: κ ≥ 0,6; thấp hơn thì sửa hướng dẫn và gán lại phần lệch.
- `quy_trinh_phan_xu.md`: ai phân xử, căn cứ nào, ghi lại lý do cho từng tin lệch, có cập nhật hướng dẫn sau phân xử không. Báo cáo cả κ, số tin lệch và kết quả phân xử.

### D3 – Tiền xử lý
- Chuẩn hóa Unicode NFC; thống nhất vị trí dấu thanh (ví dụ *hoà* → *hòa*, *thuý* → *thúy*).
- Biến thể chính tả cố ý: ký tự chèn giữa chữ (*n.g.â.n h.à.n.g*), chữ số thay chữ (*V1etc0mbank*), teencode, viết không dấu — xử lý bằng quy tắc và từ điển cố định (không học từ dữ liệu).
- Thẻ đặc trưng: `[LINK]`, `[TIỀN]`, `[SĐT]`, thống nhất với thẻ do `mask.py` sinh ra.
- Tách từ (pyvi hoặc underthesea) cho n-gram từ; n-gram ký tự chạy trên văn bản chưa tách.
- Loại trùng tuyệt đối và gần trùng (ví dụ Jaccard trên n-gram ký tự), giữ tin có `date` sớm nhất.

### D4 – Mô hình hóa
- Ba mô hình: softmax tự viết (A2, có L2), Naive Bayes đa thức, SVM tuyến tính.
- Hiệu chuẩn: thử cả ba phương pháp của A4 cho từng mô hình; tiêu chí chọn là log-loss/ECE trên val_b.
- Ma trận chi phí 3×3 `C[lớp thật, lớp đoán]` do nhóm đề xuất, biện luận trong `ma_tran_chi_phi.md` (ví dụ: bỏ lọt tin lừa đảo đắt nhất; chặn nhầm tin hợp lệ đắt hơn để lọt tin rác). Quy tắc quyết định: chọn lớp có chi phí kỳ vọng `Σᵢ C[i, j]·p(i|x)` nhỏ nhất; B5 là trường hợp riêng 2 lớp.
- Ghi vào `threshold.json`: ma trận chi phí, ngưỡng điểm lừa đảo tương ứng FPR = 1% trên val_b, tỷ lệ lớp π của train.

### D5 – Đánh giá
- Tập test là 15% tin mới nhất. So sánh kết quả val_b với test để đo trôi khái niệm (`D5_drift.csv`).
- Chỉ số cho cả 3 mô hình, trước và sau hiệu chuẩn:
  - F1 macro (theo quy tắc chi phí tối thiểu);
  - recall lớp lừa đảo tại **FPR = 1%** (lừa đảo so với phần còn lại; FPR = tỷ lệ tin không lừa đảo bị báo là lừa đảo), dùng ngưỡng chốt trên val_b, báo cả FPR thực tế đạt được trên test;
  - ECE top-label (M = 15) và ECE riêng lớp lừa đảo; log-loss.
- Điều chỉnh tỷ lệ lớp (B6, dạng đa lớp): `p′(y|x) ∝ p(y|x)·π′ᵧ/πᵧ`. Kịch bản: tin lừa đảo chiếm 5% và 1% lúc triển khai; mô phỏng bằng lấy mẫu lại tập test theo π′ nhiều lần; so sánh ECE, log-loss, chi phí kỳ vọng trước/sau điều chỉnh.
- Đặc trưng có trọng số lớn: trọng số softmax, hệ số SVM, tỷ số log-xác suất của Naive Bayes, cho cả n-gram từ và ký tự.
- Phân tích lỗi: nhóm các lỗi điển hình (lừa đảo giả dạng ngân hàng, rác bị nhầm lừa đảo…), báo cáo theo lớp × nguồn.

## 8. Cách chạy

### Bước 0 – Cài đặt và kiểm tra code
```
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
python src/core.py              # phải in ra: 1+1=2
pytest tests/
```
Phải pass hết thì mới dùng được cho các bước sau.

### Bước 1 – Phần C
Chạy `notebooks/C1` → `C5`. Các notebook độc lập, chạy cái nào trước cũng được. Kết quả lưu vào `results/C*`.

### Bước 2 – Phần D (bắt buộc đúng thứ tự)

| # | Chạy | Đầu vào → Đầu ra |
|---|---|---|
| 1 | Người thu thập gửi file lô cho Kiên; Kiên chạy `python src/mask.py <file> --collector <tên>` | file raw → thêm dòng vào `data/masked/messages_masked.csv` (`reviewed = 0`); Kiên đưa bản mới lên Drive |
| 2 | Rà soát thủ công trên Drive (xoay vòng) | `messages_masked.csv` → `reviewed = 1`; sau đó Kiên xóa file raw của lô đó |
| 3 | `notebooks/D1_thong_ke.ipynb` | `messages_masked.csv` → `results/D1_thong_ke.csv` |
| 4 | `notebooks/D2_cohen_kappa.ipynb` (phần chọn mẫu) | `messages_masked.csv` → `data/labeled/crosscheck_ids.csv` |
| 5 | Gán nhãn thủ công (vòng thử, rồi vòng chính) | → `labels_duc.csv`, `labels_nhat.csv` |
| 6 | `notebooks/D2_cohen_kappa.ipynb` (phần κ) | 2 file nhãn → `results/D2_kappa.txt`; sau phân xử → `labels_final.csv` |
| 7 | `python src/preprocess.py` | `messages_masked.csv` + `labels_final.csv` → `data/labeled/clean.csv` |
| 8 | `python src/split.py` | `clean.csv` → `train.csv`, `val_a.csv`, `val_b.csv`, `test.csv` |
| 9 | `notebooks/D4_huan_luyen.ipynb` | `train.csv`, `val_a.csv`, `val_b.csv` → toàn bộ file trong `models/` |
| 10 | `notebooks/D5_danh_gia.ipynb` | `models/` + `val_b.csv` + `test.csv` → các file `results/D5_*` |

## 9. Quy tắc bắt buộc

- **Tin chưa che chỉ nằm trong `data/raw/` trên máy Kiên.** Người thu thập gửi trực tiếp cho Kiên rồi xóa bản của mình; không đẩy lên Git, không đưa lên Drive, không dán vào công cụ AI (kể cả để Claude Code mở file); Kiên xóa sau khi lô đã che và rà soát xong.
- **Không đẩy nội dung `data/`, `models/`, `consent/` lên Git** (chỉ `.gitkeep`). Dữ liệu dùng chung qua thư mục Drive riêng của nhóm; `data/raw/` và `consent/` không bao giờ đưa lên Drive.
- **Không truy cập đường dẫn trong tin nhắn.**
- **Chỉ chia dữ liệu một lần.** Cả nhóm dùng chung `data/splits/`, không ai tự chia lại.
- **Tập test chỉ dùng ở bước cuối.** Siêu tham số chọn trong train; hiệu chuẩn fit trên val_a; chọn bộ hiệu chuẩn và ngưỡng trên val_b.
- **Bước nào phải "học" từ dữ liệu** (TF-IDF, chọn từ vựng…) thì chỉ được học từ tập train.
- **Code viết cùng AI** phải tuân thủ `RULES.md`.

## 10. Lịch thực hiện

> Hạn nộp: `__/__/____` — **điền trước khi giao việc**

| Giai đoạn | Nội dung | Song song |
|---|---|---|
| 1 | Kiên tạo repo, `requirements.txt`, `.gitignore`, `RULES.md` (gồm quy ước dữ liệu), `PROGRESS.md`, `CLAUDE.md`, khung hàm, `core.py`, `mask.py`, `mau_dong_y.md`; bàn giao | Cả nhóm thu thập D1 (sau khi chốt cột); Đức viết hướng dẫn gán nhãn |
| 2 | Hoàn thành A, B, C — Kiên ưu tiên `softmax.py` vì D4 cần | D2: vòng thử → vòng chính → κ → phân xử |
| 3 | D3 → D4 → D5 | Nhật viết sẵn `evaluate.py`; Kiên + Nhật chốt ma trận chi phí |
| 4 | Viết và ghép báo cáo | |

## 11. Danh sách việc

### Kiên
- [ ] Tạo repo + khung hàm + `requirements.txt` + `.gitignore` + `RULES.md` + `PROGRESS.md` + `CLAUDE.md`; chốt quy ước dữ liệu trong `RULES.md`; tạo thư mục Drive chung
- [ ] A2: IRLS, softmax (có/không L2), Armijo, đối chiếu L-BFGS, log-sum-exp
- [ ] B1, B2
- [ ] C1, C3
- [ ] D1: `mask.py` + `test_mask.py`, `mau_dong_y.md`, giữ văn bản đồng ý, thu thập, nhận file raw và chạy `mask.py`, rà soát lô của Nhật, `D1_thong_ke.ipynb`
- [ ] D2: phân xử các tin lệch nhãn + `quy_trinh_phan_xu.md`
- [ ] D4: `split.py` + `test_split.py`, huấn luyện, hiệu chuẩn, `ma_tran_chi_phi.md`, chọn ngưỡng → `models/`
- [ ] Cập nhật `PROGRESS.md` mỗi khi gộp PR
- [ ] Viết phần mình + ghép báo cáo

### Đức
- [ ] A1: dẫn công thức (nhị phân, softmax, L2)
- [ ] A3: khung GLM
- [ ] B3, B4
- [ ] C2: `scoring.py` (Brier, Murphy) + notebook C2
- [ ] D1: thu thập, rà soát lô của Kiên
- [ ] D2: `huong_dan_gan_nhan.md` + gán vòng thử + gán toàn bộ
- [ ] D3: `clean()` + `fit_tfidf()` + `test_preprocess.py`
- [ ] Viết phần báo cáo

### Nhật
- [ ] A4: hiệu chuẩn + `calibration_metrics.py` (ECE, biểu đồ độ tin cậy)
- [ ] B5, B6 (kèm mở rộng 3 lớp)
- [ ] C4 (`proofs/C4.md` + notebook), C5
- [ ] D1: thu thập, rà soát lô của Đức
- [ ] D2: chọn mẫu 20%, gán vòng thử + gán chéo, tính Cohen's κ
- [ ] D4: duyệt `ma_tran_chi_phi.md`
- [ ] D5: `evaluate.py` + `test_evaluate.py`, đánh giá, điều chỉnh tỷ lệ lớp (B6), phân tích đặc trưng và lỗi
- [ ] Viết phần báo cáo