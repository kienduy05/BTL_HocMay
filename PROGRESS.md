# Tiến độ (cập nhật mỗi khi gộp PR)

## Đã xong
- [2026-11-20] README.md, RULES.md, AGENTS.md — Kiên
- [2026-11-21] A2: core.py, logistic.py (IRLS), softmax.py (GD-Armijo, L-BFGS, SoftmaxRegression) + 36 test — Kiên
- [2026-11-21] A2_doi_chieu.ipynb: E1–E5 → results/A2_*.csv, A2_*.png (5 seed, ~110 s) — Kiên
- [2026-11-21] C1_newton_tay.ipynb, C3_kha_tach.ipynb — Kiên
- [2026-11-22] D3a: preprocess.py (clean, near_dup_groups, fit_tfidf/transform_tfidf) + test; clean.csv 2.872 tin (lua_dao 814 / rac 1.097 / hop_le 961), chỉ xóa 140 tin trùng hẳn — Đức
- [2026-11-22] Che tay tên thật / tài khoản FTTH / biển số / mật khẩu còn sót (38 chỗ, log: data/labeled/sua_tay_log.csv); sửa nhãn COM_618, COM_748, COM_2266, SYN_177 → rac cho thống nhất với tin gần trùng — Đức
- [2026-11-22] Rà soát nhãn bằng bộ phân loại kiểm định chéo + đọc lại 166 tin bất đồng: sửa 67 nhãn (31 hop_le→rac, 19 rac→hop_le, 17 →lua_dao), log theo id trong sua_tay_log.csv; D2: docs/huong_dan_gan_nhan.md + report/A1_A3_B3_B4_D2_D3.docx — Đức
- [2026-11-22] D4: split.py (đã chạy một lần) + test_split.py; D4_huan_luyen.ipynb (~65 s) → results/D4_cv.csv,
  D4_hieu_chuan.csv, D4_tom_tat.txt, models/ (tfidf_word/char, softmax.npz, naive_bayes, svm, calibrators, threshold.json);
  docs/ma_tran_chi_phi.md. Chọn: softmax λ = 1e-4 (hội tụ cả 5 fold), NB α = 1, SVM C = 0,03;
  hiệu chuẩn: softmax temperature, NB none, SVM temperature — Kiên



## Đang làm / bị chặn
- D4: λ = 1e-4 được chọn nằm ở biên dưới của lưới → cân nhắc thêm 3e-5, 1e-5 (dữ liệu thật: fit_path hội tụ hết, ≤ 1.230 vòng)
- D4: ma_tran_chi_phi.md chờ Nhật duyệt


## Quyết định đã chốt
- [2026-11-20] Mã hóa nhãn: lua_dao = 0, rac = 1, hop_le = 2 (core.LABELS)
- [2026-11-20] Chia theo thời gian 70 / 7,5 / 7,5 / 15; val_a fit bộ hiệu chuẩn,
  val_b chọn bộ hiệu chuẩn và chốt ngưỡng (trên xác suất đã hiệu chuẩn)
- [2026-11-20] sklearn chỉ dùng cho Naive Bayes, SVM, TF-IDF, F1/ROC/PR;
  phần A tự cài bằng NumPy
- [2026-11-20] Tin gốc gửi trực tiếp cho Kiên; chỉ Kiên giữ data/raw/ và chạy mask.py;
  messages_masked.csv đưa lên Drive; xóa raw sau rà soát
- [2026-11-21] Loss = TRUNG BÌNH log-loss trên n mẫu (có trọng số: trung bình có trọng số)
  + (λ/2)‖W‖²; hệ số chặn b không bị phạt. Đối chiếu sklearn: C = 1/(n·λ), λ = 0 ↔ C = inf
- [2026-11-21] Nhãn đưa vào hàm là mảng số nguyên (n,); one-hot chỉ tạo bên trong
- [2026-11-21] Temperature scaling (A4) nhận logit (n, K) từ SoftmaxRegression.decision_function
- [2026-11-21] Bổ sung mean ± std theo quy định đánh giá: CV 5 fold theo thời gian trong train
  (D4_cv.csv) + bootstrap 1.000 lần trên test (D5). Ảnh hưởng evaluate.py (Nhật)
  và D4_huan_luyen.ipynb (Kiên).
- [2026-11-21] softmax.py thêm n_iter_, converged_, history['stop_reason'] và fit_path
  (warm start, λ giảm dần). D4 quét λ bằng fit_path và ghi converged_ cho mọi mô hình.
  Không đổi thuật toán, kết quả A2 giữ nguyên.
- [2026-11-22] Hiệu chuẩn (A4): bộ hiệu chuẩn nhận scores (n, K) → xác suất (n, K)
  (softmax: decision_function; NB: predict_log_proba; SVM: decision_function); fit trên val_a,
  chọn bằng log-loss + ECE trên val_b; K từ n_classes, không suy ra từ y; Platt/PAVA đa lớp
  one-vs-rest, kẹp [1e-6, 1 − 1e-6], chuẩn hóa hàng. Platt smooth dùng số mẫu (không trọng số)
  cho N₊, N₋; temperature giới hạn T ∈ [1e-4, 1e4].
- [2026-11-22] ECE: 15 bin đều, bin m = ((m−1)/M, m/M], bin 1 gồm cả 0, bin rỗng bỏ qua;
  kind binary / top_label / classwise (trung bình ECE binary theo lớp);
  ECE lớp lừa đảo = ece(probs[:, 0], labels == 0, kind="binary").

- [2026-11-22] D3: chỉ xóa trùng hẳn; tin gần trùng (Jaccard 5-gram ≥ 0,8) GIỮ, gán cột nhom_gan_trung — split.py xếp cả nhóm vào cùng một tập (gán nhóm theo date của tin sớm nhất).
- [2026-11-22] Thẻ: [MONEY] → [TIỀN], link → [LINK], [SĐT] chỉ cho số sót; thêm [LINK_LẠ], [KÝ_TỰ_LẠ].
- [2026-11-22] 495 tin SYN không có date: chỉ đưa vào train (khối 1 của CV), val_a/val_b/test chỉ gồm tin thật có date. Mục tiêu README đã sửa thành ≥ 2.800 tin sau lọc. split.py (Kiên) cần làm theo quy tắc này.
- [2026-11-22] D4a: chia theo thời gian PHÂN TẦNG THEO LỚP (khác README): trong từng lớp, 15% tin mới nhất → test,
  7,5% kế trước → val_b, 7,5% kế trước nữa → val_a, còn lại → train; nhóm nhom_gan_trung đi trọn một tập; tin không date
  chỉ ở train. Lý do: date là ngày thu thập theo đợt, tin lừa đảo dồn vào 21–25/7 → chia trên toàn bộ thì val_a, val_b
  không có tin lừa đảo, test chỉ có 5. Số tin (hop_le/lua_dao/rac): train 722/619/819, val_a 60/55/70, val_b 60/43/69,
  test 119/97/139. Đã chia xong, không chạy lại split.py.
- [2026-11-22] CV D4: train chia 6 khối cùng logic phân tầng (tin không date vào khối 1); TF-IDF fit lại mỗi fold trên khối
  huấn luyện; chọn theo log-loss trung bình; std ddof = 1. SVM trong CV: softmax(decision_function / T), T fit trên khối
  huấn luyện của fold.
- [2026-11-22] Ma trận chi phí C[thật, đoán] = [[0, 5, 10], [1, 0, 1], [5, 3, 0]] (đề xuất, docs/ma_tran_chi_phi.md);
  ngưỡng FPR 1%: ngưỡng nhỏ nhất trên p(lua_dao) đã hiệu chuẩn sao cho FPR val_b ≤ 1% (≤ 1/129 tin báo nhầm).
- [2026-11-22] Chữ ký D3: clean(df, threshold=None, group_threshold=0.8) → (clean_df, removed_df, stats); fit_tfidf(train) → (vec_word, vec_char); transform_tfidf(vec_word, vec_char, df) → csr (n, d)
- [2026-11-22] Ma trận chi phí [[0,5,10],[1,0,1],[5,3,0]] đã chốt trước D5
- [2026-11-22] Đẩy dữ liệu đã che (data/masked, data/labeled, data/splits) và models/ lên Git (repo public); xóa thư mục
  data/raw/ và consent/ (chỉ còn .gitkeep). .gitignore vẫn chặn data/raw/, consent/. Đổi README mục 3, 9 — Kiên

## Bước tiếp theo
Nhật:
- [ ] D5 (README mục 7 "D5 – Đánh giá", mục 8 bước 10): `src/evaluate.py` + `tests/test_evaluate.py`, `notebooks/D5_danh_gia.ipynb`.
  - Đầu vào: `models/` (tfidf_word/char.pkl, softmax.npz, naive_bayes.pkl, svm.pkl, calibrators.pkl, threshold.json), `data/splits/val_b.csv`, `data/splits/test.csv`.
  - CHỈ đọc test.csv trong D5_danh_gia.ipynb và chạy MỘT lần. KHÔNG huấn luyện lại, KHÔNG fit lại TF-IDF / bộ hiệu chuẩn, KHÔNG chọn lại ngưỡng hay đổi ma trận chi phí sau khi đã xem kết quả test.
  - Chỉ số cho 3 mô hình, trước và sau hiệu chuẩn:
    - F1 macro theo quy tắc argmin_j Σ_i C[i,j]·p(i|x);
    - recall lua_dao tại ngưỡng FPR 1% lấy từ threshold.json, kèm FPR thực tế đạt được trên test;
    - ECE top-label và ECE lớp lua_dao (M = 15);
    - log-loss;
    - chi phí trung bình.
  - Bootstrap 1.000 lần (seed 42) trên dự đoán test đã có: giá trị gốc, mean ± std, KTC 95% → D5_metrics.csv. Bootstrap ghép cặp để so sánh softmax với NB và với SVM; chỉ kết luận khi KTC của hiệu số không chứa 0.
  - D5_drift.csv: so sánh val_b với test. Lưu ý: split phân tầng theo lớp nên chỉ đo được trôi trong từng lớp, không đo được thay đổi tỉ lệ lớp.
  - B6: p′ ∝ p·π′/π với π lấy từ threshold.json; kịch bản lua_dao chiếm 5% và 1%; lấy mẫu lại test theo π′ 1.000 lần → D5_prior_shift.csv (ECE, log-loss, chi phí trước/sau điều chỉnh).
  - D5_confusion_matrix.png, D5_reliability.png (trước/sau hiệu chuẩn), D5_top_features.csv (n-gram từ + ký tự), D5_errors.csv (chỉ tin ĐÃ CHE, phân theo lớp × nguồn).
  - Không in nội dung tin ra output notebook, trừ D5_errors.csv.
- [ ] Viết báo cáo D5, gửi các file results/D5_
