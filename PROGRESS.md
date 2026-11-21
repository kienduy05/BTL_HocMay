# Tiến độ (cập nhật mỗi khi gộp PR)

## Đã xong
- [2026-11-20] README.md, RULES.md, AGENTS.md — Kiên
- [2026-11-21] A2: core.py, logistic.py (IRLS), softmax.py (GD-Armijo, L-BFGS, SoftmaxRegression) + 36 test — Kiên
- [2026-11-21] A2_doi_chieu.ipynb: E1–E5 → results/A2_*.csv, A2_*.png (5 seed, ~110 s) — Kiên



## Đang làm / bị chặn
- D4: GD-Armijo trên TF-IDF chưa hội tụ trong 2000 vòng khi λ ≤ 1e-4 → tăng max_iter hoặc dùng warm_start khi quét λ


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

## Bước tiếp theo
- Kiên: tạo repo, requirements.txt, .gitignore (kiểm tra data/raw/ bằng git status),
  khung hàm, core.py, mask.py, mau_dong_y.md, thư mục Drive chung
- Cả nhóm: thu thập D1 sau khi chốt cột dữ liệu
- Đức: huong_dan_gan_nhan.md
- Điền hạn nộp vào README mục 10