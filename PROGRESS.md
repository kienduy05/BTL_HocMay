# Tiến độ (cập nhật mỗi khi gộp PR)

## Đã xong
- [2026-11-20] README.md, RULES.md, AGENTS.md — Kiên

## Đang làm / bị chặn
-

## Quyết định đã chốt
- [2026-11-20] Mã hóa nhãn: lua_dao = 0, rac = 1, hop_le = 2 (core.LABELS)
- [2026-11-20] Chia theo thời gian 70 / 7,5 / 7,5 / 15; val_a fit bộ hiệu chuẩn,
  val_b chọn bộ hiệu chuẩn và chốt ngưỡng (trên xác suất đã hiệu chuẩn)
- [2026-11-20] sklearn chỉ dùng cho Naive Bayes, SVM, TF-IDF, F1/ROC/PR;
  phần A tự cài bằng NumPy
- [2026-11-20] Tin gốc gửi trực tiếp cho Kiên; chỉ Kiên giữ data/raw/ và chạy mask.py;
  messages_masked.csv đưa lên Drive; xóa raw sau rà soát
- [2026-11-21] Bổ sung mean ± std theo quy định đánh giá: CV 5 fold theo thời gian trong train
  (D4_cv.csv) + bootstrap 1.000 lần trên test (D5). Ảnh hưởng evaluate.py (Nhật)
  và D4_huan_luyen.ipynb (Kiên).

## Bước tiếp theo
- Kiên: tạo repo, requirements.txt, .gitignore (kiểm tra data/raw/ bằng git status),
  khung hàm, core.py, mask.py, mau_dong_y.md, thư mục Drive chung
- Cả nhóm: thu thập D1 sau khi chốt cột dữ liệu
- Đức: huong_dan_gan_nhan.md
- Điền hạn nộp vào README mục 10