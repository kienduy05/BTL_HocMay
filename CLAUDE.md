# Chỉ dẫn cho Claude Code

@RULES.md
@PROGRESS.md

## Tuyệt đối không
- Mở, đọc, tìm kiếm hay in nội dung file trong data/raw/ và consent/
  (tin chưa che, văn bản đồng ý). Chỉ mask.py được xử lý data/raw/.
- Đọc data/splits/test.csv, trừ khi đang làm D5_danh_gia.ipynb.

## Trước khi làm, đọc mục README tương ứng
Chỉ đọc mục cần thiết, không đọc cả README.md.

| Việc | Đọc trong README.md |
|---|---|
| Phần A, B, C (src/, notebooks/C*, proofs/) | Mục 3 (vị trí file, file nào dùng cho đâu) |
| Bất kỳ việc gì phần D | Mục 6 (thứ tự) + mục 9 (quy tắc bắt buộc) |
| mask.py, test_mask.py, D1_thong_ke | Mục 7 – D1 |
| D2_cohen_kappa, file nhãn | Mục 7 – D2 |
| preprocess.py, test_preprocess.py | Mục 7 – D3 |
| split.py, D4_huan_luyen, ma_tran_chi_phi.md | Mục 7 – D4 |
| evaluate.py, D5_danh_gia | Mục 7 – D5 |
| Chạy pipeline, đầu vào/đầu ra từng bước | Mục 8 |
| Tạo file/thư mục mới | Mục 3 |
| Git, commit, PR | Mục 5 |

## Giao việc cho subagent
Mục tiêu: tiết kiệm token. Model chính chỉ làm việc cần suy luận sâu; việc nhẹ giao cho
subagent model nhỏ (truyền model khi gọi). Tự đánh giá việc nào nặng, việc nào nhẹ.

- Haiku: tìm file, grep, đọc lướt, chạy pytest và tóm tắt lỗi, kiểm tra git status,
  xóa output notebook.
- Sonnet: code khuôn mẫu, viết test theo đặc tả có sẵn, vẽ biểu đồ, notebook C/D1,
  docstring, sửa lỗi nhỏ đã rõ nguyên nhân.
- Model chính: thuật toán phần A (IRLS, softmax + Armijo, PAVA, temperature),
  kiểm tra rò rỉ dữ liệu (train/val_a/val_b/test), thiết kế hàm mới, duyệt code subagent.

Không giao việc chỉ vài dòng: chi phí khởi tạo subagent lớn hơn tự làm.
Yêu cầu subagent trả về kết quả ngắn gọn (tóm tắt, diff), không dán lại nội dung file dài.
Khi giao việc, ghi rõ subagent phải tuân thủ RULES.md và mục "Tuyệt đối không" ở trên.

## Khi xong một việc
- Chạy pytest cho phần liên quan.
- Đề xuất dòng cập nhật cho PROGRESS.md.