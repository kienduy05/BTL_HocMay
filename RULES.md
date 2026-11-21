# Quy tắc cho AI

## Quy ước dữ liệu
- Nhãn: lua_dao = 0, rac = 1, hop_le = 2 (core.LABELS).
- Shape: X (n, d) | W (d, K) | b (K,) | probs (n, K), mỗi hàng tổng = 1.
- Nhãn nhị phân: {0, 1} trong code; {−1, +1} chỉ dùng trong chứng minh B2.
- Ma trận chi phí C[thật, đoán], cùng thứ tự LABELS.
- Seed = 42 cho mọi bước có yếu tố ngẫu nhiên.

Cột file raw data/raw/<tên>_<ngày>.csv:
- text: nội dung tin gốc
- date: YYYY-MM-DD (ngày nhận với tình nguyện viên, ngày đăng với bản tin/báo chí)
- source_type: co_quan | bao_chi | tinh_nguyen
- source_ref: đường dẫn bài gốc, hoặc mã văn bản đồng ý

Cột data/masked/messages_masked.csv (do mask.py sinh):
- id: mã tin, duy nhất
- text: nội dung đã che
- date, source_type, source_ref: giữ nguyên từ file raw
- collector: người thu thập
- reviewed: 0/1; chỉ dùng tin reviewed = 1 ở các bước sau

Khung hàm mẫu:
```python
def ece(probs, labels, n_bins=15, kind="top_label") -> float:
    """probs: (n,) xác suất lớp dương hoặc (n, K); labels: (n,);
    kind: 'binary' | 'top_label' | 'classwise'"""
```

## Code tự viết (yêu cầu của đề, Phần A)
- logistic.py, softmax.py, glm.py, calibration.py, calibration_metrics.py, scoring.py
  phải tự cài bằng NumPy.
- Không dùng sklearn/statsmodels thay cho các file trên; không dùng
  sklearn.calibration (kể cả CalibratedClassifierCV) và các hàm ECE/Brier có sẵn.
- scipy.optimize (L-BFGS) chỉ dùng để đối chiếu với softmax.py.
- sklearn được dùng cho: Naive Bayes, SVM, TfidfVectorizer, F1/ROC/PR,
  và để đối chiếu trong tests.
- D4 phải dùng softmax.py làm hồi quy logistic đa lớp.

## Dữ liệu
- Chỉ mask.py được đọc hoặc ghi data/raw/.
- Không chia lại dữ liệu; chỉ đọc từ data/splits/.
- TF-IDF, từ vựng, mọi thứ "học" từ dữ liệu: chỉ fit trên train.
- Siêu tham số: chọn bằng kiểm định chéo theo thời gian trong train.
- Bộ hiệu chuẩn: chỉ fit trên val_a.
- Chọn bộ hiệu chuẩn và chốt ngưỡng trên val_b; ngưỡng tính trên xác suất ĐÃ hiệu chuẩn.
- test.csv chỉ được đọc trong D5_danh_gia.ipynb.
- Xử lý biến thể chính tả bằng quy tắc và từ điển cố định, không học từ dữ liệu.
- Thẻ đặc trưng thống nhất: [LINK], [TIỀN], [SĐT], khớp với thẻ của mask.py.
- Kiểm định chéo theo thời gian: 5 fold, cửa sổ mở rộng, chỉ trong train.
- Bootstrap trên test: 1.000 lần, seed 42, chỉ lấy mẫu lại dự đoán đã có;
  không huấn luyện lại, không chọn lại ngưỡng hay bộ hiệu chuẩn.

## Quyền riêng tư
- Không viết code truy cập URL trong tin nhắn.
- Không hiển thị nội dung tin nhắn trong output notebook hoặc log.
  Ngoại lệ: results/D5_errors.csv được chứa tin ĐÃ CHE để phân tích lỗi.

## Kỹ thuật
- Dùng core.sigmoid, core.logsumexp; không gọi np.exp trực tiếp trên logit.
- Hàm mới phải có test trong tests/.
- Docstring ghi rõ shape đầu vào/đầu ra.
- Không tự đổi chữ ký hàm trong khung; muốn đổi thì ghi vào PROGRESS.md.

## Cho người dùng AI
- Chỉ dán tin đã che vào công cụ AI.