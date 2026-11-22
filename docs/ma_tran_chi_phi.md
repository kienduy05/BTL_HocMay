# Ma trận chi phí (D4)

> Bản đề xuất của Kiên, Nhật cần duyệt (README mục 11). Giá trị đang dùng ở `models/threshold.json`
> và `notebooks/D4_huan_luyen.ipynb` (biến `C_COST`). Đổi ma trận thì sửa cả ba nơi, chạy lại D4,
> rồi ghi vào PROGRESS.md.

## 1. Ma trận

`C[lớp thật, lớp đoán]`, thứ tự lớp theo `core.LABELS` (lua_dao = 0, rac = 1, hop_le = 2).
Đơn vị là chi phí tương đối; đoán đúng thì chi phí 0.

| thật \ đoán | lua_dao | rac | hop_le |
|---|---:|---:|---:|
| **lua_dao** | 0 | 5 | **10** |
| **rac**     | 1 | 0 | 1 |
| **hop_le**  | 5 | 3 | 0 |

```python
C = [[0, 5, 10],
     [1, 0, 1],
     [5, 3, 0]]
```

## 2. Lý giải từng ô

Bối cảnh: bộ lọc tin nhắn trên điện thoại. Đoán **lua_dao** thì hiện cảnh báo đỏ hoặc chặn tin.
Đoán **rac** thì chuyển tin vào mục rác, không báo. Đoán **hop_le** thì tin vào hộp thư chính như bình thường.

| Ô | Chi phí | Lý do |
|---|---:|---|
| C[lua_dao, hop_le] | 10 | Đắt nhất. Tin lừa đảo vào hộp thư chính mà không kèm cảnh báo nào. Người nhận có thể làm theo (bấm link, chuyển tiền, đọc OTP), và thiệt hại tài chính thì khó lấy lại. |
| C[lua_dao, rac] | 5 | Tin vẫn lọt cảnh báo, nhưng nằm trong mục rác nên ít khả năng được đọc và tin theo hơn hộp thư chính. Vì vậy chỉ bằng một nửa ô trên. |
| C[hop_le, lua_dao] | 5 | Chặn nhầm tin hợp lệ, ví dụ OTP thật, thông báo của ngân hàng hay tin từ người quen. Người dùng lỡ việc quan trọng và mất lòng tin vào bộ lọc. Chặn nhầm tin hợp lệ đắt hơn để lọt tin rác (C[rac, hop_le] = 1), đúng tinh thần README mục 7. |
| C[hop_le, rac] | 3 | Tin hợp lệ bị đưa vào mục rác: vẫn còn đó nhưng dễ bị bỏ sót. Nhẹ hơn chặn kèm cảnh báo lừa đảo (5), nặng hơn để lọt rác (1). |
| C[rac, hop_le] | 1 | Lọt tin rác vào hộp thư chính chỉ gây phiền, không gây hại. |
| C[rac, lua_dao] | 1 | Báo tin rác (quảng cáo) là lừa đảo: hơi quá tay, nhưng người dùng vốn cũng không cần tin này. |

Các ràng buộc mà ma trận thỏa mãn:
- Bỏ lọt lừa đảo là đắt nhất: C[lua_dao, hop_le] = 10 lớn hơn mọi ô khác.
- Chặn nhầm tin hợp lệ đắt hơn để lọt rác: C[hop_le, lua_dao] = 5 và C[hop_le, rac] = 3, đều lớn hơn C[rac, hop_le] = 1.
- Đường chéo bằng 0, mọi ô ngoài đường chéo đều dương.

## 3. Quy tắc quyết định

Gọi p(i|x) là xác suất **đã hiệu chuẩn** (bộ hiệu chuẩn chọn trên val_b, xem `results/D4_hieu_chuan.csv`).
Chi phí kỳ vọng khi đoán lớp j:

$$R(j \mid x) = \sum_i C[i, j]\, p(i \mid x), \qquad \hat y(x) = \arg\min_j R(j \mid x).$$

Viết dạng ma trận: với `P` có shape (n, K), lấy `pred = (P @ C).argmin(axis=1)`.
Khi C = 1 − I (chi phí 0-1), quy tắc này trùng với argmax.

### Trường hợp 2 lớp chính là B5

Xét hai lớp, lừa đảo (+) và không lừa đảo (−), với C[+, +] = C[−, −] = 0, c_FN = C[+, −], c_FP = C[−, +].
Đặt p = p(+|x). Khi đó:

$$\hat y = + \iff p\,c_{FN} \ge (1-p)\,c_{FP} \iff p \ge t^\* = \frac{c_{FP}}{c_{FP} + c_{FN}}.$$

Đây đúng là ngưỡng Bayes của B5. Ví dụ, nếu chỉ giữ hai lớp lua_dao và hop_le của ma trận trên
thì c_FN = 10, c_FP = 5, nên t* = 5 / 15 = 1/3. Vì bỏ lọt đắt gấp đôi báo nhầm, mô hình báo lừa đảo ngay từ khi p ≥ 1/3, không chờ tới 1/2.

## 4. Ngưỡng tại FPR = 1% (một đầu ra khác)

Ngoài quy tắc chi phí, `threshold.json` còn lưu một ngưỡng vận hành cho bài toán lừa đảo so với phần còn lại:
- điểm số là p(lua_dao | x) sau hiệu chuẩn; báo lừa đảo khi điểm ≥ t;
- t là giá trị nhỏ nhất sao cho FPR trên val_b ≤ 1%.

val_b chỉ có 129 tin không lừa đảo, nên 1% ứng với tối đa 1 tin báo nhầm, và độ phân giải của ngưỡng rất thô
(mỗi tin = 0,78 điểm %). D5 dùng đúng ngưỡng này trên test, không chọn lại, và báo cả FPR thực tế đạt được.
