---
title: "D3. Tiền xử lý"
author: "Đức"
lang: vi
---

# D3. Tiền xử lý

Tin nhắn lừa đảo thường được viết để qua mặt bộ lọc. Người gửi chèn ký tự vào giữa chữ, thay chữ cái
bằng chữ số, dùng chữ toán học, viết không dấu, và gửi cùng một mẫu tin nhiều lần, chỉ đổi tên ngân
hàng hoặc đường link. Bước D3 đưa các biến thể đó về cùng một dạng, để mô hình học được nội dung tin
thay vì học cách trình bày bề ngoài. Bước này cũng xử lý tin trùng, để kết quả đánh giá không bị
thổi phồng.

Mã nguồn nằm ở `src/preprocess.py`, kiểm thử ở `tests/test_preprocess.py` (59 test, dữ liệu tự tạo).
D3 gồm hai phần:

- **D3a (làm sạch)** chạy **trước** khi chia dữ liệu và chỉ dùng quy tắc cùng từ điển cố định, không
  học tham số nào từ dữ liệu, nên không gây rò rỉ thông tin.
- **D3b (TF-IDF)** chạy **sau** khi chia và chỉ fit trên train.

Lệnh `python src/preprocess.py` đọc `labels_final.csv` (ghép với `messages_masked.csv` theo `id`,
chỉ lấy tin `reviewed = 1`) và ghi ra:

- `data/labeled/clean.csv`: dữ liệu đã làm sạch;
- `data/labeled/dedup_removed.csv`: danh sách tin bị xóa, ghi theo `id`;
- `results/D3_thong_ke.csv`: các số liệu thống kê.

Không file nào trong số này có nội dung tin bị in ra log.

## 1. Chuẩn hóa văn bản (D3a)

Mỗi tin lần lượt đi qua các bước dưới đây. Thứ tự các bước quan trọng: thẻ và link phải được thay
**trước** khi bỏ dấu câu, nếu không các dấu chấm và gạch chéo trong link sẽ bị tách vụn. Các thẻ được
giữ bằng một placeholder toàn chữ cái, để những bước sau (bỏ dấu câu, đổi chữ số, tách từ) không làm
vỡ thẻ. Bảng sau liệt kê từng quy tắc, kèm ví dụ (câu tự tạo) và số tin trong 2.872 tin bị tác động.

| # | Quy tắc | Ví dụ | Số tin |
|---|---|---|---|
| 1 | NFKC rồi NFC; bỏ ký tự ẩn, ký tự điều khiển (kể cả xuống dòng), biểu tượng | *𝙉𝙝𝙖𝙣 quà* → *nhan quà* | 514 |
| 2 | Tách chữ dính kiểu CamelCase, chỉ với từ dài ≥ 15 ký tự | *NhanNgayMotTram* → *nhan ngay mot tram* | 283 |
| 3 | Viết thường; dấu thanh kiểu mới cho vần *oa, oe, uy* cuối âm tiết | *Hoà, thuý* → *hòa, thúy* (giữ *hoàng, quý*) | 13 |
| 4 | Thẻ của `mask.py`: `[MONEY]` → `[TIỀN]`, các thẻ khác giữ nguyên | *nhận [MONEY]* → *nhận [TIỀN]* | – |
| 5 | Link → `[LINK]`, thêm `[LINK_LẠ]` nếu link đáng ngờ | *vào abc.top* → *vào [LINK] [LINK_LẠ]* | 1.455 |
| 6 | Số điện thoại / số tiền còn sót → `[SĐT]` / `[TIỀN]` | *nạp 100k* → *nạp [TIỀN]* | 9 |
| 7 | Ghép chữ bị chèn ký tự (từ 3 chữ cái trở lên) | *n.g.â.n h.à.n.g* → *ngân hàng* | 11 |
| 8 | Ký tự thay chữ nằm giữa hai chữ cái (0→o, 1→i, 3→e, 4→a, 6→b, @→a, !→i) | *kh0a, d!eu* → *khoa, dieu* | 84 |
| 9 | Gộp biến thể của 30 thương hiệu hay bị giả mạo | *V1etc0mbank, vlETCOMBANK* → *vietcombank* | 10 |
| 10 | Bỏ dấu câu (giữ %), rút gọn chữ lặp ≥ 3 lần | *haaay!!!* → *hay* | 92 |
| 11 | Teencode / viết tắt (34 mục, chỉ khớp nguyên từ) | *k đc, ck vào stk* → *không được, chuyển khoản vào số tài khoản* | 1.013 |
| 12 | Cờ `[KÝ_TỰ_LẠ]` khi có ≥ 4 chỗ chèn ký tự lạ giữa chữ | *Li\`xi~ph;on,g* → … `[KÝ_TỰ_LẠ]` | 24 |

Một số điểm thiết kế để tránh sửa nhầm:

- **Quy tắc 5 (link).** Danh sách đuôi tên miền là danh sách cố định. Các đuôi trùng với từ tiếng Việt
  không dấu (*.de* = để, *.me* = mẹ, *.co* = có, *.to* = tô) chỉ được nhận là link khi có ít nhất 2
  dấu chấm hoặc có đường dẫn, để không nhầm *hom nay.De biet* thành link. Cờ `[LINK_LẠ]` được bật khi
  đuôi tên miền thuộc nhóm rẻ tiền (*.top, .vip, .xyz, .cc, .icu…*) hoặc khi tên miền chứa tên thương
  hiệu nối bằng gạch ngang (*vietcombank.vn-abc.com*), một kiểu giả mạo rất phổ biến.
- **Quy tắc 8 (ký tự thay chữ).** Chỉ đổi khi ký tự nằm **giữa hai chữ cái**. Nhờ vậy các mã gói
  cước như *st5k, 5g10, tv360*, số đo *1.5gb* hay ngày *12/3* được giữ nguyên.
- **Quy tắc 9 (thương hiệu).** So khớp trên "dạng gộp": đổi ký tự thay chữ, đổi *l, j* thành *i*, bỏ
  dấu, `.`, `_`, `-`. Tên chỉ bị thay khi dạng gộp trùng đúng một thương hiệu trong danh sách.
- **Quy tắc 11 (teencode).** Chỉ thay khi khớp nguyên từ, nên *k* đổi thành *không*, nhưng *kho* thì
  không đổi. Các viết tắt mơ hồ (*b, e, a, kh*) cố ý không đưa vào từ điển.

## 2. Thẻ đặc trưng

Bảng dưới cho biết tỷ lệ tin (%) trong mỗi lớp có chứa từng thẻ, tính trên `clean.csv`.

| Thẻ | lua_dao | rac | hop_le |
|---|---|---|---|
| `[LINK]` | 68,1 | 58,1 | 27,0 |
| `[LINK_LẠ]` | **33,3** | 5,7 | 4,5 |
| `[TIỀN]` | 39,8 | **81,9** | 35,8 |
| `[NUMBER]` | 30,0 | 70,2 | 49,3 |
| `[OTP]` | 0,2 | 0,0 | **10,2** |
| `[KÝ_TỰ_LẠ]` | 2,8 | 0,1 | 0,0 |

- `[LINK_LẠ]` tách lớp lừa đảo khá rõ: gần 1/3 tin lừa đảo có thẻ này, so với dưới 6% ở hai lớp còn
  lại. Nếu chỉ thay link bằng một thẻ `[LINK]` chung, tín hiệu này sẽ mất.
- `[TIỀN]` đặc trưng cho tin rác, tức tin khuyến mại nêu giá gói cước.
- `[OTP]` gần như chỉ có ở tin hợp lệ.
- `[SĐT]` không xuất hiện trong tin nào, vì `mask.py` đã gộp số điện thoại vào `[NUMBER]`. Thẻ này
  vẫn được giữ để dùng cho các lô dữ liệu sau.

## 3. Ba dạng biểu diễn

Mỗi tin được lưu ở ba dạng:

- `text_clean`: văn bản có dấu, dùng để đọc lại và phân tích lỗi.
- `text_khongdau`: văn bản bỏ dấu (*đ → d*), dùng cho n-gram ký tự. Thẻ vẫn giữ nguyên dạng.
- `text_tok`: văn bản đã tách từ bằng pyvi, từ ghép nối bằng `_` (*chuyển_khoản*), dùng cho n-gram từ.
  pyvi đôi khi nối thẻ với từ đứng cạnh, nên sau khi tách từ, các thẻ được tách ra lại thành token
  riêng.

Có dạng `text_khongdau` vì nhiều tin được viết không dấu. Tỷ lệ tin gốc hoàn toàn không dấu là 34,4%
ở lớp lừa đảo, 52,9% ở lớp rác và 57,6% ở lớp hợp lệ, phần lớn là tin tổng đài. Với n-gram ký tự trên
dạng bỏ dấu, *Tài khoản bị khóa* và *Tai khoan bi khoa* cho cùng một biểu diễn. Độ dài trung vị
(số token) là 24 (lừa đảo), 40 (rác) và 27 (hợp lệ).

## 4. Xử lý tin trùng

### 4.1. Trùng hẳn: xóa

Hai tin được coi là trùng hẳn khi `text_khongdau` của chúng giống hệt nhau, tức là chỉ khác ở
hoa/thường, dấu câu, khoảng trắng hoặc có dấu/không dấu. Tin được duyệt theo `date` tăng dần, tin không
có ngày xếp cuối. Trong mỗi nhóm trùng, tin cũ nhất được giữ, phù hợp với việc chia dữ liệu theo thời
gian. Kết quả: xóa **140 tin**, còn **2.872 tin**.

| Lớp | Đầu vào | Xóa | Còn lại |
|---|---|---|---|
| lua_dao | 847 | 33 | 814 |
| rac | 1.114 | 17 | 1.097 |
| hop_le | 1.051 | 90 | 961 |
| **Tổng** | **3.012** | **140** | **2.872** |

Lớp hợp lệ bị xóa nhiều nhất, vì tin tổng đài như mã OTP hay thông báo thanh toán lặp lại nguyên văn.

### 4.2. Gần trùng: giữ nhưng gom nhóm

Hai tin là gần trùng khi độ Jaccard giữa hai tập 5-gram ký tự của chúng đạt từ $t = 0{,}8$ trở lên:
$$
J(A, B) = \frac{|A \cap B|}{|A \cup B|} \ge t .
$$
Gọi $X \in \{0, 1\}^{n \times V}$ là ma trận cho biết mỗi tin chứa những shingle nào. Khi đó
$|A \cap B|$ của mọi cặp là phần tử của $XX^\top$ (ma trận thưa), và
$|A \cup B| = |A| + |B| - |A \cap B|$. Với $n \approx 3.000$, toàn bộ phép tính mất dưới 1 giây.

Khi xem tay, các cặp có $J$ từ 0,8 đến 0,86 vẫn là cùng một mẫu tin, chỉ khác tên ngân hàng, tên
người hoặc số tiền. Bảng dưới cho thấy nếu xóa luôn tin gần trùng thì sẽ mất bao nhiêu tin:

| Cách xử lý | Còn lại | Xóa trùng hẳn | Xóa gần trùng |
|---|---|---|---|
| Chỉ xóa trùng hẳn (đã chọn) | 2.872 | 140 | 0 |
| Xóa cả gần trùng, $t = 0{,}95$ | 2.802 | 139 | 71 |
| Xóa cả gần trùng, $t = 0{,}90$ | 2.670 | 123 | 219 |
| Xóa cả gần trùng, $t = 0{,}80$ | 2.444 | 114 | 454 |

Xóa gần trùng ở $t = 0{,}8$ làm mất khoảng 15% dữ liệu và đưa số tin xuống dưới mục tiêu 2.800. Mặt
khác, nhiều mẫu tin lừa đảo thật sự được gửi hàng loạt với biến thể nhỏ, nên giữ các biến thể đó giúp
mô hình học được độ đa dạng.

Vì vậy nhóm chọn **giữ** tin gần trùng, nhưng gán chung mã `nhom_gan_trung`. Mã này là thành phần liên
thông của đồ thị có cạnh nối hai tin khi $J \ge 0{,}8$. Có 701 tin (24%) thuộc 240 nhóm có từ 2 tin trở
lên; nhóm lớn nhất có 26 tin. Khi chia dữ liệu (D4a), mỗi nhóm đi trọn vào một tập. Điều này ngăn
trường hợp một biến thể nằm trong train còn biến thể kia nằm trong test, vốn sẽ làm điểm số trên test
cao giả tạo.

### 4.3. Tin gần trùng nhưng khác nhãn

Nhờ gom nhóm, phát hiện được 4 cặp gần trùng mang nhãn khác nhau. Cả 4 được sửa thành `rac` theo hướng
dẫn gán nhãn. Bước rà soát nhãn sau đó (xem D2) sửa thêm 63 nhãn. Sau toàn bộ các lần sửa, không còn
nhóm gần trùng nào chứa hai nhãn khác nhau. Như vậy, bước gom nhóm còn có ích như một công cụ kiểm tra
chất lượng nhãn.

### 4.4. Hệ quả cho bước chia dữ liệu

Chỉ tin có `date` (2.377 tin thật) được chia theo thời gian 70 / 7,5 / 7,5 / 15, mỗi nhóm
`nhom_gan_trung` xếp theo `date` của tin sớm nhất trong nhóm. 495 tin tổng hợp (SYN) không có ngày
(lừa đảo 161, rác 170, hợp lệ 164) chỉ vào train và thuộc khối 1 của kiểm định chéo, nên val_a, val_b
và test chỉ gồm tin thật.

## 5. TF-IDF (D3b)

`fit_tfidf(train)` fit hai `TfidfVectorizer` **chỉ trên train**:

| Tham số | n-gram từ | n-gram ký tự |
|---|---|---|
| Cột đầu vào | `text_tok` | `text_khongdau` |
| Kiểu | `word`, tách theo khoảng trắng | `char_wb` (n-gram trong phạm vi từ) |
| Độ dài n-gram | 1–2 | 2–5 |
| `min_df` | 2 | 2 |
| `sublinear_tf` | có ($1 + \log \mathrm{tf}$) | có |

- n-gram từ bắt được cụm có nghĩa (*chuyển_khoản*, *[LINK] [LINK_LẠ]*).
- n-gram ký tự chịu được lỗi chính tả và biến thể còn sót (*vietcombamk*), và không phụ thuộc chất
  lượng tách từ.
- `min_df = 2` bỏ các n-gram chỉ xuất hiện một lần, phần lớn là chuỗi ngẫu nhiên trong tên miền lừa
  đảo.

`transform_tfidf` áp hai vectorizer đã fit cho val_a, val_b và test, rồi ghép thành ma trận thưa
$(n,\, d_{\text{từ}} + d_{\text{ký tự}})$. Từ không có trong train sẽ bị bỏ qua. D4 lưu hai vectorizer
thành `models/tfidf_word.pkl` và `models/tfidf_char.pkl`.

## 6. Kiểm thử

59 test dùng câu tự tạo. Các test kiểm tra:

- Unicode và dấu thanh (kể cả các trường hợp không được đổi như *hoàng, quý*);
- nhận đúng link và không nhầm *hom nay.de*, *1.5gb*;
- từng quy tắc xử lý biến thể cố ý;
- độ Jaccard khớp định nghĩa, giữ tin có ngày sớm nhất, gom nhóm gần trùng;
- từ vựng TF-IDF chỉ lấy từ train;
- CLI không in nội dung tin ra màn hình.

## 7. Hạn chế

- Tin viết không dấu chưa được khôi phục dấu; phần này mới chỉ được bù bằng dạng `text_khongdau`.
  Khôi phục dấu cần một mô hình riêng, nằm ngoài phạm vi đề tài.
- Danh sách đuôi tên miền, thương hiệu và teencode được lập bằng tay. Các biến thể mới (đuôi tên miền
  mới, cách viết tắt mới) cần được bổ sung theo thời gian.
