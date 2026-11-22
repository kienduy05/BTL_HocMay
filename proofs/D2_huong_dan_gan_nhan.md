---
title: "D2. Hướng dẫn gán nhãn"
author: "Đức"
lang: vi
---

# D2. Hướng dẫn gán nhãn

Phần này trình bày hướng dẫn gán nhãn (bản đầy đủ: `docs/huong_dan_gan_nhan.md`) và kết quả gán
nhãn toàn bộ dữ liệu. Độ đồng thuận κ và kết quả phân xử được trình bày ở phần của Nhật và Kiên.

## 1. Định nghĩa ba lớp

| Nhãn | Định nghĩa | Câu hỏi kiểm tra |
|---|---|---|
| `lua_dao` (0) | Tin có ý đồ chiếm đoạt tiền, tài khoản hoặc thông tin cá nhân | Nếu làm theo tin này, người nhận có thể mất tiền hoặc mất thông tin không? |
| `rac` (1) | Quảng cáo, mời chào không mong muốn, nội dung khiêu dâm hoặc vô nghĩa, không nhằm chiếm đoạt | Tin này có đang bán, mời hoặc làm phiền mà người nhận không yêu cầu không? |
| `hop_le` (2) | Giao dịch, thông báo về dịch vụ đang dùng, thông báo của cơ quan nhà nước, tin cá nhân | Người nhận có cần biết tin này không? |

Ranh giới khó nhất không nằm ở chủ đề của tin mà ở **hành động mà tin yêu cầu**. Cùng là tin "xác thực
thuê bao": nếu hướng người nhận vào ứng dụng chính thức của nhà mạng thì là `hop_le`, nếu dẫn tới một
tên miền lạ thì là `lua_dao`. Cùng là tin "mời vay": nếu là quảng cáo có cú pháp từ chối thì là `rac`,
nếu đòi "nộp phí bảo hiểm trước khi giải ngân" thì là `lua_dao`.

## 2. Quy tắc quyết định

Để hai người gán độc lập cho cùng kết quả, các trường hợp ranh giới được xử lý theo một **thứ tự ưu
tiên cố định**. Người gán xét lần lượt các câu hỏi và dừng ở câu đầu tiên trả lời "có":

1. Tin có mạo danh, hoặc yêu cầu hành động làm mất tiền/thông tin không? (link tên miền giả, đọc OTP,
   chuyển khoản vào STK cá nhân, cài app ngoài, nộp phí trước) → `lua_dao`.
2. Cờ bạc, nhà cái, cá độ, nổ hũ → `lua_dao`.
3. Đòi nợ kèm đe dọa hoặc bêu riếu → `lua_dao`.
4. Cảnh báo của cơ quan nhà nước, không có link lạ và không yêu cầu chuyển tiền → `hop_le`.
5. Quảng cáo của đơn vị có thật (`[QC]`, "Từ chối QC", tên miền chính thức) → `rac`.
6. Hẹn hò, khiêu dâm, chuỗi ký tự vô nghĩa → `rac`.
7. Các tin còn lại (OTP, số dư, xác nhận dịch vụ, hóa đơn, tin cá nhân) → `hop_le`.

Câu 1 được đặt đầu tiên vì bỏ lọt tin lừa đảo là sai lầm tốn kém nhất (xem ma trận chi phí ở D4). Một
tin lừa đảo bắt chước văn phong quảng cáo vẫn phải được gán `lua_dao`.

| Cặp hay nhầm | Gán | Câu quyết định |
|---|---|---|
| `[TB]` nhà mạng mời nạp thẻ khuyến mại ↔ báo gói đã gia hạn | `rac` ↔ `hop_le` | 5 ↔ 7 |
| Mời vay có "Từ chối QC" ↔ "vay không thẩm định, nộp phí trước" | `rac` ↔ `lua_dao` | 5 ↔ 1 |
| Tuyển dụng, khóa học thông thường ↔ việc nhẹ lương cao qua Zalo | `rac` ↔ `lua_dao` | 5 ↔ 1 |
| Khuyến cáo "không cung cấp OTP" của công an ↔ "cán bộ điều tra" yêu cầu gọi lại | `hop_le` ↔ `lua_dao` | 4 ↔ 1 |

Ngoài ra còn hai quy ước. Thứ nhất, các tin cùng mẫu (chỉ khác tên ngân hàng, tên người hoặc số tiền)
phải mang cùng nhãn. Thứ hai, khi phân vân thì không đoán mà đánh dấu `can_xem = 1` để đưa vào phân xử.

## 3. Ví dụ (câu tự tạo theo mẫu đã che)

| Nhãn | Ví dụ |
|---|---|
| `lua_dao` | *Tai khoan dang nhap tren thiet bi la, vao https://nganhang.vn-xacthuc.top de doi mat khau.* |
| `lua_dao` | *Con bị tai nạn, mẹ chuyển gấp [MONEY] vào STK [NUMBER] giúp con, số con hỏng rồi.* |
| `lua_dao` | *Hồ sơ vay [MONEY] đã duyệt, nộp phí bảo hiểm [MONEY] vào TK [NUMBER] để giải ngân.* |
| `lua_dao` | *Tuyển CTV chốt đơn tại nhà, lương [MONEY]/ngày, kết bạn Zalo [NUMBER] nhận việc.* |
| `lua_dao` | *Tan thu dang ky nhan [MONEY], nap dau tang 100%. Truy cap: abc.xyz.me.* |
| `rac` | *[QC] Đăng ký gói 5G chỉ [MONEY]/tháng có 60GB. Từ chối QC soạn TC gửi 199.* |
| `rac` | *[TB] Tặng 20% giá trị thẻ nạp trong hôm nay. Nạp thẻ online tại https://nhamang.vn/naptien.* |
| `rac` | *Căn hộ giá gốc chủ đầu tư, chiết khấu 10%. Liên hệ [NUMBER] xem nhà mẫu.* |
| `rac` | *Spa ưu đãi 70% gói chăm sóc da dịp lễ, gọi [NUMBER] đặt lịch.* |
| `rac` | *Tìm bạn tình quanh đây, kín đáo, đăng ký miễn phí ngay.* |
| `hop_le` | *Ma OTP xac thuc giao dich la [OTP], hieu luc [TIME]. Khong cung cap ma cho bat ky ai.* |
| `hop_le` | *TK [NUMBER] +[MONEY] lúc [TIME] [DATE]. SD [MONEY]. ND: chuyen tien.* |
| `hop_le` | *Gói ST30K đã gia hạn thành công, cước [MONEY]/7 ngày. Chi tiết LH 198.* |
| `hop_le` | *[TB] Bộ Công an khuyến cáo: không cung cấp mã OTP, không bấm vào đường link lạ.* |
| `hop_le` | *Mai họp lớp lúc 7h nhé mọi người.* |

## 4. Quy trình và kết quả gán nhãn

Đức gán nhãn toàn bộ 3.012 tin trên bản đã che, không truy cập link nào trong tin. Nhãn ban đầu được
gợi ý bằng một bộ quy tắc từ khóa theo đúng thứ tự ưu tiên ở mục 2. Sau đó Đức duyệt và sửa tay toàn
bộ, đặc biệt là các tin bị đánh dấu `can_xem` và các nhóm mẫu tin lặp lại. Nhật gán độc lập 20% dữ liệu
(chọn ngẫu nhiên phân tầng) mà không xem nhãn của Đức; κ được tính trên phần này.

Sau đó nhãn được rà soát thêm hai vòng. Vòng thứ nhất dùng bước gom nhóm gần trùng của D3, phát hiện
4 cặp cùng mẫu nhưng khác nhãn; cả 4 được sửa theo câu 5. Vòng thứ hai dùng một bộ phân loại tuyến tính
huấn luyện bằng kiểm định chéo 5 fold, chỉ để rà soát chứ không phải mô hình của D4. Bộ này đồng ý với
nhãn ở 94,5% số tin. 166 tin bất đồng được đọc lại từng tin và đối chiếu với các câu quyết định ở mục 2.
Kết quả là 67 nhãn được sửa:

- 31 tin `hop_le` → `rac`: khuyến mại nạp thẻ, mời đăng ký gói cước, tin có `[QC]` (câu 5);
- 19 tin `rac` → `hop_le`: thông báo về dịch vụ hoặc đơn hàng đang dùng (câu 7);
- 17 tin → `lua_dao`: tên miền giả hoặc tên miền rẻ tiền, dụ gọi lại để "khóa/mở SIM" (câu 1).

Các tin còn lại trong số 166 tin là do bộ phân loại sai, nhãn vẫn đúng. Mọi thay đổi được ghi theo
`id` trong `sua_tay_log.csv`. Hiện không còn nhóm gần trùng nào chứa hai nhãn khác nhau.

Phân bố nhãn cuối cùng (`labels_final.csv`):

| Nguồn | lua_dao | rac | hop_le | Tổng |
|---|---|---|---|---|
| Tin thật (`tran2026`) | 680 | 944 | 887 | 2.511 |
| Tổng hợp (`tong_hop`) | 167 | 170 | 164 | 501 |
| **Tổng** | **847** | **1.114** | **1.051** | **3.012** |

Để kiểm tra các quy tắc được áp dụng nhất quán, mình đếm nhãn theo từng nhóm dấu hiệu:

| Nhóm dấu hiệu | Số tin | Phân bố nhãn |
|---|---|---|
| Có `[QC]` hoặc "Từ chối QC" | 400 | rac 400 |
| Mã OTP / mã xác minh | 212 | hop_le 202, lua_dao 10 (đòi đọc OTP cho người gọi hoặc kèm link giả) |
| Cờ bạc, nhà cái, cá độ, nổ hũ | 59 | lua_dao 57, hop_le 2 (khuyến cáo của công an chống cờ bạc) |
| Đòi nợ, nợ xấu | 23 | lua_dao 22, rac 1 |
| Phạt nguội / nộp phạt | 20 | lua_dao 19, hop_le 1 |

Các nhóm dấu hiệu rõ ràng có tỷ lệ nhất quán cao. Các tin lệch đều có lý do theo thứ tự ưu tiên ở
mục 2. Tin OTP mang nhãn `lua_dao` là tin đòi đọc mã cho người gọi (câu 1). Tin nói về cờ bạc mang nhãn
`hop_le` là khuyến cáo của công an *chống* cờ bạc (câu 4). Lớp lừa đảo
chiếm 28,1%, thấp hơn hai lớp kia, vì phần lớn tin thu thập là tin tổng đài.
