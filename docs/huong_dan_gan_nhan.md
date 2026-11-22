# Hướng dẫn gán nhãn tin nhắn (D2)

Người soạn: Đức. Áp dụng cho `labels_duc.csv`, `labels_nhat.csv` và phân xử ra `labels_final.csv`.
Mọi ví dụ dưới đây là câu **tự tạo** theo mẫu đã che, không phải tin thật.

## 1. Ba lớp

| Nhãn | Mã | Định nghĩa | Câu hỏi kiểm tra |
|---|---|---|---|
| `lua_dao` | 0 | Tin có ý đồ chiếm đoạt tiền, tài khoản hoặc thông tin cá nhân của người nhận | Nếu làm theo tin này, người nhận có thể mất tiền hoặc mất thông tin không? |
| `rac` | 1 | Tin quảng cáo, mời chào không mong muốn, hoặc nội dung vô nghĩa/khiêu dâm, nhưng không nhằm chiếm đoạt | Tin này có đang bán, mời hoặc làm phiền mà người nhận không yêu cầu không? |
| `hop_le` | 2 | Tin giao dịch, thông báo dịch vụ người nhận đang dùng, thông báo của cơ quan nhà nước, tin cá nhân | Người nhận có cần biết tin này không? |

Dấu hiệu chính của mỗi lớp:

- **Lừa đảo:**
  - mạo danh ngân hàng, cơ quan nhà nước, nhà mạng, sàn thương mại hoặc người thân;
  - link có tên miền giả hoặc rẻ tiền (*vietcombank.vn-abc.top*, *.vip*, *.xyz*);
  - yêu cầu chuyển tiền vào tài khoản cá nhân, đọc mã OTP, cài ứng dụng ngoài kho, hoặc nộp phí trước để nhận thưởng hay giải ngân;
  - tạo cảm giác gấp gáp hoặc đe dọa (khóa tài khoản, khởi tố, bêu riếu).
- **Rác:**
  - có nhãn `[QC]`/`(QC)` hoặc cú pháp "Từ chối QC soạn TC…";
  - mời đăng ký gói cước, nạp thẻ khuyến mại, mời mua bất động sản, spa, khóa học, thẻ tín dụng hoặc vay từ đơn vị có thật;
  - tin hẹn hò, khiêu dâm;
  - chuỗi ký tự vô nghĩa.
- **Hợp lệ:**
  - mã OTP hoặc mã xác minh;
  - biến động số dư;
  - xác nhận đăng ký, gia hạn hoặc hủy gói cước;
  - hóa đơn, lịch hẹn, đơn hàng;
  - thông báo hoặc khuyến cáo của cơ quan nhà nước;
  - tin nhắn giữa người quen.

## 2. Ví dụ

### 2.1. `lua_dao`
1. *Tai khoan cua ban dang nhap tren thiet bi la, vui long vao https://nganhang.vn-xacthuc.top de doi mat khau.* Link giả tên miền ngân hàng.
2. *Con bị tai nạn đang cấp cứu, mẹ chuyển gấp [MONEY] vào STK [NUMBER] giúp con, số con hỏng rồi.* Mạo danh người thân.
3. *Tan thu dang ky nhan [MONEY], nap dau tang 100%. Truy cap: abc.xyz.me.* Cờ bạc, nhà cái.
4. *Tuyển CTV chốt đơn tại nhà, lương [MONEY]/ngày, kết bạn Zalo [NUMBER] nhận việc.* Việc nhẹ lương cao.
5. *Hồ sơ vay [MONEY] đã duyệt, vui lòng nộp phí bảo hiểm [MONEY] vào TK [NUMBER] để giải ngân.* Đòi nộp phí trước.
6. *Ban dang no qua han, neu khong thanh toan truoc [TIME] chung toi se gui thong tin toi danh ba.* Đòi nợ kèm đe dọa.
7. *Phương tiện của bạn có biên bản phạt nguội, tra cứu và nộp phạt tại dichvucong-vn.vip.* Mạo danh cơ quan nhà nước.

### 2.2. `rac`
1. *[QC] Đăng ký gói 5G chỉ [MONEY]/tháng có 60GB. Soạn DK gửi 191. Từ chối QC soạn TC gửi 199.*
2. *[TB] Tặng 20% giá trị thẻ nạp trong hôm nay. Nạp thẻ online tại https://nhamang.vn/naptien.*
3. *Căn hộ giá gốc chủ đầu tư, chiết khấu 10%. Liên hệ [NUMBER] xem nhà mẫu.*
4. *[QC] Vay nhanh đến [MONEY] chỉ cần CCCD, đăng ký trên ứng dụng chính thức. Từ chối QC soạn TC gửi 199.*
5. *Tìm bạn tình quanh đây, kín đáo, đăng ký miễn phí ngay.*
6. *Spa ưu đãi 70% gói chăm sóc da dịp lễ, gọi [NUMBER] đặt lịch.*

### 2.3. `hop_le`
1. *Ma OTP xac thuc giao dich la [OTP], hieu luc [TIME]. Khong cung cap ma cho bat ky ai.*
2. *TK [NUMBER] +[MONEY] lúc [TIME] [DATE]. SD [MONEY]. ND: chuyen tien.*
3. *Gói ST30K đã gia hạn thành công, cước [MONEY]/7 ngày. Chi tiết LH 198.*
4. *[TB] Bộ Công an khuyến cáo: không cung cấp mã OTP, không bấm vào đường link lạ.*
5. *Đơn hàng [NUMBER] đã giao thành công.*
6. *Mai họp lớp lúc 7h nhé mọi người.*

## 3. Trường hợp ranh giới

Xét các câu hỏi theo đúng thứ tự dưới đây và dừng ở câu đầu tiên trả lời "có".

1. **Tin có mạo danh, hoặc yêu cầu một hành động làm mất tiền/thông tin không?** (link tên miền giả,
   đọc OTP cho người gọi, chuyển khoản vào STK cá nhân, cài app ngoài, nộp phí trước) → `lua_dao`.
   Áp dụng cả khi văn phong giống hệt tin chính thức.
2. **Tin có phải cờ bạc, nhà cái, cá độ, nổ hũ, "tặng code nạp đầu" không?** → `lua_dao` (chốt
   2026-11-21).
3. **Tin có phải đòi nợ kèm đe dọa, bêu riếu, gọi danh bạ, "chuyển hồ sơ công an" không?** → `lua_dao`.
4. **Tin có phải cảnh báo hoặc khuyến cáo của cơ quan nhà nước, không có link lạ và không yêu cầu
   chuyển tiền không?** → `hop_le`, dù trong tin có chữ "lừa đảo".
5. **Tin có phải quảng cáo hoặc mời chào của đơn vị có thật không?** (có `[QC]`, "Từ chối QC", tên miền
   chính thức, tổng đài) → `rac`. Kể cả gói cước có tính phí, khuyến mại nạp thẻ, mời vay từ ứng dụng
   chính thức, mời mở thẻ tín dụng.
6. **Tin có phải hẹn hò, khiêu dâm hoặc chuỗi ký tự vô nghĩa không?** → `rac`. Riêng trường hợp có link
   lạ kèm mồi nhử kiểu "tải bộ giải mã ảnh" thì quay lại câu 1, gán `lua_dao`.
7. Các tin còn lại (OTP, số dư, xác nhận dịch vụ, hóa đơn, lịch hẹn, tin cá nhân) → `hop_le`.

Các cặp hay nhầm:

| Cặp | Gán | Lý do |
|---|---|---|
| `[TB]` nhà mạng *mời nạp thẻ khuyến mại* ↔ `[TB]` *gói đã gia hạn / hết dung lượng* | `rac` ↔ `hop_le` | Mời mua là quảng cáo; báo về dịch vụ đang dùng là thông báo |
| Mời vay trên app chính thức có "Từ chối QC" ↔ "vay không thẩm định, nộp phí trước" | `rac` ↔ `lua_dao` | Câu 1 có hiệu lực trước câu 5 |
| Tuyển dụng hoặc khóa học thông thường ↔ việc nhẹ lương cao qua Zalo/Telegram | `rac` ↔ `lua_dao` | Câu 1 |
| Xác thực thuê bao theo Thông tư 08 qua app nhà mạng ↔ xác thực qua link lạ | `hop_le` ↔ `lua_dao` | Câu 1 |
| Đổi điểm thưởng trên app chính thức ↔ "điểm sắp hết hạn, đổi tại tên miền lạ" | `hop_le`/`rac` ↔ `lua_dao` | Câu 1 |

## 4. Quy ước khi gán

- Chỉ đọc bản đã che (`messages_masked.csv`). **Không bấm vào link** trong tin.
- Thẻ che (`[NAME]`, `[MONEY]`…) không ảnh hưởng tới nhãn.
- Các tin cùng mẫu, chỉ khác tên ngân hàng, tên người hoặc số tiền, **phải cùng nhãn**. D3 kiểm tra lại
  điều này bằng cột `nhom_gan_trung`.
- Phân vân thì không đoán. Ghi `can_xem = 1` kèm lý do ngắn để đưa vào phân xử.
- Người gán thứ hai (Nhật) gán độc lập, không xem nhãn của người thứ nhất.
