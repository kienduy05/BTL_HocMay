# Hướng dẫn gán nhãn tin nhắn (D2)

> Phiên bản 2.1 — 22/11/2026. Bản đề xuất: cả nhóm duyệt, rồi ghi "Chốt hướng dẫn gán nhãn v2.1" vào `PROGRESS.md`.
> Mọi thay đổi sau đó phải tăng số phiên bản và ghi vào mục **Lịch sử thay đổi** ở cuối file.

## 1. Nguyên tắc chung

- Mỗi tin nhận **đúng một** nhãn: `lua_dao`, `rac` hoặc `hop_le` (ghi bằng chữ, theo `RULES.md`).
- Gán theo **ý định của người gửi**, đọc toàn bộ nội dung. Không gán chỉ vì một từ khóa: tin nói về lừa đảo (cảnh báo) không phải tin lừa đảo; tin có tiền tố `(QC)` vẫn có thể là lừa đảo.
- Xét theo thứ tự:
  1. Có dấu hiệu **lừa dối để chiếm đoạt** tiền, tài sản hay thông tin không? → `lua_dao`
  2. Nếu không: tin có phải **quảng cáo, tiếp thị, mời chào** mà người nhận không yêu cầu và không gắn với giao dịch của chính họ không? → `rac`
  3. Còn lại → `hop_le`
- Các thẻ che (`[NUMBER]`, `[MONEY]`, `[OTP]`, `[CODE]`, `[NAME]`, …) chỉ thay thông tin nhạy cảm, không phải dấu hiệu của lớp nào.
- Phân vân thì vẫn chọn nhãn khả dĩ nhất, ghi lý do vào `ghi_chu`, đặt `can_xem = 1`. Người phân xử sẽ xem lại.
- **Không truy cập** đường dẫn trong tin. Chỉ đánh giá đường dẫn qua hình thức của nó (tên miền, link rút gọn, địa chỉ IP).

## 2. Định nghĩa và dấu hiệu từng lớp

### 2.1 `lua_dao` — lừa đảo

Tin nhằm lừa người nhận chuyển tiền, cung cấp thông tin (OTP, mật khẩu, CCCD, tài khoản), bấm link độc hại, cài ứng dụng hay gọi lại số lạ.

Dấu hiệu (chỉ cần một dấu hiệu rõ ràng):
- **Mạo danh**: ngân hàng, nhà mạng, cơ quan nhà nước, công an, đơn vị giao hàng, điện nước, người quen; tên thương hiệu bị viết sai (`M0BlF0NE`, `VNPT-lnvoice`).
- **Link đáng ngờ**: tên miền không phải của thương hiệu được nhắc đến (`ghn.suco-donhang.cc`, `axzalo.me`), đuôi `.vip`, `.cc`, `.top`, `.icu`, `.club`, `.xyz`, link chỉ là địa chỉ IP.
- **Lợi ích phi thực tế**: trúng thưởng, nhận tiền hỗ trợ, hoàn tiền, "nhận 100K", lợi nhuận đầu tư cao.
- **Việc làm, nhiệm vụ online**: like, subscribe, bình chọn, chốt đơn nhận hoa hồng.
- **Cờ bạc, game đổi thưởng, cá cược**, kể cả khi viết biến dạng (`G@me`, `tra thu0ng`).
- **Né bộ lọc**: chữ biến dạng, xen ký tự (`T-H-U-O-N-G`), chữ hoa thường lẫn lộn, ký tự Unicode lạ.
- **Áp lực, đe dọa**: "quá hạn sẽ không được nhận", "khóa tài khoản", "khởi kiện", "bạn là người bảo lãnh khoản vay".
- **Yêu cầu** OTP, chuyển khoản, cài ứng dụng qua link, gọi lại số lạ để "nhận quà" hay "xử lý".

### 2.2 `rac` — quảng cáo rác

Tin quảng cáo, tiếp thị, mời chào hàng loạt, **không có dấu hiệu lừa đảo**, và **không** được gửi vì một giao dịch cụ thể của người nhận.

Bao gồm:
- Quảng cáo của **nhà mạng**: tin có `(QC)` hoặc `[QC]`, khuyến mãi nạp thẻ, mời đăng ký gói cước hay dịch vụ giá trị gia tăng ("soạn … gửi 888").
- Quảng cáo của **nhãn hàng, cửa hàng, sàn thương mại điện tử**: mã giảm giá, freeship, mời theo dõi fanpage.
- **Telesale**: mời vay, mở thẻ tín dụng, bảo hiểm, bất động sản, khi không có dấu hiệu lừa đảo.
- **Dịch vụ bất hợp pháp nhưng chỉ ở dạng quảng cáo**: làm bằng cấp, giấy tờ giả.
- Tin **tuyên truyền gửi hàng loạt** của tổ chức tư nhân hoặc tôn giáo.

### 2.3 `hop_le` — hợp lệ

Tin giao dịch, tin thông báo, tin cá nhân không nhằm bán hàng hay lừa đảo.

Bao gồm:
- **OTP, mã xác thực** do chính dịch vụ gửi khi người nhận thao tác.
- **Biến động số dư**, xác nhận thanh toán, nạp tiền, liên kết tài khoản.
- **Đơn hàng, giao hàng**, vé xe, lịch hẹn, nhắc lịch.
- **Thông báo dịch vụ** gắn với thuê bao hay tài khoản của người nhận: hết gói data, thay đổi cước, xác thực thuê bao chính chủ (link về đúng tên miền chính thức), tích điểm sau khi mua hàng.
- **Nhắc thanh toán, nhắc nợ** bình thường: chỉ nhắc số tiền và hạn, không đe dọa, không mạo danh.
- Thông báo, tuyên truyền của **cơ quan nhà nước** (VNeID, phòng chống ma túy), tin cảnh báo lừa đảo.
- **Tin cá nhân** giữa người với người.

## 3. Quy tắc cho các trường hợp ranh giới

| Mã | Trường hợp | Nhãn | Ghi chú |
|---|---|---|---|
| B1 | Quảng cáo của nhà mạng, nhãn hàng có thật, có `(QC)` hoặc mời đăng ký dịch vụ | `rac` | Dù người nhận là thuê bao, đây vẫn là tiếp thị chứ không phải giao dịch |
| B2 | Thông báo dịch vụ kèm một câu mời ưu đãi (hết gói data + gợi ý gói mới; tích điểm + mời đổi quà) | `hop_le` | Mục đích chính là thông báo về giao dịch hay thuê bao của người nhận |
| B3 | Tin `(QC)` hoặc mang tên thương hiệu nhưng link sai tên miền, chữ biến dạng | `lua_dao` | Dấu hiệu lừa đảo luôn được ưu tiên |
| B4 | Mời vay, mở thẻ: tổ chức có thật, không đòi phí, không link lạ | `rac` | Có "chỉ cần CCCD, giải ngân ngay, phí trước", link lạ → `lua_dao` |
| G1 | Nhắc nợ của ứng dụng cho vay, chỉ nhắc thanh toán | `hop_le` | Nếu link là địa chỉ IP hay tên miền giả, hoặc có đe dọa → `lua_dao` |
| G2 | Đòi nợ kèm đe dọa: "bạn là người bảo lãnh…", bêu tên, gửi thông tin cho danh bạ/người thân, "chuyển công an xử lý hình sự", "đến nhà", mạo danh cơ quan (UBND, phòng pháp lý, an ninh ngân hàng) | `lua_dao` | Đe dọa, mạo danh để ép chuyển tiền; khác G1 là nhắc nợ thông thường |
| G3 | Chuỗi ký tự ngẫu nhiên, không có nội dung đọc được | **loại** | Để trống `label`, ghi `ghi_chu = loai`; các tin này bị bỏ khỏi dữ liệu |
| G4 | Quảng cáo làm bằng cấp, giấy tờ giả | `rac` | Nếu đòi đặt cọc qua link lạ hay mạo danh cơ quan → `lua_dao` |
| B5 | Cờ bạc, game đổi thưởng, "kiếm tiền 300% bonus" | `lua_dao` | Kể cả khi trình bày như quảng cáo |
| B6 | Lời nhắn thoại được chuyển thành tin (`Tien ich Loi nhan thoai`) | theo nội dung lời nhắn | Không gán theo phần thông báo của nhà mạng |
| B7 | Tin cảnh báo lừa đảo, "không chia sẻ OTP" | `hop_le` | Có từ "lừa đảo" không làm tin thành lừa đảo |
| B8 | Tự xưng công an, cán bộ, yêu cầu cung cấp thông tin qua tin nhắn | `lua_dao` | Cơ quan thật không xin thông tin cá nhân qua SMS |
| B9 | Tin tiếng Anh, không dấu, viết tắt | gán bình thường | Chỉ loại khi không hiểu được nội dung (G3) |
| B10 | Quảng cáo sản phẩm, dịch vụ thông thường (nhà đất, khóa học, spa, gia sư…) kèm link tên miền lạ, nhưng **không** mạo danh, **không** hứa lợi phi thực tế | `rac` | Riêng đuôi tên miền (.vip, .cc…) chưa đủ để gán `lua_dao` |
| B11 | Thông báo hệ thống giả kèm link: "bạn có tin nhắn thoại chưa đọc", "nhận được ảnh riêng tư", "có lượt quẹt phải", "tải bộ giải mã" | `lua_dao` | Giả thông báo để dụ bấm link, kể cả trong tin hẹn hò |
| B12 | Kêu gọi quyên góp từ thiện qua tin nhắn (soạn tin ủng hộ) | **chưa chốt** | Gán tạm `hop_le`, đặt `can_xem = 1` để người phân xử quyết định |

## 4. Ví dụ

Các ví dụ lấy từ dữ liệu đã che (`messages_masked_v2.csv`), rút gọn.

### `lua_dao`
| id | Trích | Dấu hiệu |
|---|---|---|
| NEW_064 | "…da du d!eu k!en NHAN TIEN h0 tro… Bam vao www.wtudl.icu… QUA' HAN SE KHONG DUOC CHAP NHAN!" | chữ biến dạng, tên miền `.icu`, áp lực thời hạn |
| COM_1582 | "[VNPT-lnvoice] …hóa đơn điện tử… chưa được ký duyệt. Vui lòng tải file: http://vnpt.hoa-don-dien-tu.online" | mạo danh, tên miền giả |
| COM_1523 | "…subscribe kenh Youtube nhan hoa hong. Lam thu nhap ngay [MONEY]…" | việc làm online |
| COM_1557 | "…con gái m đg thi… thiếu ít vote… click vote giùm…: http://cuocthi.binhchon-idol.club" | nhờ bình chọn, link lạ |
| COM_2030 | "G@me bài tra thu0ng kie'mtien cuc de… rut nap sieu t0c" | cờ bạc, chữ biến dạng |
| COM_1464 | "Anh xác minh lại em Dũng công an khu vực. Ko phải lừa đảo cung cấp thông tin em…" | mạo danh công an, xin thông tin (B8) |
| COM_1822 | "Bạn là người bảo lãnh khoản vay của … đã bể nợ. Gọi [NUMBER] ngay…" | đe dọa (G2) |

### `rac`
| id | Trích | Lý do |
|---|---|---|
| COM_2766 | "(QC) Tang ngay [MONEY] vao TKC khi soan V gui [NUMBER]…" | quảng cáo nhà mạng (B1) |
| COM_903 | "[QC] Dành tặng Quý khách ưu đãi tháng 3! iPhone 16 Pro Max…" | quảng cáo bán hàng |
| COM_2136 | "(TB) Co hoi nhan toi [MONEY] vao TKC… soan GHP5 gui 888 de dang ky dich vu Game Happy…" | mời đăng ký dịch vụ giá trị gia tăng (B1) |
| COM_2583 | "Ma giam gia … ANGELA DANH TANG RIENG QUY KHACH…" | mã giảm giá nhãn hàng |
| SYN_276 | "Lazada: Freeship mọi đơn từ [MONEY] hôm nay. Mở ứng dụng để săn ngay!" | khuyến mãi sàn thương mại điện tử |
| COM_2920 | "Mình bên … shop… Bạn like và theo dõi chúng mình…" | tiếp thị fanpage |
| COM_083 | "Bên Em Nhận Làm Các Loại Bằng Cấp… GPLX…" | dịch vụ giấy tờ giả (G4) |

### `hop_le`
| id | Trích | Lý do |
|---|---|---|
| COM_951 | "Nhap ma OTP [OTP] de xac thuc Dang nhap tren ung dung My Viettel…" | OTP |
| SYN_394 | "Vietcombank: TK [NUMBER] +[MONEY] luc [TIME]… So du [MONEY]" | biến động số dư |
| SYN_334 | "Ninja Van: Don [NUMBER] du kien giao ngay [DATE]…" | giao hàng |
| SYN_425 | "Phòng khám: Nhắc lịch hẹn của Quý khách lúc [TIME] ngày [DATE]…" | nhắc lịch |
| COM_1111 | "Viettel TB: Gói ST60N đã kết thúc. Để biết các ưu đãi khác, truy cập https://viettel.vn/goidata…" | thông báo dịch vụ kèm gợi ý (B2) |
| COM_2399 | "Da qua han thanh toan, Quy KH … vui long thanh toan so tien [MONEY]… bo qua tin nhan nay neu da thanh toan" | nhắc thanh toán bình thường (G1) |
| COM_834 | "[TB] Thực hiện thủ tục cấp Phiếu lý lịch tư pháp trên ứng dụng VNeID…" | thông báo cơ quan nhà nước |

## 5. Quy trình

1. **Vòng thử**: Đức và Nhật gán độc lập cùng 100 tin, tính Cohen's κ, thảo luận các tin lệch, sửa hướng dẫn nếu cần (tăng phiên bản).
2. **Vòng chính**: Đức gán toàn bộ; Nhật gán độc lập 20% (`crosscheck_ids.csv`), **không xem** nhãn của Đức.
3. Nhãn do quy tắc hoặc AI gợi ý chỉ là điểm xuất phát: người gán phải đọc lại từng tin; cột `nguon_nhan` ghi rõ nhãn cuối do ai quyết định.
4. Ngưỡng chấp nhận κ ≥ 0,6. Thấp hơn thì xem lại các trường hợp ranh giới, sửa hướng dẫn, gán lại phần lệch.
5. Tin lệch nhãn và tin `can_xem = 1` do Kiên phân xử theo `quy_trinh_phan_xu.md`; lý do ghi vào `ly_do_phan_xu`.

## 6. Định dạng file nhãn

Theo `RULES.md`: `id`, `label` (`lua_dao` | `rac` | `hop_le`; để trống nếu loại), `ghi_chu`, và tùy chọn `can_xem` (0/1). Lưu CSV bằng `encoding="utf-8-sig"` để mở bằng Excel không lỗi font.

## Lịch sử thay đổi

| Phiên bản | Ngày | Thay đổi |
|---|---|---|
| v2.1 | 22/11/2026 | Mở rộng G2 (mọi tin đòi nợ có đe dọa hoặc mạo danh cơ quan); thêm B10 (quảng cáo kèm tên miền lạ nhưng không mạo danh vẫn là `rac`), B11 (thông báo hệ thống giả kèm link), B12 (từ thiện, chưa chốt). |
| v2 | 22/11/2026 | Thêm định nghĩa đầy đủ, bảng ranh giới B1–B9 và G1–G4, ví dụ từ dữ liệu thật, quy trình vòng thử/vòng chính, quy định về nhãn gợi ý của quy tắc/AI. |
