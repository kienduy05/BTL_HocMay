---
title: "A3. Khung GLM với IRLS dùng chung"
author: "Đức"
lang: vi
---

# A3. Khung GLM với IRLS dùng chung

## 1. Mô hình

GLM gồm ba thành phần: phân phối thuộc họ mũ
$p(y \mid \theta) = h(y)\exp\{(y\theta - b(\theta))/\phi\}$, bộ dự báo tuyến tính
$\eta = x^\top w + b_0$, và hàm liên kết $g(\mu) = \eta$ với $\mu = \mathbb E[y]$. (Hệ số chặn viết
$b_0$ để không trùng với hàm $b(\theta)$; trong code vẫn gọi là `b`.) Theo B3,
$\mu = b'(\theta)$ và $\operatorname{Var}(y) = \phi\, V(\mu)$ với $V(\mu) = b''(\theta)$. Liên kết
**chính tắc** chọn $\theta = \eta$. Ba họ được cài:

| Họ | $y$ | $b(\theta)$ | $\phi$ | Liên kết $g$ | $\mu = g^{-1}(\eta)$ | $V(\mu)$ |
|---|---|---|---|---|---|---|
| Bernoulli | $\{0, 1\}$ | $\log(1+e^\theta)$ | $1$ | logit | $\sigma(\eta)$ | $\mu(1-\mu)$ |
| Poisson | $\{0, 1, 2, \dots\}$ | $e^\theta$ | $1$ | log | $e^\eta$ | $\mu$ |
| Gauss | $\mathbb R$ | $\theta^2/2$ | $\sigma^2$ | đồng nhất | $\eta$ | $1$ |

Hàm mất mát (lấy $\phi = 1$, bỏ hằng số không phụ thuộc tham số), cùng dạng trung bình với
`logistic.py`:
$$
\mathcal L(w, b_0) = \frac1n \sum_{i=1}^n \bigl[b(\eta_i) - y_i\eta_i\bigr] + \frac\lambda2\|w\|^2 .
$$
Với Bernoulli đây đúng là $L/n$ của A1, với Poisson là $\frac1n\sum(\mu_i - y_i\eta_i)$, với Gauss
là $\frac1{2n}\|y - \eta\|^2$ cộng hằng số. Với Gauss, $\phi = \sigma^2$ không ảnh hưởng nghiệm nên
lấy $\phi = 1$.

## 2. Thuật toán IRLS chung

Với liên kết bất kỳ, đặt trọng số làm việc và biến phản hồi làm việc
$$
s_i = \frac{(d\mu_i/d\eta_i)^2}{V(\mu_i)}, \qquad
z_i = \eta_i + \frac{y_i - \mu_i}{d\mu_i/d\eta_i}, \qquad S = \operatorname{diag}(s_i).
$$
Mỗi vòng IRLS (Fisher scoring) giải bài toán bình phương tối thiểu có trọng số, với
$\theta = [w;\, b_0]$ và $R = \operatorname{diag}(\lambda, \dots, \lambda, 0)$ để không phạt hệ số chặn:
$$
\Bigl(\tfrac1n \tilde X^\top S\tilde X + R\Bigr)\,\theta^{+} = \tfrac1n \tilde X^\top S\, z .
$$
Với liên kết chính tắc, $d\mu/d\eta = b''(\eta) = V(\mu)$ nên $s_i = V(\mu_i)$, và Hessian của
$\mathcal L$ là $\frac1n\tilde X^\top S\tilde X + R$, không phụ thuộc $y$. Do đó bước IRLS trùng với
bước Newton (chứng minh ở B3). Với Bernoulli, công thức này trở về đúng IRLS ở A1 mục 2.

**Hai điểm khi cài để ổn định số.**

1. *Không chia cho $d\mu/d\eta$.* Khi $\sigma(\eta)$ bão hòa thì $d\mu/d\eta \approx 0$ và $z_i$ bùng nổ. Vế
   phải được viết lại thành $\tilde X^\top[s \odot \eta + a \odot (y - \mu)]$ với
   $a_i = (d\mu_i/d\eta_i)/V(\mu_i)$; với liên kết chính tắc $a \equiv 1$ nên không có phép chia nào.
2. *Giảm nửa bước.* Newton thuần không bảo đảm loss giảm khi khởi tạo xa nghiệm. Nếu $\mathcal L$
   tăng sau một bước thì bước được chia đôi, tối đa 30 lần; nếu vẫn tăng thì thuật toán dừng và
   giữ nghiệm cũ (`stop_reason = "no_descent"`). Vì vậy $\mathcal L$ không tăng qua các vòng
   (sai lệch cho phép $10^{-12}$ do làm tròn).

Điều kiện dừng là $\|\nabla\mathcal L\| < 10^{-8}$ hoặc đủ 100 vòng. Khởi tạo $w = 0$,
$b_0 = g(\bar y)$, với $\bar y$ được chặn vào miền xác định của $g$ (ví dụ $[10^{-3}, 1 - 10^{-3}]$ cho
logit).

## 3. Cài đặt (`src/glm.py`)

Mỗi họ phân phối là một lớp con của `Family`, bắt buộc khai báo năm hàm: `inverse_link` ($g^{-1}$),
`link` ($g$, dùng khi khởi tạo), `dmu_deta`, `variance` ($V$) và `nll` (mất mát từng mẫu). Các hàm
tùy chọn là `check_y` (miền của $y$), `init_mean` (chặn $\bar y$), cùng `residual` và `irls_weights`
(bản ổn định số cho liên kết chính tắc). Thủ tục `fit_glm` chỉ gọi các hàm này, không kiểm tra tên
họ, nên thêm họ mới (ví dụ Gamma) chỉ cần viết thêm một lớp, không sửa thủ tục chung.

| Hàm | Đầu vào → đầu ra |
|---|---|
| `fit_glm(X, y, family, lam)` | $X$ $(n, d)$, $y$ $(n,)$ → $w$ $(d,)$, $b_0$, lịch sử loss, $\|\nabla\|$, lý do dừng |
| `loss_grad(w, b, X, y, family, lam)` | → loss (vô hướng), gradient $(d+1,)$ |
| `predict(X, w, b, family)` | → $\mu$ $(n,)$ |

Các biện pháp ổn định số khác:

- Bernoulli dùng `core.sigmoid`, `core.log_sigmoid`. Phần dư $y - \sigma(\eta)$ được tính thành
  $y\,\sigma(-\eta) - (1-y)\,\sigma(\eta)$ để không mất chính xác khi $\sigma(\eta) \approx 1$.
- Poisson chặn $\eta \le 700$ trước khi lấy mũ.
- Hệ phương trình suy biến thì chuyển sang `np.linalg.lstsq`.
- $y$ ngoài miền của họ (ví dụ $y = 2$ với Bernoulli, $y < 0$ với Poisson) gây `ValueError`.

## 4. Kiểm chứng

Dữ liệu mô phỏng $n = 300$, $d = 3$, seed $= 42$, sinh từ đúng mô hình của từng họ. Bộ
`tests/test_glm.py` gồm 43 test, đều đạt.

| Kiểm tra | Kết quả |
|---|---|
| Gradient so với sai phân hữu hạn (3 họ, $\lambda \in \{0;\ 0{,}1\}$) | khớp, sai số tương đối $< 10^{-5}$ |
| So với `statsmodels.GLM`, $\lambda = 0$: Bernoulli / Poisson / Gauss | lệch tham số tối đa $3{,}2\cdot10^{-12}$ / $5{,}9\cdot10^{-14}$ / $2{,}2\cdot10^{-16}$ |
| Bernoulli so với `logistic.fit_irls` (A2), $\lambda = 0$ / $0{,}1$ | $2{,}8\cdot10^{-11}$ / $3{,}4\cdot10^{-13}$ |
| Poisson, $\lambda = 0{,}1$, so với `sklearn` `PoissonRegressor` | $1{,}1\cdot10^{-16}$ |
| Gauss, $\lambda = 0{,}1$, so với `sklearn` `Ridge` ($\alpha = n\lambda$) | $1{,}7\cdot10^{-16}$ |
| Số vòng hội tụ: Bernoulli / Poisson / Gauss | 3 / 4 / 1 |
| Poisson, 4 điểm khởi tạo xa (loss ban đầu tới $\approx 3\cdot10^{13}$) | hội tụ sau 12–37 vòng, không tràn số |
| Dữ liệu khả tách, Bernoulli, $\lambda = 0$ | không sinh `nan` |
| Bước đi lên dốc (họ cố tình sai dấu) | dừng `no_descent`, giữ nghiệm cũ |

**Nhận xét.**

- Gauss với liên kết đồng nhất hội tụ đúng sau **1 vòng**: khi đó $S = I$, $z = y$, nên một
  bước IRLS chính là nghiệm bình phương tối thiểu.
- Bernoulli và Poisson hội tụ bậc hai: loss của Bernoulli giảm lần lượt $6{,}8\cdot10^{-3}$,
  $2{,}6\cdot10^{-6}$, $2{,}4\cdot10^{-12}$ (số mũ gần gấp đôi mỗi vòng), và sau 3–4 vòng
  $\|\nabla\mathcal L\| \approx 10^{-12}$, đúng tính chất của phương pháp Newton.
- Bernoulli cho cùng kết quả với IRLS nhị phân riêng ở A2, xác nhận một thủ tục IRLS chung đủ
  thay cho cài đặt riêng từng họ.
- Với Poisson khởi tạo xa, số vòng tăng vì xa nghiệm thì Newton chỉ hội tụ tuyến tính, gần
  nghiệm mới hội tụ bậc hai. Ở cả 4 điểm khởi tạo xa, bước IRLS đầy đủ luôn làm giảm loss
  (tắt giảm nửa bước cho cùng kết quả), nên cơ chế này chỉ là lớp bảo vệ cho trường hợp xấu hơn.
