---
title: "A1. Gradient và Hessian của hồi quy logistic"
author: "Đức"
lang: vi
---

# A1. Gradient và Hessian của hồi quy logistic

Mục này dẫn hàm mất mát, gradient và Hessian cho hồi quy logistic nhị phân, hồi quy softmax
đa lớp và phiên bản có điều chuẩn L2. Đây là cơ sở cho phần cài đặt A2 (IRLS, hạ gradient
Armijo) và A3 (GLM).

## 1. Ký hiệu

| Ký hiệu | Ý nghĩa | Shape |
|---|---|---|
| $X$, hàng $x_i^\top$ | ma trận đặc trưng | $(n, d)$ |
| $y$, $y_i \in \{0,1\}$ / $Y$ one-hot | nhãn nhị phân / $K$ lớp | $(n,)$ / $(n, K)$ |
| $w, b$ / $W, b$ | trọng số, hệ số chặn | $(d,)$, vô hướng / $(d, K)$, $(K,)$ |
| $p = \sigma(z)$ / $P$ | xác suất dự đoán | $(n,)$ / $(n, K)$ |
| $\lambda \ge 0$ | hệ số điều chuẩn L2 | vô hướng |

Hệ số chặn được gộp bằng $\tilde X = [X,\ \mathbf 1]$, $\theta = [w;\, b]$; mọi công thức theo $w$
đúng cho $\theta$ khi thay $X$ bằng $\tilde X$. Công thức dưới đây viết dạng tổng; code dùng dạng
trung bình (chia phần dữ liệu cho $n$), tương đương khi thay $\lambda$ bằng $n\lambda$.

## 2. Hồi quy logistic nhị phân

**Hàm mất mát.** Với $z_i = w^\top x_i + b$ và $p_i = \sigma(z_i)$, log-hợp lý âm là
$-\sum_i [y_i \log p_i + (1-y_i)\log(1-p_i)]$. Dùng $\log\sigma(z) = z - \log(1+e^z)$ và
$\log(1-\sigma(z)) = -\log(1+e^z)$:
$$
L(w) = \sum_{i=1}^n \bigl[\log(1 + e^{z_i}) - y_i z_i\bigr].
$$
Dạng này không chứa $\log p_i$ nên không gặp $\log 0$ khi $p_i$ bão hòa; $\log(1+e^z)$ được
tính ổn định bằng `core.log_sigmoid` / `core.logsumexp`.

**Gradient.** Vì $\frac{d}{dz}\log(1+e^z) = \sigma(z)$, ta có $\partial L/\partial z_i = p_i - y_i$, nên
$$
\boxed{\nabla_w L = X^\top (p - y), \qquad \partial L/\partial b = \mathbf 1^\top (p - y).}
$$

**Hessian.** Vì $\sigma'(z) = \sigma(z)(1-\sigma(z))$, đặt $S = \operatorname{diag}(s_i)$ với
$s_i = p_i(1-p_i) \in (0, \tfrac14]$:
$$
\boxed{H = \nabla_w^2 L = X^\top S X.}
$$
Với mọi $v$, $v^\top H v = \sum_i s_i (x_i^\top v)^2 \ge 0$, nên $L$ lồi; nếu $X$ có hạng cột đầy
đủ thì $H \succ 0$ và $L$ lồi chặt. Khi dữ liệu khả tách tuyến tính, $L$ vẫn giảm mãi khi
$\|w\| \to \infty$ nên không có cực tiểu hữu hạn (B2, C3).

**Bước Newton là IRLS.** Thay gradient và Hessian vào $w^{+} = w - H^{-1}\nabla L$, rồi đưa
$w = (X^\top S X)^{-1}X^\top S X w$ vào trong ngoặc:
$$
w^{+} = w + (X^\top S X)^{-1} X^\top (y - p)
= (X^\top S X)^{-1} X^\top S\, u, \qquad u = Xw + S^{-1}(y - p).
$$
Mỗi bước Newton là một bài toán bình phương tối thiểu có trọng số $\min_w \sum_i s_i(u_i - x_i^\top w)^2$
với trọng số $S$ tính lại sau mỗi vòng, nên gọi là IRLS. Khi cài, giải hệ
$(X^\top S X)\Delta = X^\top(y - p)$ bằng `np.linalg.solve` thay vì nghịch đảo ma trận, và không
chia cho $s_i \approx 0$.

## 3. Hồi quy softmax $K$ lớp

**Hàm mất mát.** Với $Z = XW + \mathbf 1 b^\top$ và
$\operatorname{LSE}(z) = \log\sum_j e^{z_j}$, xác suất là $p_{ik} = \exp(z_{ik} - \operatorname{LSE}(z_i))$.
Do $\sum_k y_{ik} = 1$:
$$
L(W, b) = \sum_{i=1}^n \bigl[\operatorname{LSE}(z_i) - y_i^\top z_i\bigr].
$$
$\operatorname{LSE}$ tính bằng `core.logsumexp` (trừ $\max_j z_{ij}$ trước khi lấy mũ) để không tràn số.

**Gradient.** Từ $\partial \operatorname{LSE}/\partial z_k = p_k$, ta có $\partial L/\partial Z = P - Y$, nên
$$
\boxed{\nabla_W L = X^\top (P - Y) \in \mathbb R^{d\times K}, \qquad \nabla_b L = (P - Y)^\top \mathbf 1.}
$$
Công thức có cùng dạng "đặc trưng$^\top$ × (dự đoán − nhãn)" như trường hợp nhị phân.

**Hessian.** Từ $\partial p_k/\partial z_l = p_k(\delta_{kl} - p_l)$, với
$A_i = \operatorname{diag}(p_i) - p_i p_i^\top$ và $\operatorname{vec}(W) = [w_1; \dots; w_K]$:
$$
\boxed{H = \sum_{i=1}^n A_i \otimes x_i x_i^\top \in \mathbb R^{dK \times dK}.}
$$

- *Lồi:* $a^\top A_i a = \operatorname{Var}_{k \sim p_i}(a_k) \ge 0$, nên $H \succeq 0$.
- *Không lồi chặt:* $A_i \mathbf 1 = 0$, nên mọi hướng $\mathbf 1_K \otimes c$ nằm trong không gian rỗng
  của $H$. Cộng cùng một vector $c$ vào mọi $w_k$ không đổi xác suất: tham số softmax không định
  danh được.
- *Kiểm tra $K = 2$:* $p_{i2} = \sigma\bigl((w_2 - w_1)^\top x_i + b_2 - b_1\bigr)$, trùng mô hình nhị
  phân với $w = w_2 - w_1$.

Với TF-IDF, $d$ rất lớn nên $H$ kích thước $3d \times 3d$ không lập được; vì vậy A2 dùng hạ
gradient với tìm bước Armijo thay cho Newton.

## 4. Điều chuẩn L2

Phạt trọng số, **không phạt hệ số chặn** (phạt $b$ làm lệch xác suất trung bình):
$L_\lambda = L + \frac{\lambda}{2}\|w\|^2$ (nhị phân), $L_\lambda = L + \frac{\lambda}{2}\|W\|_F^2$ (softmax).

| | Gradient | Hessian |
|---|---|---|
| Nhị phân | $X^\top(p - y) + \lambda w$ | $X^\top S X + \lambda I_d$ |
| Softmax | $X^\top(P - Y) + \lambda W$ | $\sum_i A_i \otimes x_i x_i^\top + \lambda I_{dK}$ |
| Hệ số chặn | không đổi | không thêm $\lambda$ |

Hệ quả khi $\lambda > 0$:

- Hessian theo trọng số $\succeq \lambda I \succ 0$: nghiệm tồn tại và duy nhất, kể cả khi dữ liệu
  khả tách hoặc tham số softmax không định danh, **miễn là mỗi lớp đều xuất hiện trong dữ liệu**.
  Vì $b$ không bị phạt, nếu một lớp vắng mặt thì loss vẫn giảm mãi khi hệ số chặn của lớp đó
  $\to -\infty$ (ví dụ $y$ toàn 0 với $\lambda = 1$: IRLS đẩy $b$ tới $-18{,}9$ rồi dừng chỉ vì
  gradient đã nhỏ hơn ngưỡng). Riêng $b$ của softmax vẫn sai khác một hằng số cộng chung, nhưng
  xác suất $P$ là duy nhất.
- Hệ Newton $(X^\top S X + \lambda I)\Delta = -\nabla L_\lambda$ luôn giải được, trị riêng nhỏ nhất ít nhất $\lambda$.
- Tại nghiệm, $W^\star = -\frac1\lambda X^\top(P - Y)$; vì mỗi hàng $P - Y$ có tổng bằng 0 nên
  $\sum_k w_k^\star = 0$: L2 tự chọn một nghiệm định danh.

## 5. Chặn bước cho hạ gradient

Nếu $\nabla^2 L \preceq M I$ thì gradient $M$-Lipschitz và bước $t \le 1/M$ luôn làm giảm hàm.
Vì $s_i \le \tfrac14$ và trị riêng lớn nhất của $A_i$ không quá $\tfrac12$ (do
$a^\top A_i a = \operatorname{Var}_{k\sim p_i}(a_k) \le \tfrac14(a_{\max} - a_{\min})^2 \le \tfrac12\|a\|^2$):
$$
M_{\text{nhị phân}} \le \tfrac14\|X\|_2^2 + \lambda, \qquad M_{\text{softmax}} \le \tfrac12\|X\|_2^2 + \lambda.
$$
A2 không tính $M$ trực tiếp mà dùng tìm bước quay lui Armijo: giảm $t \leftarrow 0{,}5\,t$ đến khi
$L_\lambda(W - tG) \le L_\lambda(W) - c\,t\|G\|_F^2$ với $c = 10^{-4}$. Điều kiện chắc chắn thỏa khi
$t \le 2(1-c)/M$, nên vòng quay lui luôn dừng.

## 6. Kiểm chứng số

Gradient và Hessian được so với sai phân hữu hạn trung tâm ($\varepsilon = 10^{-5}$) trên dữ liệu
ngẫu nhiên $n = 50$, $d = 4$, $K = 3$, $\lambda = 0{,}1$, seed $= 42$:

| Kiểm tra | Sai số tương đối |
|---|---|
| Nhị phân: gradient / Hessian | $7{,}2\cdot10^{-11}$ / $4{,}4\cdot10^{-11}$ |
| Softmax: gradient / Hessian | $1{,}0\cdot10^{-10}$ / $3{,}9\cdot10^{-11}$ |
| $K = 2$ trùng nhị phân (xác suất) | $1{,}7\cdot10^{-16}$ |

Hessian softmax theo $(W, b)$ với $\lambda = 0$ có đúng $d + 1 = 5$ trị riêng bằng 0 (các hướng
$\mathbf 1_K \otimes c$); với $\lambda = 0{,}1$ chỉ còn 1, đúng hướng hệ số chặn không bị phạt.
Kết quả khớp với phân tích ở mục 3–4.
