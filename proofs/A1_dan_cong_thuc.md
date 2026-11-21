---
title: "A1. Dẫn công thức gradient và Hessian cho hồi quy logistic"
author: "Đức"
lang: vi
---

# A1. Dẫn công thức gradient và Hessian

Mục này dẫn đầy đủ hàm mất mát, gradient và Hessian cho ba mô hình được cài ở A2:
hồi quy logistic nhị phân (dùng cho Newton/IRLS), hồi quy softmax đa lớp, và phiên bản
có điều chuẩn L2 của cả hai. Kết quả cuối mỗi mục được viết ở dạng ma trận để cài trực
tiếp bằng NumPy.

## 1. Ký hiệu

| Ký hiệu | Ý nghĩa | Shape |
|---|---|---|
| $X$, hàng $x_i^\top$ | ma trận đặc trưng | $(n, d)$ |
| $y$, $y_i \in \{0, 1\}$ | nhãn nhị phân | $(n,)$ |
| $Y$, $y_{ik} \in \{0,1\}$, $\sum_k y_{ik} = 1$ | nhãn one-hot $K$ lớp | $(n, K)$ |
| $w$, $b$ | trọng số, hệ số chặn (nhị phân) | $(d,)$, vô hướng |
| $W = [w_1, \dots, w_K]$, $b$ | trọng số, hệ số chặn (softmax) | $(d, K)$, $(K,)$ |
| $p$ / $P$ | xác suất dự đoán | $(n,)$ / $(n, K)$ |
| $\lambda \ge 0$ | hệ số điều chuẩn L2 | vô hướng |

Thứ tự lớp theo `core.LABELS`: lừa đảo $= 0$, rác $= 1$, hợp lệ $= 2$.

**Gộp hệ số chặn.** Khi cần viết gọn, đặt $\tilde x_i = [x_i;\, 1] \in \mathbb{R}^{d+1}$,
$\tilde X = [X,\ \mathbf{1}]$ và $\theta = [w;\, b]$, khi đó $w^\top x_i + b = \theta^\top \tilde x_i$.
Mọi công thức theo $w$ dưới đây đúng nguyên dạng cho $\theta$ khi thay $X$ bằng $\tilde X$.

**Hai đạo hàm cơ bản.** Với $\sigma(z) = 1/(1 + e^{-z})$:
$$
\sigma'(z) = \frac{e^{-z}}{(1+e^{-z})^2} = \sigma(z)\,\bigl(1 - \sigma(z)\bigr),
\qquad
\frac{d}{dz}\log\bigl(1 + e^{z}\bigr) = \frac{e^z}{1+e^z} = \sigma(z).
$$

## 2. Hồi quy logistic nhị phân

### 2.1. Mô hình và hàm mất mát

Mô hình: $P(y_i = 1 \mid x_i) = p_i = \sigma(z_i)$ với $z_i = w^\top x_i + b$.

Hàm hợp lý của một mẫu là $p_i^{y_i}(1-p_i)^{1-y_i}$. Hàm mất mát là log-hợp lý âm
(cross-entropy):
$$
L(w) = -\sum_{i=1}^{n} \Bigl[\, y_i \log p_i + (1 - y_i)\log(1 - p_i) \Bigr].
$$

Dùng $\log \sigma(z) = -\log(1 + e^{-z})$ và $\log(1 - \sigma(z)) = -\log(1 + e^{z})$:
$$
y\log\sigma(z) + (1-y)\log(1-\sigma(z))
= -y\log(1+e^{-z}) - (1-y)\log(1+e^{z}).
$$
Vì $\log(1+e^{-z}) = \log(1+e^{z}) - z$, biểu thức trên bằng $y z - \log(1+e^{z})$. Do đó
$$
\boxed{\,L(w) = \sum_{i=1}^{n} \Bigl[\log\bigl(1 + e^{z_i}\bigr) - y_i z_i\Bigr]\,}
$$
Dạng này không chứa $\log p_i$ nên không bị $\log 0$ khi $p_i$ bão hòa. Khi cài đặt, tính
$\log(1+e^{z}) = \operatorname{logsumexp}(0, z)$ bằng `core.logsumexp`, không gọi `np.exp`
trực tiếp trên $z$.

### 2.2. Gradient

Theo quy tắc chuỗi, với $\partial z_i / \partial w = x_i$:
$$
\frac{\partial L}{\partial z_i} = \sigma(z_i) - y_i = p_i - y_i,
\qquad
\nabla_w L = \sum_{i=1}^n (p_i - y_i)\, x_i .
$$
Dạng ma trận:
$$
\boxed{\,\nabla_w L = X^\top (p - y) \in \mathbb{R}^{d},
\qquad
\frac{\partial L}{\partial b} = \mathbf{1}^\top (p - y) = \sum_{i=1}^n (p_i - y_i)\,}
$$

### 2.3. Hessian

Lấy đạo hàm $\nabla_w L = \sum_i (p_i - y_i) x_i$ thêm một lần theo $w^\top$, chỉ $p_i$ phụ
thuộc $w$, và $\partial p_i / \partial w^\top = p_i(1-p_i)\, x_i^\top$:
$$
H = \nabla_w^2 L = \sum_{i=1}^n p_i(1 - p_i)\, x_i x_i^\top .
$$
Đặt $S = \operatorname{diag}(s_1, \dots, s_n)$ với $s_i = p_i(1-p_i) \in (0, \tfrac14]$:
$$
\boxed{\,H = X^\top S X \in \mathbb{R}^{d \times d}\,}
$$
Khi gộp hệ số chặn, $H_\theta = \tilde X^\top S \tilde X$, trong đó góc dưới phải là
$\sum_i s_i$ và cột cuối là $X^\top S \mathbf 1$.

**Tính lồi.** Với mọi $v \in \mathbb{R}^d$:
$$
v^\top H v = \sum_{i=1}^n s_i\, (x_i^\top v)^2 \ \ge\ 0 ,
$$
nên $H \succeq 0$ và $L$ lồi. Nếu $X$ có hạng cột đầy đủ thì $Xv \neq 0$ với mọi $v \ne 0$,
và vì $s_i > 0$ nên $v^\top H v > 0$: $H \succ 0$, $L$ lồi chặt. Tuy vậy, khi dữ liệu khả tách
tuyến tính, $L$ giảm mãi khi $\|w\| \to \infty$ và không có cực tiểu hữu hạn (xem C3).

### 2.4. Bước Newton và dạng IRLS

Bước Newton: $w^{(t+1)} = w^{(t)} - H^{-1} \nabla_w L$. Thay công thức ở 2.2–2.3:
$$
w^{(t+1)} = w^{(t)} + (X^\top S X)^{-1} X^\top (y - p)
= (X^\top S X)^{-1} X^\top S \Bigl[ X w^{(t)} + S^{-1}(y - p) \Bigr].
$$
Đặt **biến phụ thuộc hiệu chỉnh** $u = X w^{(t)} + S^{-1}(y - p)$, tức
$u_i = z_i + (y_i - p_i)/s_i$, ta được
$$
\boxed{\,w^{(t+1)} = (X^\top S X)^{-1} X^\top S\, u\,}
$$
Đây chính là nghiệm bình phương tối thiểu có trọng số $\min_w \sum_i s_i (u_i - x_i^\top w)^2$.
Mỗi bước Newton là một bài toán bình phương tối thiểu với trọng số $S$ tính lại theo $w^{(t)}$,
nên thuật toán được gọi là IRLS (iteratively reweighted least squares). Khi cài, giải hệ
$(X^\top S X)\,\Delta = X^\top (y - p)$ bằng `np.linalg.solve` thay vì nghịch đảo ma trận,
và tránh chia cho $s_i \approx 0$ bằng cách dùng dạng $w + \Delta$.

## 3. Hồi quy softmax ($K$ lớp)

### 3.1. Mô hình và hàm mất mát

Logit của mẫu $i$: $z_i = W^\top x_i + b \in \mathbb{R}^K$, tức $z_{ik} = w_k^\top x_i + b_k$.
Dạng ma trận: $Z = XW + \mathbf{1} b^\top \in \mathbb{R}^{n \times K}$.

Xác suất:
$$
p_{ik} = \frac{e^{z_{ik}}}{\sum_{j=1}^K e^{z_{ij}}}
= \exp\Bigl(z_{ik} - \operatorname{LSE}(z_i)\Bigr),
\qquad
\operatorname{LSE}(z) = \log \sum_{j=1}^K e^{z_j}.
$$

Cross-entropy với nhãn one-hot:
$$
L(W, b) = -\sum_{i=1}^n \sum_{k=1}^K y_{ik} \log p_{ik}
= -\sum_{i=1}^n \sum_{k=1}^K y_{ik}\bigl(z_{ik} - \operatorname{LSE}(z_i)\bigr).
$$
Vì $\sum_k y_{ik} = 1$:
$$
\boxed{\,L(W, b) = \sum_{i=1}^n \Bigl[\operatorname{LSE}(z_i) - y_i^\top z_i\Bigr]\,}
$$
Khi cài, $\operatorname{LSE}$ tính bằng `core.logsumexp` (trừ $\max_j z_{ij}$ trước khi lấy mũ)
để không tràn số; $P$ tính bằng $\exp(Z - \operatorname{LSE})$, mỗi hàng tổng bằng 1.

### 3.2. Hai đạo hàm cơ bản

$$
\frac{\partial \operatorname{LSE}(z)}{\partial z_k} = \frac{e^{z_k}}{\sum_j e^{z_j}} = p_k ,
\qquad
\frac{\partial p_k}{\partial z_l} = p_k\,(\delta_{kl} - p_l),
$$
trong đó $\delta_{kl} = 1$ nếu $k = l$, ngược lại bằng $0$. Đạo hàm thứ hai suy ra từ
$p_k = e^{z_k} / \sum_j e^{z_j}$ theo quy tắc thương:
với $l = k$ được $p_k - p_k^2$, với $l \ne k$ được $-p_k p_l$.
Viết gọn cho vector: $\dfrac{\partial p}{\partial z^\top} = \operatorname{diag}(p) - p p^\top$.

### 3.3. Gradient

Theo logit của mẫu $i$:
$$
\frac{\partial L}{\partial z_{ik}} = p_{ik} - y_{ik}
\quad\Longleftrightarrow\quad
\frac{\partial L}{\partial Z} = P - Y \in \mathbb{R}^{n \times K}.
$$
Vì $z_{ik} = w_k^\top x_i + b_k$, nên $\partial z_{ik}/\partial w_k = x_i$ và $z_{ik}$ không phụ
thuộc $w_l$ với $l \ne k$:
$$
\nabla_{w_k} L = \sum_{i=1}^n (p_{ik} - y_{ik})\, x_i,
\qquad
\frac{\partial L}{\partial b_k} = \sum_{i=1}^n (p_{ik} - y_{ik}).
$$
Xếp $K$ cột lại thành ma trận:
$$
\boxed{\,\nabla_W L = X^\top (P - Y) \in \mathbb{R}^{d \times K},
\qquad
\nabla_b L = (P - Y)^\top \mathbf{1} \in \mathbb{R}^{K}\,}
$$
Công thức có cùng dạng "đặc trưng$^\top$ × (dự đoán − nhãn)" như trường hợp nhị phân.

### 3.4. Hessian

Dùng $\operatorname{vec}(W) = [w_1; w_2; \dots; w_K] \in \mathbb{R}^{dK}$ (xếp các cột nối tiếp).
Lấy đạo hàm $\nabla_{w_k} L$ theo $w_l^\top$, dùng 3.2 và $\partial z_{il}/\partial w_l = x_i$:
$$
\frac{\partial^2 L}{\partial w_k\, \partial w_l^\top}
= \sum_{i=1}^n \frac{\partial p_{ik}}{\partial z_{il}}\, x_i x_i^\top
= \sum_{i=1}^n p_{ik}(\delta_{kl} - p_{il})\, x_i x_i^\top
\in \mathbb{R}^{d \times d}.
$$
Gom $K \times K$ khối lại bằng tích Kronecker, với $A_i = \operatorname{diag}(p_i) - p_i p_i^\top \in \mathbb{R}^{K\times K}$:
$$
\boxed{\,H = \nabla^2_{\operatorname{vec}(W)} L = \sum_{i=1}^n A_i \otimes \bigl(x_i x_i^\top\bigr) \in \mathbb{R}^{dK \times dK}\,}
$$
Khi gộp hệ số chặn, thay $x_i$ bằng $\tilde x_i$. Với $K = 3$ và $d$ lớn (TF-IDF), $H$ có
kích thước $3d \times 3d$, quá lớn để lập trực tiếp, nên A2 dùng hạ gradient có Armijo
thay cho Newton. Hessian vẫn cần để chứng minh tính lồi và chặn bước (mục 5).

**Tính lồi.** Với $a \in \mathbb{R}^K$ và phân phối $p_i$:
$$
a^\top A_i a = \sum_k p_{ik} a_k^2 - \Bigl(\sum_k p_{ik} a_k\Bigr)^2 = \operatorname{Var}_{k \sim p_i}(a_k) \ \ge\ 0 .
$$
Vì tích Kronecker của hai ma trận nửa xác định dương là nửa xác định dương, và tổng các
ma trận nửa xác định dương cũng vậy, $H \succeq 0$ và $L$ lồi.

**$H$ suy biến.** $A_i \mathbf{1} = p_i - p_i (p_i^\top \mathbf{1}) = 0$, nên với mọi
$c \in \mathbb{R}^d$, $v = \mathbf{1}_K \otimes c$ thỏa $H v = 0$. Ý nghĩa: thay mọi $w_k$ bằng
$w_k + c$ (và $b_k$ bằng $b_k + c_0$) cộng cùng một lượng $c^\top x_i + c_0$ vào mọi logit của
mẫu $i$, không làm đổi $p_i$. Tham số softmax không định danh được, cực tiểu (nếu có) không duy
nhất, và $L$ không lồi chặt. Khi gộp hệ số chặn, không gian rỗng của $H$ có số chiều ít nhất
$d + 1$. Điều chuẩn L2 ở mục 4 khắc phục điều này cho $W$.

### 3.5. Kiểm tra: $K = 2$ trùng với logistic nhị phân

Với $K = 2$: $p_{i2} = e^{z_{i2}} / (e^{z_{i1}} + e^{z_{i2}}) = \sigma(z_{i2} - z_{i1})$.
Đặt $w = w_2 - w_1$ và $b = b_2 - b_1$ thì $p_{i2} = \sigma(w^\top x_i + b)$, đúng mô hình mục 2.
Gradient theo $w_2$ là $X^\top(p_{\cdot 2} - y_{\cdot 2})$, khớp với $X^\top(p - y)$. Khi
$K = 2$, $A_i = p_{i2}(1-p_{i2}) \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}$, nên Hessian
theo hướng $w_2 - w_1$ cho lại $X^\top S X$.

## 4. Điều chuẩn L2

Phạt bình phương chuẩn của trọng số, **không phạt hệ số chặn** (hệ số chặn chỉ dịch tỷ lệ
cơ sở của lớp, phạt nó làm lệch xác suất trung bình):
$$
L_\lambda(w, b) = L(w, b) + \frac{\lambda}{2}\|w\|_2^2,
\qquad
L_\lambda(W, b) = L(W, b) + \frac{\lambda}{2}\|W\|_F^2 .
$$
Vì $\nabla \bigl(\tfrac{\lambda}{2}\|w\|^2\bigr) = \lambda w$ và $\nabla^2 \bigl(\tfrac{\lambda}{2}\|w\|^2\bigr) = \lambda I$:

| | Gradient | Hessian |
|---|---|---|
| Nhị phân | $X^\top(p - y) + \lambda w$ | $X^\top S X + \lambda I_d$ |
| Softmax | $X^\top(P - Y) + \lambda W$ | $\sum_i A_i \otimes x_i x_i^\top + \lambda I_{dK}$ |
| Hệ số chặn | không đổi | không thêm $\lambda$ ở khối của $b$ |

Khi gộp hệ số chặn, phần thêm vào Hessian là $\lambda \Lambda$ với
$\Lambda = \operatorname{diag}(1, \dots, 1, 0)$ (nhị phân) hoặc $I_K \otimes \Lambda$ (softmax).

**Hệ quả.** Với $\lambda > 0$, phần Hessian theo trọng số thỏa
$\nabla^2_w L_\lambda \succeq \lambda I \succ 0$, nên $L_\lambda$ lồi chặt (mạnh) theo trọng số:

- Cực tiểu tồn tại và duy nhất theo trọng số, kể cả khi dữ liệu khả tách (nhị phân) hoặc
  khi tham số softmax không định danh (mục 3.4).
- Riêng softmax: vì $b$ không bị phạt, cộng cùng một hằng số vào mọi $b_k$ vẫn không đổi
  $L_\lambda$, nên Hessian đầy đủ theo $(W, b)$ còn đúng một hướng suy biến
  $\mathbf 1_K \otimes e_{d+1}$. $W^\star$ duy nhất, $b^\star$ duy nhất sai khác một hằng số; xác
  suất $P$ vẫn duy nhất. Khi cài có thể chuẩn hóa $b \leftarrow b - \bar b$ (trừ trung bình)
  sau mỗi bước, không ảnh hưởng kết quả.
- Hệ Newton $(X^\top S X + \lambda I)\Delta = -\nabla L_\lambda$ luôn giải được và ổn định hơn,
  vì trị riêng nhỏ nhất của ma trận ít nhất bằng $\lambda$.
- Tại cực tiểu, $\nabla_W L_\lambda = 0$ cho $W^\star = -\tfrac{1}{\lambda} X^\top (P - Y)$, và
  $\nabla_b L = 0$ cho $\sum_i (P - Y) = 0$, tức trung bình xác suất dự đoán mỗi lớp bằng tỷ lệ
  lớp đó trong dữ liệu huấn luyện. Từ đó $W^\star \mathbf 1_K = -\tfrac1\lambda X^\top (P - Y)\mathbf 1_K = 0$
  vì mỗi hàng của $P - Y$ tổng bằng 0: L2 tự chọn nghiệm có $\sum_k w_k^\star = 0$.

**Dạng trung bình.** Nếu dùng $\bar L = \tfrac{1}{n} L + \tfrac{\lambda}{2}\|W\|_F^2$ (thang
đo không phụ thuộc $n$, tiện khi chọn $\lambda$ bằng kiểm định chéo), mọi gradient và Hessian
của phần dữ liệu được chia cho $n$, phần điều chuẩn giữ nguyên. Hai dạng tương đương khi thay
$\lambda$ bằng $n\lambda$. Code và báo cáo phải ghi rõ dùng dạng nào.

## 5. Hằng số Lipschitz của gradient và bước Armijo

Hạ gradient ở A2 cần biết bước tối đa an toàn. Một hàm có $\nabla^2 L \preceq M I$ thì gradient
$M$-Lipschitz, và bước $t \le 1/M$ luôn làm giảm hàm mất mát.

- **Nhị phân:** $s_i = p_i(1-p_i) \le \tfrac14$, nên
  $X^\top S X \preceq \tfrac14 X^\top X$ và $M \le \tfrac14 \|X\|_2^2 + \lambda$.
- **Softmax:** trị riêng lớn nhất của $A_i$ không quá $\tfrac12$
  (vì $a^\top A_i a = \operatorname{Var}_{k\sim p_i}(a_k) \le \tfrac12 \|a\|^2$, chứng minh
  ở cuối mục). Do đó
  $H \preceq \tfrac12 I_K \otimes X^\top X$ và $M \le \tfrac12 \|X\|_2^2 + \lambda$.

Ở đây $\|X\|_2$ là chuẩn phổ (giá trị kỳ dị lớn nhất). Thay vì tính $M$, A2 dùng tìm bước
quay lui Armijo: bắt đầu từ $t = t_0$, nhân $t \leftarrow \beta t$ (thường $\beta = 0{,}5$) đến khi
$$
L_\lambda(W - t\, G) \ \le\ L_\lambda(W) - c\, t\, \|G\|_F^2,
\qquad G = \nabla_W L_\lambda,\ \ c = 10^{-4}.
$$
Do gradient $M$-Lipschitz, điều kiện chắc chắn thỏa khi $t \le 2(1-c)/M$, nên vòng lặp quay lui
luôn dừng sau hữu hạn bước.

*Chứng minh chặn $\tfrac12$:* với $\mu = \sum_k p_k a_k$,
$\sum_k p_k (a_k - \mu)^2 \le \sum_k p_k (a_k - m)^2$ với mọi $m$. Chọn
$m = (a_{\max} + a_{\min})/2$ được phương sai $\le (a_{\max} - a_{\min})^2/4 \le \|a\|^2 / 2$,
vì $(a_{\max} - a_{\min})^2 \le 2(a_{\max}^2 + a_{\min}^2) \le 2\|a\|^2$.

## 6. Tóm tắt để cài đặt

| Đại lượng | Nhị phân | Softmax | Shape (softmax) |
|---|---|---|---|
| Logit | $z = Xw + b$ | $Z = XW + \mathbf 1 b^\top$ | $(n, K)$ |
| Xác suất | $p = \sigma(z)$ | $P = \exp(Z - \operatorname{LSE}(Z))$ | $(n, K)$ |
| Mất mát | $\sum \log(1+e^{z}) - y^\top z$ | $\sum_i \operatorname{LSE}(z_i) - \sum_{ik} Y_{ik} Z_{ik}$ | vô hướng |
| Gradient $w$/$W$ | $X^\top(p - y) + \lambda w$ | $X^\top(P - Y) + \lambda W$ | $(d, K)$ |
| Gradient $b$ | $\sum_i (p_i - y_i)$ | $(P - Y)^\top \mathbf 1$ | $(K,)$ |
| Hessian | $X^\top S X + \lambda I$ | $\sum_i A_i \otimes x_i x_i^\top + \lambda I$ | $(dK, dK)$ |

Hàm dùng chung: `core.sigmoid` cho $\sigma$, `core.logsumexp` cho $\log(1+e^z)$ và $\operatorname{LSE}$.

## 7. Kiểm chứng số

Các công thức trên được kiểm tra bằng sai phân hữu hạn trung tâm
$\bigl(L(\theta + \varepsilon e_j) - L(\theta - \varepsilon e_j)\bigr)/(2\varepsilon)$ với
$\varepsilon = 10^{-5}$, trên dữ liệu ngẫu nhiên $n = 50$, $d = 4$, $K = 3$, $\lambda = 0{,}1$,
seed $= 42$ (không dùng dữ liệu tin nhắn):

| Kiểm tra | Sai số tương đối |
|---|---|
| Nhị phân, gradient | 7.2e-11 |
| Nhị phân, Hessian | 4.4e-11 |
| Softmax, gradient | 1.0e-10 |
| Softmax, Hessian | 3.9e-11 |
| K = 2 trùng nhị phân (xác suất) | 1.7e-16 |

Hessian softmax theo $(W, b)$: với $\lambda = 0$ có 5 trị riêng bằng 0, đúng bằng $d + 1 = 5$ (các hướng $\mathbf 1_K \otimes c$, mục 3.4); với $\lambda = 0{,}1$ chỉ còn 1 trị riêng bằng 0, đúng hướng $\mathbf 1_K \otimes e_{d+1}$ của hệ số chặn không bị phạt (mục 4); trị riêng nhỏ nhất khác 0 là 0.100.

Sai số tương đối cỡ $10^{-10}$, khớp với độ chính xác của sai phân trung tâm, xác nhận
gradient và Hessian ở mục 2–4.
