# AETHER · 数学基础与出处（MATH）

> 作者：晨星 · 每个算法标注世界顶级数学/算法来源，并给出可验证不变量

## 1. 线性代数（core/linalg）

### 1.1 单侧 Jacobi SVD
- **公式**：对矩阵列做旋转正交化，`A' = UᵀA` 列正交 → `S = ‖列‖`，`U = 列/S`，`V` = 累积旋转。
- **结果**：`A = U·diag(S)·Vᵀ`，`U` 瘦长且列正交，`V` 正交。
- **出处**：Hestenes (1958)；Drmač 现代实现；与 LAPACK `xGEJSV` 同源。
  Golub & Van Loan, *Matrix Computations* (4th, 2013), §8.5。
- **不变量**：`max‖A − U·diag(S)·Vᵀ‖ ≤ 1e-7`；`UᵀU = I`；`VᵀV = I`。

### 1.2 对称特征值（Jacobi）
- **公式**：循环旋转消去非对角元，`A·vᵢ = λᵢ·vᵢ`，`V` 正交。
- **出处**：Golub & Van Loan §8.4.3。
- **不变量**：`‖A·vᵢ − λᵢvᵢ‖ ≈ 0`；`VᵀV = I`。

### 1.3 Cholesky
- **公式**：`A = L·Lᵀ`（A 对称正定）。
- **不变量**：`‖L·Lᵀ − A‖ ≤ 1e-9`。

### 1.4 Householder QR
- **公式**：`A = Q·R`，`Q = H₀H₁…H_{n−1}`，`Hₖ = I − 2vₖvₖᵀ`，`R` 上三角。
- **出处**：Golub & Van Loan §5.2。
- **不变量**：`QᵀQ = I`；`Q·R = A`；R 上三角。

## 2. 数值稳定（core/numerics）

### 2.1 logsumexp
- **公式**：`log Σ e^{xᵢ} = m + log Σ e^{xᵢ−m}`，`m = max xᵢ`。防止 exp 溢出。
- **出处**：Goodfellow et al. §3.2。

### 2.2 softmax
- **公式**：`σ(x)ᵢ = e^{xᵢ−m} / Σⱼ e^{xⱼ−m}`。输出为概率分布（和为 1，非负）。

## 3. 机器学习（ml/）

### 3.1 PCA（SVD 路线）
- **公式**：中心化 `X` → `X = U·S·Vᵀ` → 主成分 = `V` 的行（右奇异向量）。
- **出处**：Bishop, *PRML* §12.1；Jolliffe, *PCA* (2002)。
- **不变量**：主成分正交归一；全维重构误差 ≈ 0；`Σ 解释方差占比 = 1`。

### 3.2 GMM（EM）
- **E-step**：`γ(zₙₖ) = p(zₙₖ)·N(xₙ|μₖ,Σₖ) / Σₖ' …`（log-sum-exp 稳定）。
- **M-step**：以责任度为权重更新 `πₖ, μₖ, Σₖ`。
- **高斯对数密度**：经 Cholesky `Σ = L·Lᵀ`，`log N = −½(d log2π + 2Σlog Lᵢᵢ + ‖L⁻¹(x−μ)‖²)`。
- **出处**：Bishop §9.2；Murphy §11.4。
- **不变量（EM 定理）**：完整数据对数似然**单调不减**；责任度每行和 = 1；`Σπₖ = 1`。

### 3.3 k-means
- **公式**：Lloyd 迭代（指派 + 更新质心）；k-means++ 初始化。
- **出处**：Arthur & Vassilvitskii (2007)；Bishop §9.1。
- **不变量**：簇内平方和（inertia）**单调不增**。

### 3.4 逻辑回归
- **公式**：`max p(y|x)` ⇔ `min −Σ[y log σ(wᵀx+b) + (1−y)log(1−σ)]`；梯度 `∂ = (1/n)Σ(σ−y)·x`。
- **出处**：Bishop §4.3；Goodfellow §6.2。
- **不变量**：解析梯度 vs 中心差分 `maxRelErr ≈ 1e-7`；损失单调下降。

### 3.5 k-NN
- **不变量**：`k=1` 时训练集误差为 0（记忆性）；欧氏/余弦距离可选。

## 4. 深度学习（dl/）

### 4.1 自动微分（反向模式）
- **公式**：链式法则的反向遍历，`∂L/∂x = Σⱼ (∂L/∂yⱼ)(∂yⱼ/∂x)`。
- **出处**：Baydin et al. (2018), *Automatic Differentiation in Machine Learning*；Goodfellow §6.5。
- **不变量**：解析梯度 vs 数值梯度 `maxRelErr ≤ 1e-5`（MLP 全参数）。

### 4.2 MLP
- **前向**：`h = act(XW₁+b₁) → … → logits = hW_L+b_L`；Xavier/Glorot 初始化。
- **训练**：BCE + Adam（Kingma & Ba, 2015）。
- **出处**：Goodfellow §6, §8；Bishop §5。

### 4.3 Transformer 注意力
- **缩放点积注意力**：`Attention(Q,K,V) = softmax(QKᵀ/√dₖ)·V`。
- **多头**：并行 `h` 个注意力头后拼接，再经 `W_O`。
- **位置编码**：`PE(pos,2i)=sin(pos/10000^{2i/d})`，`PE(pos,2i+1)=cos(…)`。
- **Block**：`x → LayerNorm(x + Attn(x)) → LayerNorm(· + FFN(·))`。
- **出处**：Vaswani et al. (2017), *Attention Is All You Need*。
- **不变量**：注意力权重每行和 = 1（softmax）；位置编码确定可复现。

## 5. 优化（core/optim）
- **Adam**：`m ← β₁m + (1−β₁)g`，`v ← β₂v + (1−β₂)g²`，
  `θ ← θ − η·m̂/(√v̂ + ε)`。Kingma & Ba (2015)。
