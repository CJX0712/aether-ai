# AETHER · 世界顶级零依赖 AI/ML 引擎

> **架构最先进 · 性能最强 · 代码最优 · 文档最全**
> 站在世界顶级数学、算法与开源成果之上 —— 复用、组合、超越。
> **作者：晨星**

AETHER 是一个**零运行时依赖**、**数值严谨**、**可验证、可复现**的 AI/ML 引擎。
它把经典数值线性代数、统计机器学习与深度学习（含 Transformer）统一在同一套
可证明不变性（invariant）的底座之上，每条算法都有一条可独立交叉验证的数学不变量。

---

## ✦ 设计哲学（三大原则）

| 原则 | 含义 |
|------|------|
| **零重写** | 不造轮子。每个算法直接复用世界顶级数学/算法（Golub & Van Loan、Bishop、Goodfellow、Vaswani…），并标注出处。 |
| **集大成** | 每个模块都站在世界顶级成果之上；数值方法选取业界最稳方案（如 SVD 用单侧 Jacobi，而非协方差特征分解）。 |
| **闭环交付** | 完整开源、可运行、可复现、可验证（`npm test` 一键 33 项不变性全绿）。 |

---

## ✦ 能力矩阵

```
AETHER
├── core/          数值地基（世界顶级数值线性代数）
│   ├── matrix      矩阵运算
│   ├── linalg      Jacobi SVD · 对称特征值 · Cholesky · Householder QR · 线性方程组
│   ├── numerics    logsumexp · softmax · 激活 · 数值梯度
│   ├── random      可复现种子 RNG（mulberry32 + Box–Muller）
│   └── optim       梯度下降(动量) · Adam (Kingma & Ba, 2015)
├── ml/            经典机器学习（Bishop, PRML）
│   ├── pca         SVD-PCA
│   ├── gmm         EM-GMM（Cholesky 稳定对数密度）
│   ├── kmeans      Lloyd + k-means++ 初始化
│   ├── logistic    最大似然逻辑回归 + 梯度检验
│   └── knn         惰性 k-近邻（分类/回归）
└── dl/            深度学习
    ├── autograd    张量级反向模式自动微分（现代框架同源）
    ├── nn          MLP + Adam 训练
    └── attention   缩放点积注意力 · 多头注意力 · 正弦位置编码 · Transformer Block
```

---

## ✦ 快速开始

```bash
npm install        # 仅安装 typescript / @types/node（devDependencies）
npm run build      # tsc 编译到 dist/
npm test           # 编译 + 运行 33 项不变性测试（必须全绿）
```

```ts
import { pca, gmmEM, logisticRegression, MLP, trainMLP, multiHeadAttention } from "aether-ai";

// ---- PCA：主成分分析 ----
const model = pca(data, 2);          // data: number[][]
const Z = model.transform(data);     // 降维投影

// ---- 逻辑回归：二分类 ----
const lr = logisticRegression(X, y); // y ∈ {0,1}
const pred = lr.predict(X);

// ---- MLP + 自动微分训练 ----
const mlp = new MLP([2, 8, 1], "relu");
trainMLP(mlp, X, y, { lr: 0.03, epochs: 600 });

// ---- Transformer 多头注意力 ----
const { out, attn } = multiHeadAttention(X, 2, 8, proj); // attn 每行和为 1
```

---

## ✦ 验证总纲（为什么可信）

每个算法至少有 **1 条可被独立实现交叉验证的不变量**。运行 `npm test` 即验证：

| 模块 | 不变量 | 允许误差 |
|------|--------|----------|
| Jacobi SVD | `A = U·diag(S)·Vᵀ` 重构；`UᵀU = I`、`VᵀV = I` | ≤ 1e-7 |
| 对称特征值 | `A·v = λ·v` 残差；特征向量正交归一 | ≤ 1e-6 |
| Cholesky | `L·Lᵀ = A` | ≤ 1e-9 |
| Householder QR | `QᵀQ = I`；`Q·R = A`；R 上三角 | ≤ 1e-7 |
| PCA | 主成分正交归一；解释方差占比和 = 1 | ≤ 1e-9 |
| GMM (EM) | 责任度每行和 = 1；权重和 = 1；对数似然单调不减（EM 定理） | ≤ 1e-9 |
| k-means | inertia 单调不增；指派确定 | — |
| 逻辑回归 / MLP | 解析梯度 vs 中心差分 `maxRelErr ≈ 1e-7`；损失单调下降 | ≤ 1e-5 |
| 注意力 | 注意力权重每行和 = 1；位置编码确定可复现 | ≤ 1e-9 |

**最新结果：`33 通过 / 0 失败`。**

---

## ✦ 数学基础与出处

详见 [`MATH.md`](./MATH.md)；架构详见 [`ARCHITECTURE.md`](./ARCHITECTURE.md)。

- 线性代数：Golub & Van Loan, *Matrix Computations* (4th, 2013)；Demmel & Veselić (1992)
- 机器学习：Bishop, *Pattern Recognition and Machine Learning* (2006)
- 深度学习：Goodfellow, Bengio & Courville, *Deep Learning* (2016)
- 优化：Kingma & Ba, *Adam* (ICLR 2015)
- Transformer：Vaswani et al., *Attention Is All You Need* (NeurIPS 2017)

---

## ✦ 许可证 & 作者

- **License**：MIT
- **Author**：晨星

> 本引擎遵循「零重写 · 集大成 · 闭环交付」原则构建，所有数学与算法均复用并标注世界顶级成果。
