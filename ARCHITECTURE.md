# AETHER · 架构说明（ARCHITECTURE）

> 作者：晨星 · 零依赖、数值严谨、可验证的 AI/ML 引擎架构

## 1. 总体架构

AETHER 采用**分层单向依赖**的清晰架构，每一层只依赖其下层，无循环依赖：

```
        ┌─────────────────────────────────────────┐
   应用 │  示例 / 集成测试 / 不变性验证 (tests/)     │
        └───────────────────┬─────────────────────┘
                            │
        ┌───────────────────┴─────────────────────┐
   深度学习│  dl/autograd → dl/nn → dl/attention      │
        └───────────────────┬─────────────────────┘
                            │
        ┌───────────────────┴─────────────────────┐
   经典 ML│  ml/pca · ml/gmm · ml/kmeans · ml/logistic │
          │  · ml/knn                                │
        └───────────────────┬─────────────────────┘
                            │
        ┌───────────────────┴─────────────────────┐
   数值地基│  core/linalg · core/matrix · core/optim   │
          │  · core/numerics · core/random           │
        └───────────────────────────────────────────┘
```

## 2. 数据契约

- **矩阵** `Mat = number[][]`（行优先）。可读、零拷贝、调试友好。
- **向量** `Vec = number[]`。
- **张量** `Tensor { data: Mat; grad: Mat; _backward; _prev }` —— 自动微分的计算图节点。

设计取舍：小规模（≤ 数千维）性能足够；以**正确性、可读性、可验证性**优先，
契合 forge 系列「浏览器内一键自检」的哲学。

## 3. 数值层（core）设计要点

| 组件 | 选型理由（集大成） |
|------|--------------------|
| **单侧 Jacobi SVD** | 与 LAPACK `xGEJSV` 同源，是中小规模**精度最高**的 SVD；列正交化自然保证 `U` 正交。优于「协方差矩阵 + 特征分解」（会平方条件数）。 |
| **Jacobi 特征值** | 经典对称特征算法，数值稳定，特征向量天然正交。 |
| **Cholesky** | 正定矩阵求逆/解线性方程组的标准分解；GMM 用它做稳定对数密度（避免显式求逆）。 |
| **Householder QR** | 数值稳定的正交三角化；`Q` 正交、`R` 上三角，用于最小二乘与数值诊断。 |
| **可复现 RNG** | mulberry32 + Box–Muller；**随机种子即实验契约**，保证结果可复现。 |
| **Adam** | Kingma & Ba (2015) 自适应优化器，业界标准。 |

## 4. 自动微分（dl/autograd）设计要点

反向模式（reverse-mode）自动微分，与现代深度学习框架同源思想：

1. **前向**构建计算图：每个 `Tensor` 记录 `_prev`（父节点）与 `_backward`（局部雅可比累积闭包）。
2. **反向**拓扑排序遍历，自顶向下累积梯度。
3. 支持：`matmul`、`广播加减`、`逐元素乘`、`转置`、`relu/sigmoid/tanh`、`行级 softmax`、`均值/求和`、`log`。

梯度正确性由**中心差分数值梯度**交叉验证（`maxRelErr ≈ 1e-7`）。

## 5. 深度学习层组合

- **MLP** = 自动微分 + 线性层 + 激活，端到端可训练（BCE + Adam）。
- **Transformer** = 缩放点积注意力 → 多头注意力 → 正弦位置编码 →
  Post-LN Block（注意力子层 + 前馈子层）。注意力权重行和为 1（softmax 不变量）。

## 6. 可验证性（闭环交付核心）

所有不变量集中在 `src/tests/run.ts`，零依赖运行。CI / 本地均执行 `npm test`，
**全绿即代表每个数学断言成立**。这是「可复现、可验证」的硬证据，而非口头保证。

## 7. 目录结构

```
aether-ai/
├── src/
│   ├── core/      matrix · linalg · numerics · random · optim
│   ├── ml/        pca · gmm · kmeans · logistic · knn
│   ├── dl/        autograd · nn · attention
│   ├── tests/     run.ts（不变性测试套件）
│   └── index.ts   公共 API 出口
├── dist/          编译产物（gitignore）
├── package.json   scripts: build / test / verify
├── tsconfig.json
├── .npmrc         官方 npm 源
├── README.md
├── ARCHITECTURE.md
├── MATH.md
└── LICENSE
```
