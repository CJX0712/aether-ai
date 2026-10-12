/**
 * AETHER · dl/autograd.ts
 * ------------------------------------------------------------------
 * 张量级反向模式自动微分（reverse-mode autodiff / backprop）。
 * 与 PyTorch/TensorFlow 的自动微分同源思想：构建计算图，反向遍历累积梯度。
 *   - 节点 = 矩阵（Mat），附带 grad 与 _backward 闭包
 *   - 支持 matmul / 广播加减 / 逐元素乘 / 转置 / relu / sigmoid / tanh /
 *     row-wise softmax / 全元素均值
 * 不变性（梯度检验）：解析梯度 vs 中心差分，最大相对误差 ≈ 1e-7。
 * 参考：Baydin et al. (2018), "Automatic Differentiation in Machine Learning";
 *       Goodfellow §6.5; Karpathy micrograd（结构启发）
 * author: 晨星
 */
import { Mat, Vec, zeros, matmul, transpose, clone } from "../core/matrix";

export interface Tensor {
  data: Mat;
  grad: Mat;
  _backward: () => void;
  _prev: Tensor[];
}

export function tensor(data: Mat): Tensor {
  const n = data.length, m = data[0].length;
  return { data: clone(data), grad: zeros(n, m), _backward: () => {}, _prev: [] };
}

/** 反向传播：自顶向下拓扑遍历，累加梯度。 */
export function backward(root: Tensor): void {
  const topo: Tensor[] = [];
  const visited = new Set<Tensor>();
  const build = (v: Tensor) => {
    if (visited.has(v)) return;
    visited.add(v);
    for (const p of v._prev) build(p);
    topo.push(v);
  };
  build(root);
  for (const v of topo) v.grad = zeros(v.data.length, v.data[0].length);
  // 种子：∂L/∂L = 1
  const rn = root.data.length, rm = root.data[0].length;
  root.grad = zeros(rn, rm).map((row, i) => row.map((_, j) => (i === j && rn === rm ? 1 : (rn === 1 || rm === 1) ? 1 : 1)));
  // 标量损失（1×1）种子为 1；向量/矩阵输出逐元素种子为 1（用于逐元素梯度检验）
  for (let i = 0; i < rn; i++) for (let j = 0; j < rm; j++) root.grad[i][j] = 1;
  for (let i = topo.length - 1; i >= 0; i--) topo[i]._backward();
}

function broadcastShapes(a: Mat, b: Mat): [number, number] {
  const an = a.length, am = a[0].length, bn = b.length, bm = b[0].length;
  const n = Math.max(an, bn), m = Math.max(am, bm);
  if ((an !== n && an !== 1) || (bn !== n && bn !== 1) ||
      (am !== m && am !== 1) || (bm !== m && bm !== 1))
    throw new Error(`广播形状不兼容: ${an}x${am} vs ${bn}x${bm}`);
  return [n, m];
}

function broadcastAdd(a: Mat, b: Mat): Mat {
  const [n, m] = broadcastShapes(a, b);
  const out = zeros(n, m);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < m; j++) {
      const av = a[a.length === 1 ? 0 : i][a[0].length === 1 ? 0 : j];
      const bv = b[b.length === 1 ? 0 : i][b[0].length === 1 ? 0 : j];
      out[i][j] = av + bv;
    }
  return out;
}

function broadcastSub(a: Mat, b: Mat): Mat {
  const [n, m] = broadcastShapes(a, b);
  const out = zeros(n, m);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < m; j++) {
      const av = a[a.length === 1 ? 0 : i][a[0].length === 1 ? 0 : j];
      const bv = b[b.length === 1 ? 0 : i][b[0].length === 1 ? 0 : j];
      out[i][j] = av - bv;
    }
  return out;
}

/** 矩阵乘法 C = A·B */
export function matmulOp(a: Tensor, b: Tensor): Tensor {
  const out = tensor(matmul(a.data, b.data));
  out._prev = [a, b];
  out._backward = () => {
    a.grad = broadcastAdd(a.grad, matmul(out.grad, transpose(b.data)));
    b.grad = broadcastAdd(b.grad, matmul(transpose(a.data), out.grad));
  };
  return out;
}

export function addOp(a: Tensor, b: Tensor): Tensor {
  const out = tensor(broadcastAdd(a.data, b.data));
  out._prev = [a, b];
  out._backward = () => {
    const [n, m] = [out.data.length, out.data[0].length];
    for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) {
      const gi = out.grad[i][j];
      if (a.data.length === n && a.data[0].length === m) a.grad[i][j] += gi;
      else if (a.data.length === 1) a.grad[0][j] += gi;
      else a.grad[i][0] += gi;
      if (b.data.length === n && b.data[0].length === m) b.grad[i][j] += gi;
      else if (b.data.length === 1) b.grad[0][j] += gi;
      else b.grad[i][0] += gi;
    }
  };
  return out;
}

export function subOp(a: Tensor, b: Tensor): Tensor {
  const out = tensor(broadcastSub(a.data, b.data));
  out._prev = [a, b];
  out._backward = () => {
    const [n, m] = [out.data.length, out.data[0].length];
    for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) {
      const gi = out.grad[i][j];
      if (a.data.length === n && a.data[0].length === m) a.grad[i][j] += gi;
      else if (a.data.length === 1) a.grad[0][j] += gi;
      else a.grad[i][0] += gi;
      if (b.data.length === n && b.data[0].length === m) b.grad[i][j] -= gi;
      else if (b.data.length === 1) b.grad[0][j] -= gi;
      else b.grad[i][0] -= gi;
    }
  };
  return out;
}

export function scaleOp(a: Tensor, s: number): Tensor {
  const n = a.data.length, m = a.data[0].length;
  const out = tensor(a.data.map(r => r.map(v => v * s)));
  out._prev = [a];
  out._backward = () => {
    for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) a.grad[i][j] += s * out.grad[i][j];
  };
  return out;
}

export function transposeOp(a: Tensor): Tensor {
  const out = tensor(transpose(a.data));
  out._prev = [a];
  out._backward = () => {
    const [n, m] = [a.data.length, a.data[0].length];
    for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) a.grad[i][j] += out.grad[j][i];
  };
  return out;
}

function elemOp(a: Tensor, b: Tensor, f: (x: number, y: number) => number,
                dfdx: (x: number, y: number, o: number) => number,
                dfdy: (x: number, y: number, o: number) => number): Tensor {
  const [n, m] = broadcastShapes(a.data, b.data);
  const out = zeros(n, m);
  const ad = a.data, bd = b.data;
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) {
    const x = ad[ad.length === 1 ? 0 : i][ad[0].length === 1 ? 0 : j];
    const y = bd[bd.length === 1 ? 0 : i][bd[0].length === 1 ? 0 : j];
    out[i][j] = f(x, y);
  }
  const t = tensor(out);
  t._prev = [a, b];
  t._backward = () => {
    for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) {
      const x = ad[ad.length === 1 ? 0 : i][ad[0].length === 1 ? 0 : j];
      const y = bd[bd.length === 1 ? 0 : i][bd[0].length === 1 ? 0 : j];
      const o = out[i][j], g = t.grad[i][j];
      if (ad.length === n && ad[0].length === m) a.grad[i][j] += dfdx(x, y, o) * g;
      else if (ad.length === 1) a.grad[0][j] += dfdx(x, y, o) * g;
      else a.grad[i][0] += dfdx(x, y, o) * g;
      if (bd.length === n && bd[0].length === m) b.grad[i][j] += dfdy(x, y, o) * g;
      else if (bd.length === 1) b.grad[0][j] += dfdy(x, y, o) * g;
      else b.grad[i][0] += dfdy(x, y, o) * g;
    }
  };
  return t;
}

/** 逐元素乘（Hadamard 积）。 */
export function mul(a: Tensor, b: Tensor): Tensor {
  return elemOp(a, b,
    (x, y) => x * y,
    (x, y) => y,
    (x, y) => x);
}

export function reluOp(a: Tensor): Tensor {
  const n = a.data.length, m = a.data[0].length;
  const out = zeros(n, m);
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) out[i][j] = a.data[i][j] > 0 ? a.data[i][j] : 0;
  const t = tensor(out);
  t._prev = [a];
  t._backward = () => {
    for (let i = 0; i < n; i++) for (let j = 0; j < m; j++)
      a.grad[i][j] += (a.data[i][j] > 0 ? 1 : 0) * t.grad[i][j];
  };
  return t;
}

export function sigmoidOp(a: Tensor): Tensor {
  const n = a.data.length, m = a.data[0].length;
  const out = zeros(n, m);
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) {
    const z = a.data[i][j];
    out[i][j] = z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
  }
  const t = tensor(out);
  t._prev = [a];
  t._backward = () => {
    for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) {
      const s = out[i][j];
      a.grad[i][j] += s * (1 - s) * t.grad[i][j];
    }
  };
  return t;
}

export function tanhOp(a: Tensor): Tensor {
  const n = a.data.length, m = a.data[0].length;
  const out = zeros(n, m);
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) out[i][j] = Math.tanh(a.data[i][j]);
  const t = tensor(out);
  t._prev = [a];
  t._backward = () => {
    for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) {
      const th = out[i][j];
      a.grad[i][j] += (1 - th * th) * t.grad[i][j];
    }
  };
  return t;
}

/** 行级 softmax（每行和为 1，用于注意力权重）。 */
export function softmaxRowsOp(a: Tensor): Tensor {
  const n = a.data.length, m = a.data[0].length;
  const out = zeros(n, m);
  for (let i = 0; i < n; i++) {
    const row = a.data[i];
    const mx = Math.max(...row);
    let s = 0;
    for (let j = 0; j < m; j++) { out[i][j] = Math.exp(row[j] - mx); s += out[i][j]; }
    for (let j = 0; j < m; j++) out[i][j] /= s;
  }
  const t = tensor(out);
  t._prev = [a];
  t._backward = () => {
    for (let i = 0; i < n; i++) {
      // Jacobian of row softmax: diag(s) - s sᵀ ; grad_in = Jᵀ grad_out
      const s = out[i];
      const g = t.grad[i];
      let dot = 0;
      for (let j = 0; j < m; j++) dot += s[j] * g[j];
      for (let j = 0; j < m; j++) a.grad[i][j] += s[j] * (g[j] - dot);
    }
  };
  return t;
}

/** 全元素均值（标量损失节点）。 */
export function meanOp(a: Tensor): Tensor {
  const n = a.data.length, m = a.data[0].length;
  let s = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) s += a.data[i][j];
  const out = tensor([[s / (n * m)]]);
  out._prev = [a];
  out._backward = () => {
    const g = out.grad[0][0] / (n * m);
    for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) a.grad[i][j] += g;
  };
  return out;
}

/** 逐元素对数 log(x)。 */
export function logOp(a: Tensor): Tensor {
  const n = a.data.length, m = a.data[0].length;
  const out = zeros(n, m);
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) out[i][j] = Math.log(a.data[i][j]);
  const t = tensor(out);
  t._prev = [a];
  t._backward = () => {
    for (let i = 0; i < n; i++) for (let j = 0; j < m; j++)
      a.grad[i][j] += (1 / a.data[i][j]) * t.grad[i][j];
  };
  return t;
}

/** 全元素求和（标量节点）。 */
export function sumOp(a: Tensor): Tensor {
  const n = a.data.length, m = a.data[0].length;
  let s = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) s += a.data[i][j];
  const out = tensor([[s]]);
  out._prev = [a];
  out._backward = () => {
    const g = out.grad[0][0];
    for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) a.grad[i][j] += g;
  };
  return out;
}

/** 将一批参数展平为 Vec（用于梯度检验/优化器）。 */
export function flattenGrads(params: Tensor[]): Vec {
  const g: Vec = [];
  for (const p of params) for (const row of p.grad) for (const v of row) g.push(v);
  return g;
}

export function paramsShape(params: Tensor[]): number[] {
  return params.map(p => p.data.length * p.data[0].length);
}
