/**
 * AETHER · core/numerics.ts
 * ------------------------------------------------------------------
 * 数值稳定通用工具。
 *   - logsumexp：防止 exp 溢出的对数求和技巧（机器学习标配）
 *   - softmax / sigmoid / relu：激活与归一化
 * 参考：Goodfellow, Bengio & Courville, "Deep Learning" (2016), §3.2, §6.2
 * author: 晨星
 */
import { Vec } from "./matrix";

export const EPS = Number.EPSILON;
export const PI = Math.PI;

/** log(Σ exp(xᵢ))，数值稳定（减去最大值） */
export function logsumexp(x: Vec): number {
  let m = -Infinity;
  for (const v of x) if (v > m) m = v;
  if (!isFinite(m)) return m;
  let s = 0;
  for (const v of x) s += Math.exp(v - m);
  return m + Math.log(s);
}

/** softmax：输出为概率分布（分量和为 1，非负） */
export function softmax(x: Vec): Vec {
  const m = Math.max(...x);
  const ex = x.map(v => Math.exp(v - m));
  const s = ex.reduce((a, b) => a + b, 0);
  return ex.map(v => v / s);
}

export function sigmoid(x: number): number {
  // 数值稳定写法
  if (x >= 0) return 1 / (1 + Math.exp(-x));
  const e = Math.exp(x);
  return e / (1 + e);
}

export function relu(x: number): number {
  return x > 0 ? x : 0;
}

export function reluGrad(x: number): number {
  return x > 0 ? 1 : 0;
}

export function tanh(x: number): number {
  return Math.tanh(x);
}

/** 数值梯度（中心差分），用于梯度检验 */
export function numericalGrad(
  f: (x: Vec) => number,
  x: Vec,
  h = 1e-5
): Vec {
  const g = new Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const xp = x.slice(); xp[i] += h;
    const xm = x.slice(); xm[i] -= h;
    g[i] = (f(xp) - f(xm)) / (2 * h);
  }
  return g;
}

/** 两向量最大相对误差 */
export function maxRelErr(a: Vec, b: Vec): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const denom = Math.max(Math.abs(a[i]), Math.abs(b[i]), 1e-12);
    m = Math.max(m, Math.abs(a[i] - b[i]) / denom);
  }
  return m;
}
