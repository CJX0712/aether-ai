/**
 * AETHER · core/optim.ts
 * ------------------------------------------------------------------
 * 一阶/自适应优化器（站在世界级优化理论之上）：
 *   - 梯度下降（含动量）        → 经典梯度法
 *   - Adam                     → Kingma & Ba (2015), ICLR，业界标准自适应优化器
 * 接口：objective(w) → { value, grad }，返回最优参数与损失轨迹（便于不变量检验）。
 * author: 晨星
 */
import { Vec } from "./matrix";

export interface ObjFun {
  (w: Vec): { value: number; grad: Vec };
}

export interface OptResult {
  w: Vec;
  trace: number[];
  iters: number;
}

export interface GDOpts {
  lr?: number;
  momentum?: number;
  maxIter?: number;
  tol?: number;
}

/** 带动量的梯度下降。 */
export function gradientDescent(f: ObjFun, w0: Vec, opts: GDOpts = {}): OptResult {
  const lr = opts.lr ?? 0.1;
  const momentum = opts.momentum ?? 0.0;
  const maxIter = opts.maxIter ?? 1000;
  const tol = opts.tol ?? 1e-8;

  let w = w0.slice();
  let v = new Array(w.length).fill(0);
  const trace: number[] = [];
  let iters = 0;

  for (let t = 0; t < maxIter; t++) {
    const { value, grad } = f(w);
    trace.push(value);
    iters = t + 1;
    if (Math.abs(value) < tol && t > 0) break;
    for (let i = 0; i < w.length; i++) {
      v[i] = momentum * v[i] - lr * grad[i];
      w[i] += v[i];
    }
    // 收敛判据：梯度范数
    let gn = 0;
    for (let i = 0; i < grad.length; i++) gn += grad[i] * grad[i];
    if (Math.sqrt(gn) < tol) break;
  }
  return { w, trace, iters };
}

export interface AdamOpts {
  lr?: number;
  beta1?: number;
  beta2?: number;
  eps?: number;
  maxIter?: number;
  tol?: number;
}

/** Adam 优化器（Kingma & Ba, 2015）。 */
export function adam(f: ObjFun, w0: Vec, opts: AdamOpts = {}): OptResult {
  const lr = opts.lr ?? 0.01;
  const b1 = opts.beta1 ?? 0.9;
  const b2 = opts.beta2 ?? 0.999;
  const eps = opts.eps ?? 1e-8;
  const maxIter = opts.maxIter ?? 1000;
  const tol = opts.tol ?? 1e-8;

  let w = w0.slice();
  const m = new Array(w.length).fill(0);
  const v = new Array(w.length).fill(0);
  const trace: number[] = [];
  let iters = 0;

  for (let t = 0; t < maxIter; t++) {
    const { value, grad } = f(w);
    trace.push(value);
    iters = t + 1;
    const tc = t + 1;
    let gn = 0;
    for (let i = 0; i < w.length; i++) {
      m[i] = b1 * m[i] + (1 - b1) * grad[i];
      v[i] = b2 * v[i] + (1 - b2) * grad[i] * grad[i];
      const mhat = m[i] / (1 - Math.pow(b1, tc));
      const vhat = v[i] / (1 - Math.pow(b2, tc));
      w[i] -= lr * mhat / (Math.sqrt(vhat) + eps);
      gn += grad[i] * grad[i];
    }
    if (Math.sqrt(gn) < tol) break;
  }
  return { w, trace, iters };
}
