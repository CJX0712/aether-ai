/**
 * AETHER · ml/logistic.ts
 * ------------------------------------------------------------------
 * 逻辑回归（二分类），最大似然 = 最小化负对数似然（NLL）。
 *   梯度：∂L/∂w = (1/n) Σ (σ(wᵀx+b) − y)·x ；∂L/∂b = (1/n) Σ (σ − y)
 * 用 Adam 优化；提供梯度检验不变量。
 * 参考：Bishop, PRML §4.3；Goodfellow §6.2
 * author: 晨星
 */
import { Mat, Vec } from "../core/matrix";
import { sigmoid } from "../core/numerics";
import { adam, AdamOpts } from "../core/optim";

export interface LogisticModel {
  weights: Vec;
  bias: number;
  lossTrace: number[];
  predictProba(X: Mat): Vec;
  predict(X: Mat): number[];
}

export interface LogisticOpts extends AdamOpts {
  seed?: number;
}

export function logisticRegression(X: Mat, y: Vec, opts: LogisticOpts = {}): LogisticModel {
  const n = X.length, d = X[0].length;

  // 目标函数：参数 = [w(0..d-1), b]
  const objective = (p: Vec): { value: number; grad: Vec } => {
    const w = p.slice(0, d);
    const b = p[d];
    let loss = 0;
    const gw = new Array(d).fill(0);
    let gb = 0;
    for (let i = 0; i < n; i++) {
      let z = b;
      const xi = X[i];
      for (let j = 0; j < d; j++) z += w[j] * xi[j];
      const phat = sigmoid(z);
      // NLL 分量
      const yi = y[i];
      const term = -(yi * Math.log(phat + 1e-300) + (1 - yi) * Math.log(1 - phat + 1e-300));
      loss += term;
      const err = phat - yi;
      for (let j = 0; j < d; j++) gw[j] += err * xi[j];
      gb += err;
    }
    loss /= n;
    for (let j = 0; j < d; j++) gw[j] /= n;
    gb /= n;
    const grad = gw.concat([gb]);
    return { value: loss, grad };
  };

  const w0 = new Array(d + 1).fill(0);
  const res = adam(objective, w0, opts);
  const weights = res.w.slice(0, d);
  const bias = res.w[d];

  const predictProba = (Xin: Mat): Vec =>
    Xin.map(row => {
      let z = bias;
      for (let j = 0; j < d; j++) z += weights[j] * row[j];
      return sigmoid(z);
    });

  const predict = (Xin: Mat): number[] => predictProba(Xin).map(p => (p >= 0.5 ? 1 : 0));

  return { weights, bias, lossTrace: res.trace, predictProba, predict };
}
