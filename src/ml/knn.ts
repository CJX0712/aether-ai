/**
 * AETHER · ml/knn.ts
 * ------------------------------------------------------------------
 * k-近邻（惰性学习）：分类取众数，回归取均值。
 * 不变性：k=1 时训练集误差为 0（记忆）；距离度量可切换欧氏/余弦。
 * 参考：Cover & Hart (1967)；Bishop, PRML §2.5
 * author: 晨星
 */
import { Mat, Vec } from "../core/matrix";

export type Metric = "euclidean" | "cosine";

export interface KNNModel {
  k: number;
  predict(X: Mat): Vec;
}

export interface KNNOpts {
  metric?: Metric;
  /** true=回归（均值），false=分类（众数） */
  regression?: boolean;
}

function dist(a: Vec, b: Vec, metric: Metric): number {
  if (metric === "cosine") {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    const denom = Math.sqrt(na) * Math.sqrt(nb);
    if (denom === 0) return 0;
    return 1 - dot / denom;
  }
  let s = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; }
  return Math.sqrt(s);
}

export function knn(X: Mat, y: Vec, k: number, opts: KNNOpts = {}): KNNModel {
  const metric = opts.metric ?? "euclidean";
  const regression = opts.regression ?? false;

  const predictOne = (x: Vec): number => {
    const idx = X.map((xi, i) => ({ i, d: dist(x, xi, metric) }))
      .sort((a, b) => a.d - b.d)
      .slice(0, k);
    if (regression) {
      return idx.reduce((s, o) => s + y[o.i], 0) / idx.length;
    }
    const counts = new Map<number, number>();
    for (const o of idx) counts.set(y[o.i], (counts.get(y[o.i]) ?? 0) + 1);
    let best = 0, bestC = -1;
    for (const [label, c] of counts) if (c > bestC) { bestC = c; best = label; }
    return best;
  };

  return {
    k,
    predict: (Xin: Mat): Vec => Xin.map(predictOne)
  };
}
