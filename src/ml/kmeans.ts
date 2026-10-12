/**
 * AETHER · ml/kmeans.ts
 * ------------------------------------------------------------------
 * k-means 聚类（Lloyd 算法 + k-means++ 初始化）。
 * 不变性：每次迭代簇内平方和（inertia）单调不增；最终质心为簇均值。
 * 参考：Arthur & Vassilvitskii (2007), "k-means++: The Advantages of Careful Seeding"
 *       Bishop, PRML §9.1
 * author: 晨星
 */
import { Mat, Vec } from "../core/matrix";
import { RNG } from "../core/random";

export interface KMeansModel {
  centroids: Mat;
  assignments: number[];
  inertia: number;
  predict(x: Vec): number;
  transform(X: Mat): number[];
}

function sqDist(a: Vec, b: Vec): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; }
  return s;
}

export interface KMeansOpts {
  seed?: number;
  maxIter?: number;
  tol?: number;
}

export function kmeans(X: Mat, K: number, opts: KMeansOpts = {}): KMeansModel {
  const n = X.length, d = X[0].length;
  const rng = new RNG(opts.seed ?? 0x1234);
  const maxIter = opts.maxIter ?? 300;
  const tol = opts.tol ?? 1e-6;

  // ---- k-means++ 初始化 ----
  const centroids: Mat = [X[rng.randInt(n)].slice()];
  const dist2 = new Array(n).fill(Infinity);
  for (let c = 1; c < K; c++) {
    const last = centroids[c - 1];
    for (let i = 0; i < n; i++) {
      const dd = sqDist(X[i], last);
      if (dd < dist2[i]) dist2[i] = dd;
    }
    const sum = dist2.reduce((a, b) => a + b, 0) || 1;
    let r = rng.next() * sum, idx = n - 1;
    for (let i = 0; i < n; i++) { r -= dist2[i]; if (r <= 0) { idx = i; break; } }
    centroids.push(X[idx].slice());
  }

  const assignments = new Array(n).fill(0);
  let inertia = Infinity;

  for (let iter = 0; iter < maxIter; iter++) {
    // 指派
    let newInertia = 0;
    for (let i = 0; i < n; i++) {
      let best = 0, bestD = Infinity;
      for (let c = 0; c < K; c++) {
        const dd = sqDist(X[i], centroids[c]);
        if (dd < bestD) { bestD = dd; best = c; }
      }
      assignments[i] = best;
      newInertia += bestD;
    }
    // 更新质心
    const sum = Array.from({ length: K }, () => new Array(d).fill(0));
    const cnt = new Array(K).fill(0);
    for (let i = 0; i < n; i++) {
      const a = assignments[i];
      cnt[a]++;
      const si = sum[a];
      const xi = X[i];
      for (let j = 0; j < d; j++) si[j] += xi[j];
    }
    for (let c = 0; c < K; c++) {
      if (cnt[c] > 0) for (let j = 0; j < d; j++) centroids[c][j] = sum[c][j] / cnt[c];
    }
    if (Math.abs(newInertia - inertia) < tol) { inertia = newInertia; break; }
    inertia = newInertia;
  }

  const predict = (x: Vec): number => {
    let best = 0, bestD = Infinity;
    for (let c = 0; c < K; c++) {
      const dd = sqDist(x, centroids[c]);
      if (dd < bestD) { bestD = dd; best = c; }
    }
    return best;
  };

  return { centroids, assignments, inertia, predict, transform: (Xin: Mat) => Xin.map(predict) };
}
