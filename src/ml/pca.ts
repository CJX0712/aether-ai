/**
 * AETHER · ml/pca.ts
 * ------------------------------------------------------------------
 * 主成分分析（PCA），基于 SVD（数值最稳，优于协方差特征分解）。
 *   A_centered = U·S·Vᵀ  →  主成分 = V 的行（右奇异向量）
 * 不变性：主成分彼此正交归一（VᵀV=I）；全维重构误差 ≈ 0；解释方差递减。
 * 参考：Bishop, "Pattern Recognition and Machine Learning" (2006), §12.1
 *       Jolliffe, "Principal Component Analysis" (2002)
 * author: 晨星
 */
import { Mat, Vec, zeros } from "../core/matrix";
import { jacobiSVD } from "../core/linalg";

export interface PCAModel {
  mean: Vec;
  /** 主成分（K × d），每行一个方向 */
  components: Mat;
  /** 奇异值 */
  S: Vec;
  /** 各主成分解释的方差 */
  explainedVariance: Vec;
  /** 解释方差占比（累计和为 1） */
  explainedRatio: Vec;
  k: number;
  transform(X: Mat): Mat;
  reconstruct(Z: Mat): Mat;
}

export function pca(X: Mat, k?: number): PCAModel {
  const n = X.length, d = X[0].length;
  if (n === 0) throw new Error("PCA: 空数据集");

  const mean = new Array(d).fill(0);
  for (const row of X) for (let j = 0; j < d; j++) mean[j] += row[j];
  for (let j = 0; j < d; j++) mean[j] /= n;

  const Xc = X.map(row => row.map((v, j) => v - mean[j]));

  const { S, V } = jacobiSVD(Xc);
  const K = k ?? d;
  if (K > d) throw new Error(`PCA: k(${K}) 不能大于维度 d(${d})`);

  const components = V.slice(0, K); // K × d
  const totalVar = S.reduce((s, v) => s + v * v, 0) / n || 1;
  const explainedVariance = components.map((_, i) => (S[i] * S[i]) / n);
  const explainedRatio = explainedVariance.map(v => v / totalVar);

  const transform = (Xin: Mat): Mat =>
    Xin.map(row => {
      const c = row.map((v, j) => v - mean[j]);
      return components.map(comp => {
        let s = 0;
        for (let j = 0; j < d; j++) s += c[j] * comp[j];
        return s;
      });
    });

  const reconstruct = (Z: Mat): Mat =>
    Z.map(z => {
      const c = new Array(d).fill(0);
      for (let i = 0; i < K; i++)
        for (let j = 0; j < d; j++) c[j] += z[i] * components[i][j];
      return c.map((v, j) => v + mean[j]);
    });

  return { mean, components, S, explainedVariance, explainedRatio, k: K, transform, reconstruct };
}
