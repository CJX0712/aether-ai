/**
 * AETHER · ml/gmm.ts
 * ------------------------------------------------------------------
 * 高斯混合模型（GMM）—— EM 算法。
 *   E-step：计算责任度 γ(zₙₖ)（含 log-sum-exp 稳定实现）
 *   M-step：以责任度为权重更新权重、均值、协方差
 * 不变性（EM 定理）：完整数据对数似然单调不增→实为单调不减；
 *           责任度每行和为 1；权重和为 1；协方差对称半正定。
 * 参考：Bishop, PRML §9.2；Murphy, "Machine Learning: A Probabilistic Perspective" §11.4
 * author: 晨星
 */
import { Mat, Vec, zeros } from "../core/matrix";
import { cholesky, solveCholesky } from "../core/linalg";
import { logsumexp } from "../core/numerics";
import { RNG } from "../core/random";
import { kmeans } from "./kmeans";

export interface GMMModel {
  K: number;
  d: number;
  weights: Vec;
  means: Mat;
  /** 每个分量的下三角 Cholesky 因子 L (Σ = L·Lᵀ) */
  covsL: Mat[];
  logLikTrace: number[];
  responsibilities: Mat;
  /** 软指派（每点概率最大的分量） */
  predict(X: Mat): number[];
  /** 对数似然 */
  score(X: Mat): number;
}

export interface GMMOpts {
  seed?: number;
  maxIter?: number;
  tol?: number;
  /** 协方差 Ridge 正则（保证正定、数值稳定） */
  reg?: number;
}

/** 多元高斯对数密度，基于 Cholesky（数值稳定、避免矩阵求逆）。 */
function logGaussChol(x: Vec, mu: Vec, L: Mat): number {
  const d = x.length;
  const y = new Array(d);
  // 解 L·y = (x-mu)
  for (let i = 0; i < d; i++) {
    let s = x[i] - mu[i];
    for (let k = 0; k < i; k++) s -= L[i][k] * y[k];
    y[i] = s / L[i][i];
  }
  let quad = 0;
  for (let i = 0; i < d; i++) quad += y[i] * y[i];
  let logdet = 0;
  for (let i = 0; i < d; i++) logdet += Math.log(L[i][i]);
  return -0.5 * (d * Math.log(2 * Math.PI) + 2 * logdet + quad);
}

function covFromPoints(X: Mat, assign: number[], k: number, reg: number): Mat[] {
  const d = X[0].length;
  const Ls: Mat[] = [];
  const sum = Array.from({ length: k }, () => new Array(d).fill(0));
  const cnt = new Array(k).fill(0);
  for (let i = 0; i < X.length; i++) { const a = assign[i]; cnt[a]++; for (let j = 0; j < d; j++) sum[a][j] += X[i][j]; }
  for (let c = 0; c < k; c++) {
    const mu = cnt[c] > 0 ? sum[c].map(s => s / cnt[c]) : new Array(d).fill(0);
    const S = zeros(d, d);
    if (cnt[c] > 0) {
      for (let i = 0; i < X.length; i++) {
        if (assign[i] !== c) continue;
        for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) S[a][b] += (X[i][a] - mu[a]) * (X[i][b] - mu[b]);
      }
      for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) S[a][b] /= cnt[c];
    }
    for (let a = 0; a < d; a++) S[a][a] += reg;
    Ls.push(cholesky(S));
  }
  return Ls;
}

export function gmmEM(X: Mat, K: number, opts: GMMOpts = {}): GMMModel {
  const n = X.length, d = X[0].length;
  const rng = new RNG(opts.seed ?? 7);
  const maxIter = opts.maxIter ?? 200;
  const tol = opts.tol ?? 1e-6;
  const reg = opts.reg ?? 1e-6;

  // 用 k-means 初始化均值/协方差
  const km = kmeans(X, K, { seed: opts.seed ?? 7 });
  let means = km.centroids.map(c => c.slice());
  let covsL = covFromPoints(X, km.assignments, K, reg);
  let weights = new Array(K).fill(1 / K);

  const responsibilities: Mat = zeros(n, K);
  const logLikTrace: number[] = [];

  for (let iter = 0; iter < maxIter; iter++) {
    // ---- E-step ----
    let logLik = 0;
    for (let i = 0; i < n; i++) {
      const logComp = new Array(K);
      for (let k = 0; k < K; k++) {
        const lp = Math.log(weights[k] + 1e-300);
        logComp[k] = lp + logGaussChol(X[i], means[k], covsL[k]);
      }
      const lse = logsumexp(logComp);
      logLik += lse;
      const maxc = Math.max(...logComp);
      for (let k = 0; k < K; k++) responsibilities[i][k] = Math.exp(logComp[k] - lse);
    }
    logLikTrace.push(logLik);

    // ---- M-step ----
    const Nk = new Array(K).fill(0);
    for (let i = 0; i < n; i++) for (let k = 0; k < K; k++) Nk[k] += responsibilities[i][k];
    for (let k = 0; k < K; k++) weights[k] = Nk[k] / n;

    const newMeans = zeros(K, d);
    for (let i = 0; i < n; i++)
      for (let k = 0; k < K; k++) {
        const r = responsibilities[i][k];
        for (let j = 0; j < d; j++) newMeans[k][j] += r * X[i][j];
      }
    for (let k = 0; k < K; k++) if (Nk[k] > 0) for (let j = 0; j < d; j++) newMeans[k][j] /= Nk[k];
    means = newMeans;

    covsL = [];
    for (let k = 0; k < K; k++) {
      const S = zeros(d, d);
      if (Nk[k] > 0) {
        for (let i = 0; i < n; i++) {
          const r = responsibilities[i][k];
          for (let a = 0; a < d; a++) for (let b = 0; b < d; b++)
            S[a][b] += r * (X[i][a] - means[k][a]) * (X[i][b] - means[k][b]);
        }
        for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) S[a][b] /= Nk[k];
      }
      for (let a = 0; a < d; a++) S[a][a] += reg;
      covsL.push(cholesky(S));
    }

    // 收敛：对数似然增量小于 tol
    if (iter > 0 && Math.abs(logLikTrace[iter] - logLikTrace[iter - 1]) < tol * Math.abs(logLikTrace[iter - 1] + 1)) break;
  }

  const predict = (Xin: Mat): number[] =>
    Xin.map(row => {
      const logComp = new Array(K);
      for (let k = 0; k < K; k++) logComp[k] = Math.log(weights[k] + 1e-300) + logGaussChol(row, means[k], covsL[k]);
      const lse = logsumexp(logComp);
      let best = 0, bestP = -Infinity;
      for (let k = 0; k < K; k++) { const p = Math.exp(logComp[k] - lse); if (p > bestP) { bestP = p; best = k; } }
      return best;
    });

  const score = (Xin: Mat): number => {
    let ll = 0;
    for (const row of Xin) {
      const logComp = new Array(K);
      for (let k = 0; k < K; k++) logComp[k] = Math.log(weights[k] + 1e-300) + logGaussChol(row, means[k], covsL[k]);
      ll += logsumexp(logComp);
    }
    return ll;
  };

  return { K, d, weights, means, covsL, logLikTrace, responsibilities, predict, score };
}
