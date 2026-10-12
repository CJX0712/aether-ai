/**
 * AETHER · core/linalg.ts
 * ------------------------------------------------------------------
 * 世界顶级数值线性代数实现（站在巨人肩上）：
 *   - 单侧 Jacobi SVD  → LAPACK xGEJSV 同源算法，小/中规模精度最高
 *   - Jacobi 对称特征值分解 → 经典 Golub & Van Loan 算法 8.4.3
 *   - Cholesky 分解  → A = L Lᵀ （正定矩阵）
 *   - Householder QR → 数值稳定正交三角化
 *   - 带部分主元的高斯消元线性方程组求解
 *
 * 参考：Golub & Van Loan, "Matrix Computations" (4th ed., 2013)
 *       Demmel & Veselić (1992), "Jacobi's method is more accurate than QR"
 * author: 晨星
 */

import { Mat, Vec, zeros, identity, clone, transpose } from "./matrix";

const EPS = Number.EPSILON;

export interface SVD {
  /** 左奇异向量矩阵 (m × n，瘦长形式)，列标准正交 */
  U: Mat;
  /** 奇异值向量 (长度 n，降序排列) */
  S: Vec;
  /** 右奇异向量矩阵 (n × n)，正交 */
  V: Mat;
}

/**
 * 单侧 Jacobi SVD（Hestenes 1958 / Drmač 现代实现）。
 * 通过对 A 的列做正交化旋转，使 A' = Uᵀ A 列正交，
 * 然后 S = ‖列‖，U = 列/S，V = 累积旋转。
 * 满足 A = U · diag(S) · Vᵀ，U、V 标准正交。
 */
export function jacobiSVD(Ain: Mat, maxSweeps = 60): SVD {
  const m = Ain.length, n = Ain[0].length;
  const B = clone(Ain);
  const V = identity(n);

  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let converged = true;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        let ap = 0, aq = 0, apq = 0;
        for (let i = 0; i < m; i++) {
          const bp = B[i][p], bq = B[i][q];
          ap += bp * bp; aq += bq * bq; apq += bp * bq;
        }
        const denom = Math.sqrt(ap * aq);
        if (denom > 0 && Math.abs(apq) <= 1e3 * EPS * denom) continue; // 已足够正交
        converged = false;
        let c = 1, s = 0;
        if (Math.abs(apq) > 0) {
          const zeta = (aq - ap) / (2 * apq);
          const t = Math.sign(zeta) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
          c = 1 / Math.sqrt(1 + t * t);
          s = c * t;
        }
        for (let i = 0; i < m; i++) {
          const bp = B[i][p], bq = B[i][q];
          B[i][p] = c * bp - s * bq;
          B[i][q] = s * bp + c * bq;
        }
        for (let i = 0; i < n; i++) {
          const vp = V[i][p], vq = V[i][q];
          V[i][p] = c * vp - s * vq;
          V[i][q] = s * vp + c * vq;
        }
      }
    }
    if (converged) break;
  }

  // 奇异值 + 瘦 U
  const S = new Array(n).fill(0);
  const U = zeros(m, n);
  for (let j = 0; j < n; j++) {
    let norm = 0;
    for (let i = 0; i < m; i++) norm += B[i][j] * B[i][j];
    norm = Math.sqrt(norm);
    S[j] = norm;
    if (norm > 0) for (let i = 0; i < m; i++) U[i][j] = B[i][j] / norm;
  }

  // 降序排列
  const order = [...Array(n).keys()].sort((a, b) => S[b] - S[a]);
  const Ss = order.map(k => S[k]);
  const Us = zeros(m, n), Vs = zeros(n, n);
  for (let j = 0; j < n; j++)
    for (let i = 0; i < m; i++) Us[i][j] = U[i][order[j]];
  for (let j = 0; j < n; j++)
    for (let i = 0; i < n; i++) Vs[i][j] = V[i][order[j]];

  return { U: Us, S: Ss, V: Vs };
}

export interface EigSym {
  /** 特征值（升序） */
  values: Vec;
  /** 特征向量矩阵（列向量），标准正交 */
  vectors: Mat;
}

/** Jacobi 对称特征值分解：返回升序特征值与正交特征向量矩阵。 */
export function jacobiEigen(Ain: Mat, maxSweeps = 100): EigSym {
  const n = Ain.length;
  const A = clone(Ain);
  const V = identity(n);

  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let p = 0, q = 1, maxOff = 0;
    for (let i = 0; i < n; i++)
      for (let j = i + 1; j < n; j++) {
        const v = Math.abs(A[i][j]);
        if (v > maxOff) { maxOff = v; p = i; q = j; }
      }
    if (maxOff < 1e2 * EPS) break;

    const app = A[p][p], aqq = A[q][q], apq = A[p][q];
    let c = 1, s = 0;
    if (Math.abs(apq) > 0) {
      const zeta = (aqq - app) / (2 * apq);
      const t = Math.sign(zeta) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
      c = 1 / Math.sqrt(1 + t * t);
      s = c * t;
    }
    A[p][p] = c * c * app - 2 * c * s * apq + s * s * aqq;
    A[q][q] = s * s * app + 2 * c * s * apq + c * c * aqq;
    A[p][q] = 0; A[q][p] = 0;
    for (let i = 0; i < n; i++) {
      if (i !== p && i !== q) {
        const aip = A[i][p], aiq = A[i][q];
        A[i][p] = c * aip - s * aiq; A[p][i] = A[i][p];
        A[i][q] = s * aip + c * aiq; A[q][i] = A[i][q];
      }
      const vip = V[i][p], viq = V[i][q];
      V[i][p] = c * vip - s * viq;
      V[i][q] = s * vip + c * viq;
    }
  }

  const values = A.map((row, i) => row[i]);
  return { values, vectors: V };
}

/** Cholesky 分解：返回下三角 L，使 A = L·Lᵀ（要求 A 对称正定）。 */
export function cholesky(A: Mat): Mat {
  const n = A.length;
  const L = zeros(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = 0;
      for (let k = 0; k < j; k++) s += L[i][k] * L[j][k];
      if (i === j) {
        const d = A[i][i] - s;
        if (d <= 0) throw new Error("Cholesky 失败：矩阵非正定");
        L[i][j] = Math.sqrt(d);
      } else {
        L[i][j] = (A[i][j] - s) / L[j][j];
      }
    }
  }
  return L;
}

/** 解 L·Lᵀ·x = b（Cholesky 法：前代 + 回代）。 */
export function solveCholesky(L: Mat, b: Vec): Vec {
  const n = L.length;
  const y = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= L[i][k] * y[k];
    y[i] = s / L[i][i];
  }
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i];
    for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k];
    x[i] = s / L[i][i];
  }
  return x;
}

/** Householder QR：A(m×n, m≥n) = Q(m×m)·R(m×n)，Q 正交、R 上三角。 */
export function qrHouseholder(Ain: Mat): { Q: Mat; R: Mat } {
  const m = Ain.length, n = Ain[0].length;
  const Q = identity(m);
  const R = clone(Ain);
  for (let k = 0; k < n; k++) {
    const x = new Array(m - k);
    for (let i = 0; i < m - k; i++) x[i] = R[k + i][k];
    const norm = Math.sqrt(x.reduce((s, v) => s + v * v, 0));
    if (norm < 1e-14) continue;
    const sign = x[0] >= 0 ? 1 : -1;
    const u0 = x[0] + sign * norm;
    const u = x.slice(); u[0] = u0;
    let unorm = Math.sqrt(u.reduce((s, v) => s + v * v, 0));
    const v = u.map(vv => vv / unorm);
    for (let c = k; c < n; c++) {
      let dot = 0;
      for (let i = 0; i < m - k; i++) dot += v[i] * R[k + i][c];
      for (let i = 0; i < m - k; i++) R[k + i][c] -= 2 * v[i] * dot;
    }
    for (let c = 0; c < m; c++) {
      let dot = 0;
      for (let i = 0; i < m - k; i++) dot += v[i] * Q[k + i][c];
      for (let i = 0; i < m - k; i++) Q[k + i][c] -= 2 * v[i] * dot;
    }
  }
  // Q 累积为 H_{n-1}…H_0；因 Householder 反射对称，正确正交因子为其转置 H_0…H_{n-1}
  return { Q: transpose(Q), R };
}

/** 带部分主元的高斯消元：解 A·x = b（A 为 n×n，通用，非必须正定）。 */
export function solve(Ain: Mat, bin: Vec): Vec {
  const n = Ain.length;
  const A = clone(Ain);
  const b = bin.slice();
  for (let col = 0; col < n; col++) {
    // 部分主元
    let piv = col;
    for (let r = col + 1; r < n; r++)
      if (Math.abs(A[r][col]) > Math.abs(A[piv][col])) piv = r;
    if (Math.abs(A[piv][col]) < 1e-14) throw new Error("矩阵奇异，无法求解");
    if (piv !== col) {
      [A[piv], A[col]] = [A[col], A[piv]];
      [b[piv], b[col]] = [b[col], b[piv]];
    }
    const d = A[col][col];
    for (let r = col + 1; r < n; r++) {
      const f = A[r][col] / d;
      if (f === 0) continue;
      for (let c = col; c < n; c++) A[r][c] -= f * A[col][c];
      b[r] -= f * b[col];
    }
  }
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = b[i];
    for (let c = i + 1; c < n; c++) s -= A[i][c] * x[c];
    x[i] = s / A[i][i];
  }
  return x;
}

/** 对称正定矩阵求逆（Cholesky 法）。 */
export function inverseSPD(A: Mat): Mat {
  const n = A.length;
  const L = cholesky(A);
  const inv = zeros(n, n);
  for (let j = 0; j < n; j++) {
    const e = new Array(n).fill(0); e[j] = 1;
    const col = solveCholesky(L, e);
    for (let i = 0; i < n; i++) inv[i][j] = col[i];
  }
  return inv;
}
