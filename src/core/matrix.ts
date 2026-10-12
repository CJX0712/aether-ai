/**
 * AETHER · core/matrix.ts
 * ------------------------------------------------------------------
 * 矩阵基础类型与运算。
 * 设计：矩阵以「行优先的 number[][]」表示（Mat），API 清晰、可读、
 * 数值正确优先；小规模（≤数千维）性能足够，符合 forge 系列哲学。
 *
 * 数学基础参考：
 *   - Golub & Van Loan, "Matrix Computations" (4th ed., 2013)
 *   - Trefethen & Bau, "Numerical Linear Algebra" (1997)
 *
 * author: 晨星
 */

export type Mat = number[][];
export type Vec = number[];

/** 构造 n×m 零矩阵 */
export function zeros(n: number, m: number): Mat {
  const A: Mat = new Array(n);
  for (let i = 0; i < n; i++) A[i] = new Array(m).fill(0);
  return A;
}

/** 构造 n×n 单位阵 */
export function identity(n: number): Mat {
  const I = zeros(n, n);
  for (let i = 0; i < n; i++) I[i][i] = 1;
  return I;
}

/** 矩阵转置 A^T */
export function transpose(A: Mat): Mat {
  const n = A.length, m = A[0].length;
  const T = zeros(m, n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < m; j++) T[j][i] = A[i][j];
  return T;
}

/** 矩阵乘法 C = A·B （维度 n×k 与 k×m） */
export function matmul(A: Mat, B: Mat): Mat {
  const n = A.length, k = A[0].length, m = B[0].length;
  if (B.length !== k) throw new Error(`matmul 维度不匹配: A 列=${k}, B 行=${B.length}`);
  const C = zeros(n, m);
  for (let i = 0; i < n; i++) {
    const Ai = A[i];
    const Ci = C[i];
    for (let p = 0; p < k; p++) {
      const a = Ai[p];
      if (a === 0) continue;
      const Bp = B[p];
      for (let j = 0; j < m; j++) Ci[j] += a * Bp[j];
    }
  }
  return C;
}

/** 标量乘矩阵 */
export function scale(A: Mat, s: number): Mat {
  return A.map(row => row.map(v => v * s));
}

/** 矩阵逐元素相加 */
export function add(A: Mat, B: Mat): Mat {
  const n = A.length, m = A[0].length;
  const C = zeros(n, m);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < m; j++) C[i][j] = A[i][j] + B[i][j];
  return C;
}

/** 矩阵逐元素相减 */
export function sub(A: Mat, B: Mat): Mat {
  const n = A.length, m = A[0].length;
  const C = zeros(n, m);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < m; j++) C[i][j] = A[i][j] - B[i][j];
  return C;
}

/** 向量（视为列）乘矩阵：返回行向量 a^T·B */
export function vecMat(a: Vec, B: Mat): Vec {
  const k = a.length, m = B[0].length;
  const out = new Array(m).fill(0);
  for (let p = 0; p < k; p++) {
    const ap = a[p];
    if (ap === 0) continue;
    const Bp = B[p];
    for (let j = 0; j < m; j++) out[j] += ap * Bp[j];
  }
  return out;
}

/** 矩阵乘向量：A·x （x 为列向量） */
export function matVec(A: Mat, x: Vec): Vec {
  const n = A.length, k = A[0].length;
  const out = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let s = 0;
    const Ai = A[i];
    for (let p = 0; p < k; p++) s += Ai[p] * x[p];
    out[i] = s;
  }
  return out;
}

/** Frobenius 范数 */
export function frobenius(A: Mat): number {
  let s = 0;
  for (const row of A) for (const v of row) s += v * v;
  return Math.sqrt(s);
}

/** 两矩阵最大绝对差（用于不变量检验） */
export function maxAbsDiff(A: Mat, B: Mat): number {
  let m = 0;
  for (let i = 0; i < A.length; i++)
    for (let j = 0; j < A[0].length; j++)
      m = Math.max(m, Math.abs(A[i][j] - B[i][j]));
  return m;
}

/** 深拷贝 */
export function clone(A: Mat): Mat {
  return A.map(row => row.slice());
}
