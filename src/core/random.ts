/**
 * AETHER · core/random.ts
 * ------------------------------------------------------------------
 * 可复现随机数发生器（seeded RNG）。
 *   - mulberry32：快速、质量足够的 32-bit PRNG（用于可复现实验）
 *   - 标准正态：Box–Muller 变换
 * 可复现性是世界顶级实验系统的硬要求（随机种子即实验契约）。
 * author: 晨星
 */
import { Vec } from "./matrix";

export class RNG {
  private state: number;
  constructor(seed = 0x9e3779b9) {
    this.state = seed >>> 0;
  }
  /** [0,1) 均匀分布 */
  next(): number {
    let t = (this.state += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  /** [a,b) 均匀分布 */
  uniform(a: number, b: number): number {
    return a + (b - a) * this.next();
  }
  /** 整数 [0,n) */
  randInt(n: number): number {
    return Math.floor(this.next() * n);
  }
  /** 标准正态 N(0,1)（Box–Muller） */
  gaussian(mean = 0, std = 1): number {
    let u = 0, v = 0;
    while (u === 0) u = this.next();
    while (v === 0) v = this.next();
    const z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    return mean + std * z;
  }
  /** 填充满给定形状的独立标准正态矩阵 */
  gaussianMat(rows: number, cols: number, std = 1): number[][] {
    const out: number[][] = new Array(rows);
    for (let i = 0; i < rows; i++) {
      const row = new Array(cols);
      for (let j = 0; j < cols; j++) row[j] = this.gaussian(0, std);
      out[i] = row;
    }
    return out;
  }
  /** 原地 Fisher–Yates 洗牌 */
  shuffle<T>(arr: T[]): T[] {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = this.randInt(i + 1);
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  }
  /** 从数组中随机采样 k 个不重复元素 */
  sample<T>(arr: T[], k: number): T[] {
    const idx = [...Array(arr.length).keys()];
    this.shuffle(idx);
    return idx.slice(0, k).map(i => arr[i]);
  }
}

export function makeVec(n: number, f: (i: number) => number): Vec {
  const v = new Array(n);
  for (let i = 0; i < n; i++) v[i] = f(i);
  return v;
}
