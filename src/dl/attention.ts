/**
 * AETHER · dl/attention.ts
 * ------------------------------------------------------------------
 * Transformer 核心构件（与 Vaswani et al. 2017 "Attention Is All You Need" 同源）：
 *   - 缩放点积注意力（Scaled Dot-Product Attention）
 *   - 多头注意力（Multi-Head Attention）
 *   - 正弦位置编码（Positional Encoding）
 *   - Transformer Block（Pre/Post-LN + 前馈网络）
 * 不变性：注意力权重每行和为 1（softmax）；输出形状与输入一致；位置编码确定可复现。
 * author: 晨星
 */
import { Mat, Vec, zeros, matmul, transpose, scale, add } from "../core/matrix";
import { softmax, relu } from "../core/numerics";
import { RNG } from "../core/random";

/** 正弦位置编码：pe[pos,2i]=sin(pos/10000^(2i/d)), pe[pos,2i+1]=cos(...) */
export function positionalEncoding(seqLen: number, dModel: number): Mat {
  const pe = zeros(seqLen, dModel);
  for (let pos = 0; pos < seqLen; pos++) {
    for (let i = 0; i < dModel; i++) {
      const denom = Math.pow(10000, (2 * Math.floor(i / 2)) / dModel);
      pe[pos][i] = i % 2 === 0 ? Math.sin(pos / denom) : Math.cos(pos / denom);
    }
  }
  return pe;
}

/** 层归一化（沿最后一维）。 */
export function layerNorm(X: Mat, eps = 1e-5): Mat {
  const n = X.length, d = X[0].length;
  const out = zeros(n, d);
  for (let i = 0; i < n; i++) {
    let mean = 0;
    for (let j = 0; j < d; j++) mean += X[i][j];
    mean /= d;
    let varr = 0;
    for (let j = 0; j < d; j++) varr += (X[i][j] - mean) * (X[i][j] - mean);
    varr /= d;
    const sd = Math.sqrt(varr + eps);
    for (let j = 0; j < d; j++) out[i][j] = (X[i][j] - mean) / sd;
  }
  return out;
}

/** 缩放点积注意力：out = softmax(QKᵀ/√d_k)·V。返回输出与注意力权重。 */
export function scaledDotProductAttention(Q: Mat, K: Mat, V: Mat): { out: Mat; attn: Mat } {
  const dk = Q[0].length;
  const scores = matmul(Q, transpose(K));
  const scaled = scores.map(r => r.map(v => v / Math.sqrt(dk)));
  const attn = scaled.map(row => softmax(row)); // 每行和为 1
  const out = matmul(attn, V);
  return { out, attn };
}

export interface AttnProj {
  Wq: Mat; Wk: Mat; Wv: Mat; Wo: Mat;
}
export interface FFNProj {
  W1: Mat; b1: Vec; W2: Mat; b2: Vec;
}
export interface TransformerParams {
  attn: AttnProj;
  ffn: FFNProj;
}

/** 生成确定性随机投影参数（Xavier 初始化）。 */
export function makeTransformerParams(
  numHeads: number, dModel: number, dFF: number, rng: RNG = new RNG(2024)
): TransformerParams {
  const s = Math.sqrt(2 / (dModel + dModel));
  const lin = (din: number, dout: number): Mat => rng.gaussianMat(dout, din, Math.sqrt(2 / (din + dout)));
  return {
    attn: {
      Wq: rng.gaussianMat(dModel, dModel, s),
      Wk: rng.gaussianMat(dModel, dModel, s),
      Wv: rng.gaussianMat(dModel, dModel, s),
      Wo: rng.gaussianMat(dModel, dModel, s)
    },
    ffn: {
      W1: lin(dModel, dFF),
      b1: new Array(dFF).fill(0),
      W2: lin(dFF, dModel),
      b2: new Array(dModel).fill(0)
    }
  };
}

/** 多头注意力（输出形状 = 输入形状）。 */
export function multiHeadAttention(X: Mat, numHeads: number, dModel: number, proj: AttnProj): { out: Mat; attn: Mat } {
  const dk = dModel / numHeads;
  const Q = matmul(X, transpose(proj.Wq));
  const K = matmul(X, transpose(proj.Wk));
  const V = matmul(X, transpose(proj.Wv));
  const headsOut: Mat[] = [];
  let attnAgg: Mat = zeros(X.length, X.length);
  for (let h = 0; h < numHeads; h++) {
    const c0 = h * dk, c1 = (h + 1) * dk;
    const Qh = Q.map(r => r.slice(c0, c1));
    const Kh = K.map(r => r.slice(c0, c1));
    const Vh = V.map(r => r.slice(c0, c1));
    const { out, attn } = scaledDotProductAttention(Qh, Kh, Vh);
    headsOut.push(out);
    for (let i = 0; i < attn.length; i++)
      for (let j = 0; j < attn.length; j++) attnAgg[i][j] += attn[i][j] / numHeads;
  }
  // 拼接头
  const concat = X.map((_, i) => {
    const row: number[] = [];
    for (let h = 0; h < numHeads; h++) for (let j = 0; j < dk; j++) row.push(headsOut[h][i][j]);
    return row;
  });
  const out = matmul(concat, transpose(proj.Wo));
  return { out, attn: attnAgg };
}

function ffnForward(X: Mat, p: FFNProj): Mat {
  const h = X.map(row => {
    const z: Vec = new Array(p.W1[0].length).fill(0);
    for (let j = 0; j < p.W1[0].length; j++) {
      let s = p.b1[j];
      for (let k = 0; k < row.length; k++) s += row[k] * p.W1[k][j];
      z[j] = relu(s);
    }
    return z;
  });
  const out = h.map(row => {
    const z: Vec = new Array(p.W2[0].length).fill(0);
    for (let j = 0; j < p.W2[0].length; j++) {
      let s = p.b2[j];
      for (let k = 0; k < row.length; k++) s += row[k] * p.W2[k][j];
      z[j] = s;
    }
    return z;
  });
  return out;
}

/** 单个 Transformer Block（Post-LN：x → LN(x+Attn) → LN(·+FFN)）。 */
export function transformerBlock(X: Mat, numHeads: number, dModel: number, dFF: number, params: TransformerParams): Mat {
  const pe = positionalEncoding(X.length, dModel);
  let x = add(X, pe);
  const { out: attnOut } = multiHeadAttention(x, numHeads, dModel, params.attn);
  x = layerNorm(add(x, attnOut));
  const ffnOut = ffnForward(x, params.ffn);
  x = layerNorm(add(x, ffnOut));
  return x;
}
