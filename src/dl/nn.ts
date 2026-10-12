/**
 * AETHER · dl/nn.ts
 * ------------------------------------------------------------------
 * 多层感知机（MLP），全部建立在自动微分引擎之上（dl/autograd）。
 *   - 前向：h = act(X·W₁+b₁) → … → logits = h·W_L+b_L
 *   - 训练：BCE 损失 + Adam（张量级），损失单调下降为不变量
 * 参考：Goodfellow §6, §8；Bishop §5
 * author: 晨星
 */
import { Mat, Vec, zeros } from "../core/matrix";
import { RNG } from "../core/random";
import {
  Tensor, tensor, backward, matmulOp, addOp, subOp, scaleOp,
  reluOp, sigmoidOp, tanhOp, softmaxRowsOp, meanOp, logOp, mul
} from "./autograd";

export type Activation = "relu" | "sigmoid" | "tanh";

interface Layer {
  W: Tensor;
  b: Tensor;
}

export class MLP {
  layers: Layer[] = [];
  act: Activation;

  constructor(layerSizes: number[], act: Activation = "relu", rng: RNG = new RNG(1)) {
    this.act = act;
    for (let i = 0; i < layerSizes.length - 1; i++) {
      const din = layerSizes[i], dout = layerSizes[i + 1];
      // Xavier/Glorot 初始化；W 形状 (din × dout)，前向 h·W + b
      const scale = Math.sqrt(2 / (din + dout));
      const Wdata: Mat = rng.gaussianMat(din, dout, scale);
      const bdata: Mat = zeros(1, dout);
      this.layers.push({ W: tensor(Wdata), b: tensor(bdata) });
    }
  }

  /** 前向（返回 logits）。 */
  forward(X: Mat): Tensor {
    let h: Tensor = tensor(X);
    for (let i = 0; i < this.layers.length; i++) {
      const { W, b } = this.layers[i];
      h = addOp(matmulOp(h, W), b);
      const isLast = i === this.layers.length - 1;
      if (!isLast) {
        if (this.act === "relu") h = reluOp(h);
        else if (this.act === "sigmoid") h = sigmoidOp(h);
        else h = tanhOp(h);
      }
    }
    return h;
  }

  parameters(): Tensor[] {
    const p: Tensor[] = [];
    for (const l of this.layers) { p.push(l.W); p.push(l.b); }
    return p;
  }
}

export interface TrainOpts {
  lr?: number;
  epochs?: number;
  seed?: number;
  batchSize?: number;
}

/**
 * 用 Adam 训练 MLP 做二分类（BCE）。
 * 返回每若干步的损失记录（用于「损失单调下降」不变量检验）。
 */
export function trainMLP(mlp: MLP, X: Mat, y: Vec, opts: TrainOpts = {}): number[] {
  const lr = opts.lr ?? 0.05;
  const epochs = opts.epochs ?? 400;
  const n = X.length;
  const params = mlp.parameters();
  const b1 = 0.9, b2 = 0.999, eps = 1e-8;
  const m: Mat[] = params.map(p => zeros(p.data.length, p.data[0].length));
  const v: Mat[] = params.map(p => zeros(p.data.length, p.data[0].length));

  const ym = tensor(y.map(vv => [vv]));
  const one = tensor(zeros(n, 1).map(() => [1]));

  const lossTrace: number[] = [];

  for (let epoch = 0; epoch < epochs; epoch++) {
    const logits = mlp.forward(X);
    const p = sigmoidOp(logits);
    const term1 = mul(ym, logOp(p));
    const term2 = mul(subOp(one, ym), logOp(subOp(one, p)));
    const loss = meanOp(subOp(scaleOp(term1, -1), term2));

    backward(loss);

    const t = epoch + 1;
    for (let pi = 0; pi < params.length; pi++) {
      const pr = params[pi];
      const g = pr.grad;
      const mn = m[pi], vn = v[pi];
      for (let i = 0; i < pr.data.length; i++)
        for (let j = 0; j < pr.data[0].length; j++) {
          mn[i][j] = b1 * mn[i][j] + (1 - b1) * g[i][j];
          vn[i][j] = b2 * vn[i][j] + (1 - b2) * g[i][j] * g[i][j];
          const mhat = mn[i][j] / (1 - Math.pow(b1, t));
          const vhat = vn[i][j] / (1 - Math.pow(b2, t));
          pr.data[i][j] -= lr * mhat / (Math.sqrt(vhat) + eps);
        }
    }
    if (epoch % Math.max(1, Math.floor(epochs / 10)) === 0) lossTrace.push(loss.data[0][0]);
  }
  return lossTrace;
}

/** 前向推理（概率）。 */
export function mlpPredictProba(mlp: MLP, X: Mat): Vec {
  const logits = mlp.forward(X);
  const n = logits.data.length;
  const out: Vec = new Array(n);
  for (let i = 0; i < n; i++) {
    const z = logits.data[i][0];
    out[i] = z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
  }
  return out;
}
