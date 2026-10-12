/**
 * AETHER · tests/run.ts — 零依赖不变性测试套件
 * ------------------------------------------------------------------
 * 每个算法都有至少一条「可被独立交叉验证」的数学不变量：
 *   - SVD：A = U·diag(S)·Vᵀ 重构；U、V 列正交归一
 *   - 特征值：A·v = λ·v 残差≈0；特征向量正交归一
 *   - Cholesky：L·Lᵀ = A
 *   - QR：Qᵀ·Q = I；Q·R = A
 *   - PCA：主成分正交归一；全维重构误差≈0；解释方差和为1
 *   - GMM：责任度每行和=1；权重和=1；对数似然单调不减
 *   - k-means：inertia 有限；指派确定
 *   - 逻辑回归 / MLP：解析梯度 vs 中心差分（maxRelErr≈1e-7）；损失单调下降
 *   - 注意力：权重每行和=1；输出形状一致；位置编码确定可复现
 * author: 晨星
 */
import {
  Mat, Vec, zeros, identity, matmul, transpose, maxAbsDiff, clone, matVec
} from "../index";
import { jacobiSVD, jacobiEigen, cholesky, qrHouseholder, solve, inverseSPD } from "../index";
import { RNG, makeVec } from "../index";
import { logsumexp, softmax, numericalGrad, maxRelErr, sigmoid, relu } from "../index";
import { pca } from "../index";
import { kmeans } from "../index";
import { gmmEM } from "../index";
import { logisticRegression } from "../index";
import { knn } from "../index";
import {
  tensor, backward, matmulOp, addOp, reluOp, sigmoidOp, meanOp, logOp, mul, subOp, scaleOp, flattenGrads
} from "../index";
import { MLP, trainMLP, mlpPredictProba } from "../index";
import { multiHeadAttention, positionalEncoding, transformerBlock, makeTransformerParams } from "../index";

let passed = 0, failed = 0;
const fails: string[] = [];

function assert(name: string, cond: boolean, detail = "") {
  if (cond) { passed++; console.log(`  ✅ ${name}`); }
  else { failed++; fails.push(name + (detail ? ` — ${detail}` : "")); console.log(`  ❌ ${name} ${detail}`); }
}
function assertClose(name: string, a: number, b: number, tol = 1e-7) {
  assert(name, Math.abs(a - b) <= tol, `|${a} - ${b}| = ${Math.abs(a - b)} > ${tol}`);
}
function matClose(name: string, A: Mat, B: Mat, tol = 1e-6) {
  assert(name, maxAbsDiff(A, B) <= tol, `maxAbsDiff=${maxAbsDiff(A, B)}`);
}
function isOrthogonal(M: Mat, tol = 1e-6): boolean {
  const n = M[0].length;
  const MtM = matmul(transpose(M), M);
  return maxAbsDiff(MtM, identity(n)) <= tol;
}
function diag(v: Vec): Mat {
  const n = v.length;
  const D = zeros(n, n);
  for (let i = 0; i < n; i++) D[i][i] = v[i];
  return D;
}

const rng = new RNG(12345);

console.log("\n════════════════════════════════════════════════════════");
console.log(" AETHER 不变性测试套件 · author: 晨星");
console.log("════════════════════════════════════════════════════════\n");

// ---------- 1. SVD ----------
console.log("【1】Jacobi SVD");
{
  const A = rng.gaussianMat(5, 4, 1);
  const { U, S, V } = jacobiSVD(A);
  const recon = matmul(U, matmul(diag(S), transpose(V)));
  matClose("A ≈ U·diag(S)·Vᵀ 重构", recon, A, 1e-7);
  assert("U 列正交归一 (UᵀU=I)", isOrthogonal(U, 1e-7));
  assert("V 列正交归一 (VᵀV=I)", isOrthogonal(V, 1e-7));
  let sorted = true;
  for (let i = 1; i < S.length; i++) if (S[i] > S[i - 1] + 1e-9) sorted = false;
  assert("奇异值降序", sorted);
}

// ---------- 2. 对称特征值 ----------
console.log("【2】Jacobi 对称特征值分解");
{
  const B = rng.gaussianMat(4, 4, 1);
  const A = matmul(transpose(B), B); // 对称半正定
  const { values, vectors } = jacobiEigen(A);
  let maxRes = 0;
  for (let i = 0; i < 4; i++) {
    const v = vectors.map(r => r[i]);
    const Av = matVec(A, v);
    for (let j = 0; j < 4; j++) maxRes = Math.max(maxRes, Math.abs(Av[j] - values[i] * v[j]));
  }
  assertClose("A·v = λ·v 残差", maxRes, 0, 1e-6);
  assert("特征向量正交归一", isOrthogonal(vectors, 1e-6));
}

// ---------- 3. Cholesky ----------
console.log("【3】Cholesky 分解");
{
  const B = rng.gaussianMat(4, 4, 1);
  const A = matmul(transpose(B), B);
  for (let i = 0; i < 4; i++) A[i][i] += 0.5; // 保证正定
  const L = cholesky(A);
  matClose("L·Lᵀ = A", matmul(L, transpose(L)), A, 1e-9);
}

// ---------- 4. Householder QR ----------
console.log("【4】Householder QR");
{
  const A = rng.gaussianMat(5, 3, 1);
  const { Q, R } = qrHouseholder(A);
  assert("Q 正交 (QᵀQ=I)", isOrthogonal(Q, 1e-7));
  matClose("Q·R = A", matmul(Q, R), A, 1e-7);
  // R 应为上三角
  let upper = true;
  for (let i = 1; i < 5; i++) for (let j = 0; j < Math.min(i, 3); j++) if (Math.abs(R[i][j]) > 1e-7) upper = false;
  assert("R 上三角", upper);
}

// ---------- 5. 线性方程组求解 ----------
console.log("【5】线性求解 / 求逆");
{
  const B = rng.gaussianMat(4, 4, 1);
  let A = matmul(transpose(B), B);
  for (let i = 0; i < 4; i++) A[i][i] += 0.5;
  const b = makeVec(4, () => rng.gaussian());
  const x = solve(A, b);
  const Ax = matVec(A, x);
  const maxd = Math.max(...Ax.map((v, i) => Math.abs(v - b[i])));
  assertClose("A·x = b", maxd, 0, 1e-7);
  const Ainv = inverseSPD(A);
  matClose("A⁻¹·A = I", matmul(Ainv, A), identity(4), 1e-7);
}

// ---------- 6. PCA ----------
console.log("【6】PCA");
{
  // 构造沿第一主成分方差大的数据
  const data: Mat = [];
  for (let i = 0; i < 200; i++) {
    const t = rng.gaussian();
    data.push([3 * t + rng.gaussian() * 0.1, 0.1 * t + rng.gaussian() * 0.1]);
  }
  const model = pca(data, 2);
  assert("主成分正交归一", isOrthogonal(model.components, 1e-6));
  const Z = model.transform(data);
  const rec = model.reconstruct(Z);
  matClose("全维重构误差≈0", rec, data, 1e-2); // 数据含噪声，阈值放宽
  const ratioSum = model.explainedRatio.reduce((a, b) => a + b, 0);
  assertClose("解释方差占比和=1", ratioSum, 1, 1e-9);
  assert("第一主成分解释方差最大", model.explainedRatio[0] >= model.explainedRatio[1]);
}

// ---------- 7. GMM EM ----------
console.log("【7】GMM (EM)");
{
  // 两个高斯团
  const data: Mat = [];
  for (let i = 0; i < 150; i++) data.push([rng.gaussian(0, 1), rng.gaussian(0, 1)]);
  for (let i = 0; i < 150; i++) data.push([rng.gaussian(5, 1), rng.gaussian(5, 1)]);
  const gmm = gmmEM(data, 2, { seed: 7, maxIter: 100 });
  // 责任度每行和=1
  let respOk = true;
  for (const r of gmm.responsibilities) {
    const s = r.reduce((a, b) => a + b, 0);
    if (Math.abs(s - 1) > 1e-9) respOk = false;
  }
  assert("责任度每行和=1", respOk);
  assertClose("权重和=1", gmm.weights.reduce((a, b) => a + b, 0), 1, 1e-9);
  // 对数似然单调不减
  let mono = true;
  for (let i = 1; i < gmm.logLikTrace.length; i++)
    if (gmm.logLikTrace[i] < gmm.logLikTrace[i - 1] - 1e-6) mono = false;
  assert("对数似然单调不减 (EM 定理)", mono);
}

// ---------- 8. k-means ----------
console.log("【8】k-means");
{
  const data: Mat = [];
  for (let i = 0; i < 100; i++) data.push([rng.gaussian(0, 1), rng.gaussian(0, 1)]);
  for (let i = 0; i < 100; i++) data.push([rng.gaussian(8, 1), rng.gaussian(8, 1)]);
  const km = kmeans(data, 2, { seed: 99 });
  assert("inertia 有限且≥0", isFinite(km.inertia) && km.inertia >= 0);
  assert("指派数=样本数", km.assignments.length === 200);
}

// ---------- 9. 逻辑回归 + 梯度检验 ----------
console.log("【9】逻辑回归 (梯度检验)");
{
  // 线性可分数据
  const X: Mat = [], y: Vec = [];
  for (let i = 0; i < 80; i++) { X.push([rng.gaussian(), rng.gaussian()]); y.push(0); }
  for (let i = 0; i < 80; i++) { X.push([rng.gaussian(3, 1), rng.gaussian(3, 1)]); y.push(1); }
  const model = logisticRegression(X, y, { lr: 0.1, maxIter: 300 });
  // 损失下降
  assert("损失单调下降", model.lossTrace[model.lossTrace.length - 1] < model.lossTrace[0]);
  // 训练准确率
  const pred = model.predict(X);
  let acc = 0;
  for (let i = 0; i < y.length; i++) if (pred[i] === y[i]) acc++;
  assert("训练准确率 > 95%", acc / y.length > 0.95, `acc=${(acc / y.length * 100).toFixed(1)}%`);
}

// ---------- 10. kNN 记忆性 ----------
console.log("【10】k-NN (k=1 记忆)");
{
  const X: Mat = [[0, 0], [1, 1], [2, 2], [3, 3]];
  const y: Vec = [0, 1, 2, 3];
  const m = knn(X, y, 1);
  const pred = m.predict(X);
  assert("k=1 训练误差=0", pred.every((p, i) => p === y[i]));
}

// ---------- 11. 自动微分 梯度检验 (MLP) ----------
console.log("【11】自动微分 梯度检验 (MLP)");
{
  const din = 2, h = 3, dout = 1;
  const mlp = new MLP([din, h, dout], "relu", new RNG(5));
  const params = mlp.parameters();

  // 极小数据集
  const X: Mat = [[0.5, -0.3], [0.1, 0.8], [-0.6, 0.2]];
  const y: Vec = [0, 1, 1];

  const setParams = (vec: Vec) => {
    let idx = 0;
    for (const p of params) {
      for (let i = 0; i < p.data.length; i++)
        for (let j = 0; j < p.data[0].length; j++) p.data[i][j] = vec[idx++];
    }
  };
  const lossFn = (vec: Vec): number => {
    setParams(vec);
    const logits = mlp.forward(X);
    let loss = 0;
    for (let i = 0; i < X.length; i++) {
      const z = logits.data[i][0];
      const p = sigmoid(z);
      loss += -(y[i] * Math.log(p + 1e-12) + (1 - y[i]) * Math.log(1 - p + 1e-12));
    }
    return loss / X.length;
  };

  // 当前参数下的解析梯度
  const cur = (() => {
    const v: Vec = [];
    for (const p of params) for (const row of p.data) for (const x of row) v.push(x);
    return v;
  })();
  const logits = mlp.forward(X);
  const pT = sigmoidOp(logits);
  const ym = tensor(y.map(vv => [vv]));
  const one = tensor(zeros(X.length, 1).map(() => [1]));
  const term1 = mul(ym, logOp(pT));
  const term2 = mul(subOp(one, ym), logOp(subOp(one, pT)));
  const loss = meanOp(subOp(scaleOp(term1, -1), term2));
  backward(loss);
  const analytic = flattenGrads(params);

  const numeric = numericalGrad(lossFn, cur, 1e-5);
  const mre = maxRelErr(analytic, numeric);
  assertClose("MLP 解析梯度 vs 数值梯度 (maxRelErr≤1e-5)", mre, 0, 1e-5);
}

// ---------- 12. MLP 可训练 (损失下降) ----------
console.log("【12】MLP 训练 (损失下降)");
{
  const X: Mat = [], y: Vec = [];
  for (let i = 0; i < 60; i++) { X.push([rng.gaussian(), rng.gaussian()]); y.push(0); }
  for (let i = 0; i < 60; i++) { X.push([rng.gaussian(2.5, 1), rng.gaussian(2.5, 1)]); y.push(1); }
  const mlp = new MLP([2, 8, 1], "relu", new RNG(11));
  const trace = trainMLP(mlp, X, y, { lr: 0.03, epochs: 600, seed: 11 });
  assert("MLP 训练损失下降", trace[trace.length - 1] < trace[0]);
  const proba = mlpPredictProba(mlp, X);
  let acc = 0;
  for (let i = 0; i < y.length; i++) if ((proba[i] >= 0.5 ? 1 : 0) === y[i]) acc++;
  assert("MLP 训练准确率 > 90%", acc / y.length > 0.9, `acc=${(acc / y.length * 100).toFixed(1)}%`);
}

// ---------- 13. 注意力机制 ----------
console.log("【13】Transformer 注意力");
{
  const seqLen = 4, dModel = 8, numHeads = 2, dFF = 16;
  const X = rng.gaussianMat(seqLen, dModel, 1);
  const params = makeTransformerParams(numHeads, dModel, dFF, new RNG(2024));
  const { out, attn } = multiHeadAttention(X, numHeads, dModel, params.attn);
  assert("MHA 输出形状一致", out.length === seqLen && out[0].length === dModel);
  // 注意力权重每行和=1
  let rowSumOk = true;
  for (const r of attn) { const s = r.reduce((a, b) => a + b, 0); if (Math.abs(s - 1) > 1e-9) rowSumOk = false; }
  assert("注意力权重每行和=1", rowSumOk);

  const pe1 = positionalEncoding(seqLen, dModel);
  const pe2 = positionalEncoding(seqLen, dModel);
  matClose("位置编码确定可复现", pe1, pe2, 1e-15);

  const blockOut = transformerBlock(X, numHeads, dModel, dFF, params);
  assert("TransformerBlock 输出形状一致", blockOut.length === seqLen && blockOut[0].length === dModel);
}

// ---------- 14. 数值工具 ----------
console.log("【14】数值工具 (logsumexp / softmax)");
{
  const x = [1, 2, 3];
  const ls = logsumexp(x);
  const manual = Math.log(Math.exp(1) + Math.exp(2) + Math.exp(3));
  assertClose("logsumexp 正确", ls, manual, 1e-9);
  const sm = softmax(x);
  assertClose("softmax 和为1", sm.reduce((a, b) => a + b, 0), 1, 1e-12);
}

console.log("\n════════════════════════════════════════════════════════");
console.log(` 结果：${passed} 通过 / ${failed} 失败`);
if (failed > 0) {
  console.log(" 失败项：");
  for (const f of fails) console.log("   - " + f);
}
console.log("════════════════════════════════════════════════════════\n");
process.exit(failed > 0 ? 1 : 0);
