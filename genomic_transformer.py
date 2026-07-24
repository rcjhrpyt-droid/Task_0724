"""
=============================================================================
GBLUP-Residual Transformer — Wheat599 10-Fold CV
=============================================================================
两阶段:
  1. GBLUP: 捕获线性加性效应
  2. Transformer: 学习 GBLUP 残差中的非线性模式

不确定性估计 (MC Dropout):
  对每个测试样本, 30 次 Dropout 前向传播 → 残差预测分布
  → 双侧 t 检验 H0: μ_residual = 0
  → p < 0.05 接受 Transformer 修正; 否则仅用 GBLUP

输出:
  - results.txt: 最终汇总表
  - results_full.txt: 每折详细信息
=============================================================================
"""

import os, time, warnings, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy import stats
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

# ======================== 超参数 ========================
CONV_KERNEL   = 12
CONV_STRIDE   = 12
N_MARKERS     = 1279
SEQ_LEN       = (N_MARKERS - CONV_KERNEL) // CONV_STRIDE + 1  # = 106

D             = 24       # embed dim
H             = 4        # heads
L             = 2        # layers
FF            = 96       # FFN dim
DROPOUT       = 0.25
MC_SAMPLES    = 30       # MC Dropout 采样次数
P_THRESHOLD   = 0.05     # 显著性阈值

REG_HIDDEN    = 48
BATCH_SIZE    = 32
LR            = 1e-3
WEIGHT_DECAY  = 0.02
MAX_EPOCHS    = 80
PATIENCE      = 20

N_FOLDS       = 10
VAL_SPLIT     = 0.1
SEED          = 42
GBLUP_H2      = 0.5   # GBLUP 遗传力先验

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================== 数据加载 ========================
def load_data():
    print("\n[1/5] Loading data...")
    X = np.loadtxt("data/genotype.txt", dtype=np.float64, delimiter="\t")
    Y = {}
    for i in range(1, 5):
        y = np.loadtxt(f"data/phenotype_env{i}.txt", dtype=np.float64)
        Y[f"env{i}"] = y
        print(f"  env{i}: mean={y.mean():.4f}, sd={y.std():.4f}")
    print(f"  Genotype: {X.shape[0]} x {X.shape[1]} markers")
    return X, Y

# ======================== GBLUP ========================
def compute_G(X):
    Xc = X - X.mean(axis=0)
    return Xc @ Xc.T / Xc.shape[1]

def gblup_predict(G, y, tr_idx, te_idx, h2=GBLUP_H2):
    n_tr = len(tr_idx)
    G_tr   = G[np.ix_(tr_idx, tr_idx)]
    G_tetr = G[np.ix_(te_idx, tr_idx)]
    mu = y[tr_idx].mean()
    yc = y[tr_idx] - mu
    lam = (1.0 - h2) / h2
    alpha = np.linalg.solve(G_tr + lam * np.eye(n_tr), yc)
    return mu + G_tetr @ alpha

# ======================== Conv1D + Transformer ========================
class ResidualTransformer(nn.Module):
    """学习 GBLUP 残差的 Transformer"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, D, CONV_KERNEL, CONV_STRIDE)
        self.bn   = nn.BatchNorm1d(D)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=H, dim_feedforward=FF,
            dropout=DROPOUT, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=L)
        self.head = nn.Sequential(
            nn.Linear(D, REG_HIDDEN), nn.GELU(),
            nn.Dropout(DROPOUT), nn.Linear(REG_HIDDEN, 1))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv(x.unsqueeze(1))
        x = self.bn(x)
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.head(x).squeeze(-1)

# ======================== MC Dropout ========================
@torch.no_grad()
def mc_dropout_predict(model, X_tensor, n_samples=MC_SAMPLES):
    """多次 Dropout 前向传播 → 残差预测分布 → 均值/标准差/p值"""
    model.train()  # 保持 Dropout 激活
    preds = np.zeros((n_samples, X_tensor.shape[0]))
    for i in range(n_samples):
        preds[i] = model(X_tensor.to(DEVICE)).detach().cpu().numpy()
    model.eval()

    mu  = preds.mean(axis=0)
    std = preds.std(axis=0, ddof=1)

    # 双侧 t 检验 H0: 残差 = 0
    se = std / np.sqrt(n_samples)
    t_vals = mu / (se + 1e-8)
    p_vals = 2.0 * stats.t.sf(np.abs(t_vals), df=n_samples - 1)

    return mu, std, p_vals

# ======================== 训练 ========================
def train_epoch(model, Xm, yv, opt):
    model.train()
    n = Xm.shape[0]; perm = torch.randperm(n); tl = 0.0; nb = 0
    for s in range(0, n, BATCH_SIZE):
        e = min(s + BATCH_SIZE, n); idx = perm[s:e]
        loss = F.mse_loss(
            model(Xm[idx].to(DEVICE)),
            yv[idx].to(DEVICE))
        opt.zero_grad(); loss.backward(); opt.step()
        tl += loss.item(); nb += 1
    return tl / nb

@torch.no_grad()
def evaluate(model, Xm, yv):
    model.eval()
    p = model(Xm.to(DEVICE)).detach().cpu().numpy()
    t = yv.cpu().numpy()
    return np.corrcoef(p, t)[0, 1], np.sqrt(np.mean((p - t)**2))

def train_model(model, trX, trY, vaX, vaY, verbose=False):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=5)
    best_pcc = -float("inf"); best_wts = None; wait = 0; best_ep = 0
    for ep in range(1, MAX_EPOCHS + 1):
        _ = train_epoch(model, trX, trY, opt)
        vp, _ = evaluate(model, vaX, vaY)
        sch.step(vp)
        if vp > best_pcc:
            best_pcc = vp; best_wts = copy.deepcopy(model.state_dict())
            wait = 0; best_ep = ep
        else:
            wait += 1
            if wait >= PATIENCE: break
        if verbose and (ep % 20 == 0 or ep == 1):
            print(f"      Ep {ep:3d} | Val PCC={vp:.4f}")
    model.load_state_dict(best_wts)
    return best_ep

# ======================== 单环境 CV ========================
def run_one_env(X_full, y_full, G_full, env_name, out_f):
    """对一个环境跑完整 10-Fold CV，返回所有结果字典和写入文件"""
    out_f.write(f"\n{'='*70}\n")
    out_f.write(f"  {env_name.upper()} — GBLUP-Residual Transformer\n")
    out_f.write(f"{'='*70}\n\n")

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    rows = []  # 存储每折结果
    t0 = time.time()

    for fold, (tr_idx, te_idx) in enumerate(kf.split(X_full)):
        rng = np.random.RandomState(SEED + fold)
        rest_idx = np.array(tr_idx); rng.shuffle(rest_idx)
        n_val = max(1, int(len(rest_idx) * VAL_SPLIT))
        va_idx = rest_idx[:n_val]; tr_idx_inner = rest_idx[n_val:]

        # ---- Stage 1: GBLUP ----
        g_pred_te = gblup_predict(G_full, y_full, tr_idx_inner, te_idx)
        g_pcc  = np.corrcoef(g_pred_te, y_full[te_idx])[0, 1]
        g_rmse = np.sqrt(np.mean((g_pred_te - y_full[te_idx])**2))

        # 计算训练残差
        g_pred_tr = gblup_predict(G_full, y_full, tr_idx_inner, tr_idx_inner)
        resid_tr  = y_full[tr_idx_inner] - g_pred_tr
        g_pred_va = gblup_predict(G_full, y_full, tr_idx_inner, va_idx)
        resid_va  = y_full[va_idx] - g_pred_va

        # ---- Stage 2: Transformer 学残差 ----
        tr_X = torch.tensor(X_full[tr_idx_inner], dtype=torch.float32)
        tr_R = torch.tensor(resid_tr, dtype=torch.float32)
        va_X = torch.tensor(X_full[va_idx], dtype=torch.float32)
        va_R = torch.tensor(resid_va, dtype=torch.float32)
        te_X = torch.tensor(X_full[te_idx], dtype=torch.float32)

        model = ResidualTransformer()
        best_ep = train_model(model, tr_X, tr_R, va_X, va_R)

        # ---- MC Dropout 假设检验 ----
        r_mu, _, p_vals = mc_dropout_predict(model, te_X, MC_SAMPLES)

        # ---- 融合预测 ----
        bayes_pred = g_pred_te.copy()
        accepted   = p_vals < P_THRESHOLD
        bayes_pred[accepted] += r_mu[accepted]
        accept_rate = accepted.mean()

        bayes_pcc  = np.corrcoef(bayes_pred, y_full[te_idx])[0, 1]
        bayes_rmse = np.sqrt(np.mean((bayes_pred - y_full[te_idx])**2))

        # 残差本身的 PCC（Transformer 直接预测残差 vs 真实残差）
        r_single = model(te_X.to(DEVICE)).detach().cpu().numpy()
        resid_true = y_full[te_idx] - g_pred_te
        resid_pcc = np.corrcoef(r_single, resid_true)[0, 1]

        delta = bayes_pcc - g_pcc

        rows.append({
            "fold": fold+1,
            "gblup_pcc": g_pcc, "gblup_rmse": g_rmse,
            "bayes_pcc": bayes_pcc, "bayes_rmse": bayes_rmse,
            "resid_pcc": resid_pcc,
            "delta": delta, "accept_rate": accept_rate,
            "epoch": best_ep, "n_train": len(tr_idx_inner),
            "n_val": len(va_idx), "n_test": len(te_idx),
        })

        line = (f"  Fold {fold+1:2d}/{N_FOLDS} | "
                f"GBLUP PCC={g_pcc:.4f} RMSE={g_rmse:.4f} | "
                f"GBT PCC={bayes_pcc:.4f} RMSE={bayes_rmse:.4f} | "
                f"Δ={delta:+.4f} | "
                f"Resid PCC={resid_pcc:.4f} | "
                f"Accept={accept_rate:.1%} | Ep={best_ep}")
        print(line)
        out_f.write(line + "\n")

    elapsed = time.time() - t0

    gblup_pccs  = [r["gblup_pcc"]  for r in rows]
    gblup_rmses = [r["gblup_rmse"] for r in rows]
    bayes_pccs  = [r["bayes_pcc"]  for r in rows]
    bayes_rmses = [r["bayes_rmse"] for r in rows]
    deltas      = [r["delta"]      for r in rows]
    accepts     = [r["accept_rate"] for r in rows]

    summary = (
        f"\n  >>> {env_name.upper()} Summary ({elapsed:.0f}s)\n"
        f"  GBLUP:    PCC = {np.mean(gblup_pccs):.4f} +/- {np.std(gblup_pccs):.4f}  |  "
        f"RMSE = {np.mean(gblup_rmses):.4f} +/- {np.std(gblup_rmses):.4f}\n"
        f"  GBT:      PCC = {np.mean(bayes_pccs):.4f} +/- {np.std(bayes_pccs):.4f}  |  "
        f"RMSE = {np.mean(bayes_rmses):.4f} +/- {np.std(bayes_rmses):.4f}\n"
        f"  Δ PCC:    {np.mean(deltas):+.4f} +/- {np.std(deltas):.4f}\n"
        f"  Resid PCC:{np.mean([r['resid_pcc'] for r in rows]):.4f}\n"
        f"  Accept:   {np.mean(accepts):.1%}\n"
    )
    print(summary)
    out_f.write(summary)

    return {
        "env": env_name,
        "gblup_mean": np.mean(gblup_pccs), "gblup_std": np.std(gblup_pccs),
        "bayes_mean": np.mean(bayes_pccs), "bayes_std": np.std(bayes_pccs),
        "gblup_rmse_mean": np.mean(gblup_rmses), "gblup_rmse_std": np.std(gblup_rmses),
        "bayes_rmse_mean": np.mean(bayes_rmses), "bayes_rmse_std": np.std(bayes_rmses),
        "delta_mean": np.mean(deltas), "delta_std": np.std(deltas),
        "accept_mean": np.mean(accepts),
        "resid_pcc_mean": np.mean([r["resid_pcc"] for r in rows]),
        "rows": rows, "time": elapsed,
    }

# ======================== 主流程 ========================
def main():
    print("\n" + "=" * 70)
    print("  GBLUP-Residual Transformer — Wheat599 10-Fold CV")
    print("=" * 70)
    print(f"  Device: {DEVICE}")
    print(f"  Architecture: GBLUP → Residuals → Conv1D+Transformer → MC Dropout")
    print(f"  Conv1D: {N_MARKERS}→{SEQ_LEN} tokens | Transformer: {L}L×{H}H | MC: {MC_SAMPLES}samples")
    print(f"  p < {P_THRESHOLD}: accept Transformer correction")

    # 加载
    X, Y = load_data()
    G = compute_G(X)
    X_f32 = X.astype(np.float32)

    n_params = sum(p.numel() for p in ResidualTransformer().parameters())
    print(f"\n[2/5] Residual Transformer: {n_params:,} params")

    # 结果文件
    f_full = open("results_full.txt", "w", encoding="utf-8")
    f_full.write("GBLUP-Residual Transformer — Wheat599 10-Fold CV\n")
    f_full.write("=" * 70 + "\n")
    f_full.write(f"Device: {DEVICE}\n")
    f_full.write(f"Arch: GBLUP → Residuals → Conv1D+Transformer → MC Dropout\n")
    f_full.write(f"Conv1D: {N_MARKERS}→{SEQ_LEN} tokens, "
                 f"Transformer: {L}L×{H}H, MC: {MC_SAMPLES}samples\n")
    f_full.write(f"p threshold: {P_THRESHOLD}, h2 prior: {GBLUP_H2}\n\n")

    print("\n[3/5] Running 10-Fold CV for 4 environments...")
    all_env = {}

    for env_name in ["env1", "env2", "env3", "env4"]:
        print(f"\n{'─'*70}")
        print(f"  {env_name.upper()}")
        print(f"{'─'*70}")
        all_env[env_name] = run_one_env(X_f32, Y[env_name], G, env_name, f_full)

    f_full.close()

    # ======================== 汇总表 ========================
    summary_lines = [
        "\n\n" + "=" * 78,
        "  GBLUP-Residual Transformer — 10-Fold CV Final Summary",
        "=" * 78,
        f"  {'Env':>6} | {'GBLUP PCC':>18} | {'GBT PCC':>18} | {'Δ PCC':>8} | {'Resid PCC':>10} | {'Accept':>8}",
        "-" * 78,
    ]

    gblup_avgs = []; bayes_avgs = []
    for env in ["env1", "env2", "env3", "env4"]:
        r = all_env[env]
        gblup_avgs.append(r["gblup_mean"])
        bayes_avgs.append(r["bayes_mean"])
        summary_lines.append(
            f"  {env:>6} | {r['gblup_mean']:8.4f}+/-{r['gblup_std']:.4f} | "
            f"{r['bayes_mean']:8.4f}+/-{r['bayes_std']:.4f} | "
            f"{r['delta_mean']:+7.4f} | "
            f"{r['resid_pcc_mean']:10.4f} | {r['accept_mean']:7.1%}")
    summary_lines.append("-" * 78)
    summary_lines.append(
        f"  {'Avg':>6} | {np.mean(gblup_avgs):8.4f}             | "
        f"{np.mean(bayes_avgs):8.4f}             | "
        f"{np.mean(bayes_avgs)-np.mean(gblup_avgs):+7.4f} | {'─':>10} | {'─':>8}")
    summary_lines.append("=" * 78)

    # RMSE 汇总
    summary_lines.append(f"\n  {'Env':>6} | {'GBLUP RMSE':>18} | {'GBT RMSE':>18}")
    summary_lines.append("-" * 78)
    for env in ["env1", "env2", "env3", "env4"]:
        r = all_env[env]
        summary_lines.append(
            f"  {env:>6} | {r['gblup_rmse_mean']:8.4f}+/-{r['gblup_rmse_std']:.4f} | "
            f"{r['bayes_rmse_mean']:8.4f}+/-{r['bayes_rmse_std']:.4f}")
    summary_lines.append("=" * 78)

    # 用时
    total_t = sum(r["time"] for r in all_env.values())
    summary_lines.append(f"\n  Total time: {total_t:.0f}s ({total_t/60:.1f} min)")

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    # 写入 results.txt
    with open("results.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"\n[5/5] Results saved:")
    print(f"  results.txt       — 最终汇总表")
    print(f"  results_full.txt  — 每折详细信息")
    print("\nDone.")

if __name__ == "__main__":
    main()
