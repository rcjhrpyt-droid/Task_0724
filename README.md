# GBLUP-Residual Transformer — Wheat599 10-Fold CV

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org)

GBLUP 与 Transformer 的两阶段混合基因组预测模型，针对 Wheat599 数据集（599 个小麦样本 × 1279 个 DArT 二元标记 × 4 环境产量表型）。

---

## 方法概述

```
Stage 1: GBLUP
  └─→ 捕获线性加性效应，输出 ŷ_GBLUP

Stage 2: Conv1D + Transformer
  └─→ 学习真实残差 r = y − ŷ_GBLUP 中的非线性模式

融合 (MC Dropout):
  └─→ 对每个测试样本，30 次 Dropout 前向传播得到残差预测分布
  └─→ 双侧 t 检验 H₀: μ_residual = 0
  └─→ p < 0.05 → 接受 Transformer 修正: ŷ_final = ŷ_GBLUP + r̂
  └─→ p ≥ 0.05 → 仅用 GBLUP
```

### 架构参数

| 模块 | 参数 |
|------|------|
| Conv1D | kernel=12, stride=12, 1279 → 106 tokens |
| Embedding | 24 → 24 (数值嵌入) |
| Transformer Encoder | 2 层 × 4 heads, FFN=96 |
| 正则化 | Dropout=0.25, Weight Decay=0.02 |
| MC Dropout | n_samples=30, p_threshold=0.05 |
| 训练 | AdamW, lr=1e-3, ReduceLROnPlateau, 80 epochs, patience=20 |
| 交叉验证 | 10-Fold, 10% 内部验证集 |
| GBLUP 先验 | h²=0.5 |

---

## 结果

### 10-Fold CV 汇总

#### PCC 对比

| Env | GBLUP PCC | GBT PCC | Δ PCC |
|-----|-----------|---------|-------|
| env1 | 0.4464±0.0821 | 0.4782±0.0894 | +0.0318 |
| env2 | 0.4756±0.0977 | 0.4539±0.1054 | -0.0217 |
| env3 | 0.3882±0.1066 | 0.3832±0.0956 | -0.0050 |
| env4 | 0.4274±0.1140 | 0.4165±0.1129 | -0.0109 |
| **Avg** | **0.4344** | **0.4330** | **-0.0014** |

#### RMSE 对比

| Env | GBLUP RMSE | GBT RMSE | Δ RMSE |
|-----|------------|----------|--------|
| env1 | 0.9117±0.0506 | 0.8873±0.0611 | -0.0244 |
| env2 | 0.8958±0.0973 | 0.8981±0.0983 | +0.0023 |
| env3 | 0.9238±0.1385 | 0.9251±0.1380 | +0.0013 |
| env4 | 0.9090±0.0926 | 0.9143±0.0911 | +0.0053 |
| **Avg** | **0.9101** | **0.9062** | **-0.0039** |


---

## 快速开始

### 1. 环境依赖

```bash
pip install numpy scipy scikit-learn torch pandas
```

### 2. 数据准备

将原始数据文件放入 `data_raw/`：

```
data_raw/
├── wheat599_X.pkl    # 基因型矩阵（pandas DataFrame, 599×1279）
├── wheat1.Y          # 环境1表型（首行表头，后续为 float）
├── wheat2.Y
├── wheat3.Y
└── wheat4.Y
```

运行转换脚本：

```bash
python convert_data.py
```

生成 `data/` 目录。

### 3. 运行

```bash
python genomic_transformer.py
```

输出：
- `results.txt` — 最终汇总表
- `results_full.txt` — 每折详细信息



---

## 文件说明

| 文件 | 说明 |
|------|------|
| `genomic_transformer.py` | 核心代码（GBLUP + Transformer + 10-Fold CV） |
| `convert_data.py` | 数据格式转换脚本 |
| `results.txt` | 10-Fold CV 最终汇总表 |
| `results_full.txt` | 每折详细信息（PCC / RMSE / Accept Rate / Epoch） |

---


