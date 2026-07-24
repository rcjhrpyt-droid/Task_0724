"""
=============================================================================
convert_data.py — 将原始 Wheat599 数据转为 genomic_transformer.py 可读格式
=============================================================================
输入（手动放置在 data_raw/ 下）：
  - wheat599_X.pkl       : 599 × 1279 DArT 二元标记矩阵（pandas DataFrame）
  - wheat1.Y ~ wheat4.Y  : 4 环境产量表型（已标准化，第 1 行为表头）

输出（生成到 data/ 下）：
  - genotype.txt         : 599 × 1279，tab 分隔，仅 0/1
  - phenotype_env1.txt   : 599 行，每行一个 float
  - phenotype_env2.txt
  - phenotype_env3.txt
  - phenotype_env4.txt

用法：
  python convert_data.py
=============================================================================
"""

import os
import numpy as np
import pandas as pd

def main():
    # ---- 获取脚本所在目录 ----
    base_dir = os.path.dirname(os.path.abspath(__file__))

    raw_dir  = os.path.join(base_dir, "data_raw")
    out_dir  = os.path.join(base_dir, "data")
    os.makedirs(out_dir, exist_ok=True)

    # ==================== 1. 基因型 ====================
    print("[1/2] Converting genotype...")
    pkl_path = os.path.join(raw_dir, "wheat599_X.pkl")
    X = pd.read_pickle(pkl_path)

    # 确保是 0/1 整数
    X = X.astype(int)

    print(f"  Shape: {X.shape[0]} samples x {X.shape[1]} markers")
    print(f"  Unique values: {np.unique(X.values)}")
    print(f"  value=0: {(X.values == 0).sum()} ({(X.values == 0).mean():.1%})")
    print(f"  value=1: {(X.values == 1).sum()} ({(X.values == 1).mean():.1%})")

    np.savetxt(
        os.path.join(out_dir, "genotype.txt"),
        X.values,
        fmt="%d",
        delimiter="\t",
    )
    print(f"  Saved → data/genotype.txt")

    # ==================== 2. 表型 ====================
    print("\n[2/2] Converting phenotypes...")
    for env_id in range(1, 5):
        y_path = os.path.join(raw_dir, f"wheat{env_id}.Y")
        with open(y_path) as f:
            header = f.readline().strip()  # e.g. "env1"
        y = np.loadtxt(y_path, skiprows=1)

        out_path = os.path.join(out_dir, f"phenotype_{header}.txt")
        np.savetxt(out_path, y, fmt="%.7f")
        print(f"  {header}: mean={y.mean():.4f}, sd={y.std():.4f}, "
              f"min={y.min():.4f}, max={y.max():.4f}")
        print(f"    Saved → data/phenotype_{header}.txt")

    # ==================== 3. 验证 ====================
    print("\n[3/3] Verifying output files...")
    X2 = np.loadtxt(os.path.join(out_dir, "genotype.txt"), dtype=int)
    assert X2.shape == (599, 1279), f"Genotype shape mismatch: {X2.shape}"
    for env_id in range(1, 5):
        y2 = np.loadtxt(os.path.join(out_dir, f"phenotype_env{env_id}.txt"))
        assert len(y2) == 599, f"env{env_id} length mismatch: {len(y2)}"
    print("  All checks passed.")

    print("\nDone. You can now run genomic_transformer.py.")

if __name__ == "__main__":
    main()
