#!/usr/bin/env python3
"""
PU_Presence 数据层根因诊断脚本

目标：
1. 分析 PU_Signal_Strength 在 PU_Presence=0/1 两类中的分布
2. 分析 PU_drift_type 在两类中的交叉频率表
3. 从 SU*_cov_flat 重建协方差矩阵，计算特征值统计量
4. 输出所有标量特征与 PU_Presence 的相关性排序
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------- 配置 ----------
CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "Cognitive Radio Spectrum Sensing Dataset.csv"
FLOAT_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")


def parse_array_string(text: str) -> np.ndarray | None:
    """与 data_pipeline._parse_array_like_value 保持一致的解析逻辑。"""
    if not isinstance(text, str):
        return None
    text = text.strip()
    if text.startswith("array(") and text.endswith(")"):
        text = text[6:-1].strip()
    if not (text.startswith("[") and text.endswith("]")):
        return None
    inner = text[1:-1].replace("\n", " ").replace(",", " ").strip()
    if not inner:
        return np.array([], dtype=float)
    try:
        parsed = np.fromstring(inner, sep=" ", dtype=float)
    except ValueError:
        parsed = np.array([], dtype=float)
    if parsed.size > 0:
        return parsed
    matches = FLOAT_RE.findall(inner)
    if not matches:
        return None
    return np.array([float(m) for m in matches], dtype=float)


def rebuild_cov_matrix(flat: np.ndarray) -> np.ndarray | None:
    """尝试将展平数组重建为方阵（协方差矩阵）。"""
    n = flat.size
    side = int(np.round(np.sqrt(n)))
    if side * side == n:
        return flat.reshape(side, side)
    return None


def cov_features(mat: np.ndarray) -> dict[str, float]:
    """从协方差矩阵提取物理检测统计量。"""
    eigvals = np.linalg.eigvalsh(mat)
    eigvals_sorted = np.sort(eigvals)[::-1]
    lmax = eigvals_sorted[0]
    lmin = eigvals_sorted[-1]
    tr = np.trace(mat)
    total_abs = np.abs(mat).sum()
    diag_abs = np.abs(np.diag(mat)).sum()
    return {
        "trace_ed": tr,
        "max_eig": lmax,
        "min_eig": lmin,
        "mme_ratio": lmax / max(abs(lmin), 1e-12),
        "rle_ratio": lmax / max(abs(tr), 1e-12),
        "cav_ratio": diag_abs / max(total_abs, 1e-12),
        "cond_number": lmax / max(abs(lmin), 1e-12),
        "log_det": np.log(max(abs(np.linalg.det(mat)), 1e-30)),
        "eig_spread": lmax - lmin,
    }


def main() -> None:
    print(f"加载数据: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"数据集大小: {df.shape[0]} 行 x {df.shape[1]} 列")
    print(f"列名: {list(df.columns)}\n")

    y = df["PU_Presence"].values
    print("=" * 60)
    print("1. PU_Presence 类别分布")
    print("=" * 60)
    for cls in sorted(np.unique(y)):
        print(f"  class {cls}: {(y == cls).sum()} 样本")

    # --- PU_Signal_Strength 分布 ---
    print("\n" + "=" * 60)
    print("2. PU_Signal_Strength 在两类中的分布")
    print("=" * 60)
    if "PU_Signal_Strength" in df.columns:
        for cls in [0, 1]:
            vals = df.loc[y == cls, "PU_Signal_Strength"]
            print(f"  class {cls}: mean={vals.mean():.4f}, std={vals.std():.4f}, "
                  f"min={vals.min():.4f}, max={vals.max():.4f}, median={vals.median():.4f}")
        # 判断是否为泄漏
        absent_vals = df.loc[y == 0, "PU_Signal_Strength"]
        if (absent_vals == 0).all():
            print("  >>> PU 不存在时 strength 全为 0 -> 这是完美泄漏特征!")
        elif absent_vals.std() < 1e-6:
            print("  >>> PU 不存在时 strength 几乎常数 -> 高度泄漏!")
        else:
            print("  >>> PU 不存在时 strength 有变化 -> 可能是合法特征")

    # --- PU_drift_type 交叉表 ---
    print("\n" + "=" * 60)
    print("3. PU_drift_type 在两类中的交叉频率表")
    print("=" * 60)
    if "PU_drift_type" in df.columns:
        ct = pd.crosstab(df["PU_drift_type"], df["PU_Presence"])
        print(ct.to_string())
        # 检查是否某类独占某值
        for drift_val in ct.index:
            row = ct.loc[drift_val]
            if row[0] == 0 or row[1] == 0:
                print(f"  >>> '{drift_val}' 只出现在某一类 -> 泄漏!")

    # --- Frequency_Band 交叉表 ---
    print("\n" + "=" * 60)
    print("4. Frequency_Band 在两类中的分布")
    print("=" * 60)
    if "Frequency_Band" in df.columns:
        ct = pd.crosstab(df["Frequency_Band"], df["PU_Presence"])
        print(ct.to_string())

    # --- 标量特征相关性 ---
    print("\n" + "=" * 60)
    print("5. 标量特征与 PU_Presence 的 point-biserial 相关系数")
    print("=" * 60)
    scalar_cols = [c for c in df.columns
                   if c != "PU_Presence" and pd.api.types.is_numeric_dtype(df[c])]
    corrs = []
    for col in scalar_cols:
        vals = df[col].values.astype(float)
        mask = ~np.isnan(vals)
        if mask.sum() < 10:
            continue
        r = np.corrcoef(vals[mask], y[mask])[0, 1]
        corrs.append((col, r, abs(r)))
    corrs.sort(key=lambda x: x[2], reverse=True)
    for col, r, ar in corrs:
        print(f"  {col:30s}  r={r:+.4f}  |r|={ar:.4f}")

    # --- SU 协方差矩阵重建与特征值分析 ---
    print("\n" + "=" * 60)
    print("6. SU 协方差矩阵特征值分析")
    print("=" * 60)
    su_cov_cols = [c for c in df.columns if "cov_flat" in c.lower()]
    for col in su_cov_cols:
        print(f"\n  --- {col} ---")
        # 解析并重建矩阵
        feats_by_class: dict[int, list[dict]] = {0: [], 1: []}
        parse_ok = 0
        rebuild_ok = 0
        for idx, val in enumerate(df[col]):
            arr = parse_array_string(str(val))
            if arr is None or arr.size == 0:
                continue
            parse_ok += 1
            mat = rebuild_cov_matrix(arr)
            if mat is None:
                continue
            rebuild_ok += 1
            feats = cov_features(mat)
            feats_by_class[int(y[idx])].append(feats)

        print(f"  解析成功: {parse_ok}, 重建为方阵: {rebuild_ok}")
        if rebuild_ok == 0:
            # 如果不是方阵，尝试直接从展平数组提取统计量
            print(f"  展平数组长度不是完全平方数，尝试直接统计量...")
            for cls in [0, 1]:
                flat_vals = []
                for idx in range(len(df)):
                    if y[idx] != cls:
                        continue
                    arr = parse_array_string(str(df[col].iloc[idx]))
                    if arr is not None and arr.size > 0:
                        flat_vals.append(arr)
                if flat_vals:
                    all_arr = np.vstack(flat_vals) if len(set(a.size for a in flat_vals)) == 1 else None
                    if all_arr is not None:
                        print(f"  class {cls}: shape={all_arr.shape}, "
                              f"mean={all_arr.mean():.6f}, std={all_arr.std():.6f}")
            continue

        # 比较两类的检测统计量
        for metric in ["trace_ed", "mme_ratio", "rle_ratio", "cav_ratio", "log_det"]:
            vals0 = [f[metric] for f in feats_by_class[0]]
            vals1 = [f[metric] for f in feats_by_class[1]]
            if vals0 and vals1:
                m0, s0 = np.mean(vals0), np.std(vals0)
                m1, s1 = np.mean(vals1), np.std(vals1)
                # Cohen's d 效应量
                pooled_std = np.sqrt((s0**2 + s1**2) / 2) if (s0 + s1) > 0 else 1e-12
                d = abs(m1 - m0) / pooled_std
                print(f"  {metric:15s}: class0={m0:+.4f}(±{s0:.4f}), "
                      f"class1={m1:+.4f}(±{s1:.4f}), Cohen_d={d:.3f}")

    # --- temporal_cov 分析 ---
    print("\n" + "=" * 60)
    print("7. SU temporal_cov 数组结构分析")
    print("=" * 60)
    temporal_cols = [c for c in df.columns if "temporal_cov" in c.lower()]
    for col in temporal_cols:
        print(f"\n  --- {col} ---")
        lengths = []
        for val in df[col]:
            arr = parse_array_string(str(val))
            if arr is not None:
                lengths.append(arr.size)
        if lengths:
            print(f"  解析后数组长度: min={min(lengths)}, max={max(lengths)}, "
                  f"mean={np.mean(lengths):.1f}, unique={len(set(lengths))}")
            # 检查是否能重建为矩阵
            common_len = max(set(lengths), key=lengths.count)
            side = int(np.round(np.sqrt(common_len)))
            print(f"  最常见长度: {common_len}, sqrt={side}, "
                  f"是完全平方: {side*side == common_len}")

    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
