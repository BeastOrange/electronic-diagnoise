#!/usr/bin/env python3
"""
PU_Presence 深度诊断 v2

重点分析：
1. cov_flat / temporal_cov 的每个元素与 PU_Presence 的相关性
2. 从数组提取聚合统计量（L2 norm、variance、max-min）的判别力
3. 跨 SU 的关系特征
4. 看到底有没有"信号"藏在数据里
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "Cognitive Radio Spectrum Sensing Dataset.csv"
FLOAT_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")


def parse_array_string(text: str) -> np.ndarray | None:
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


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    y = df["PU_Presence"].values
    n = len(y)

    # ---------- 1. 展开数组列，逐元素计算相关性 ----------
    print("=" * 70)
    print("1. 数组特征逐元素与 PU_Presence 的相关性（按 |r| 排序）")
    print("=" * 70)

    array_cols = [c for c in df.columns if "cov_flat" in c or "temporal_cov" in c]
    all_corrs = []  # (name, r)

    # 同时收集展开后的矩阵用于后续分析
    su_arrays: dict[str, np.ndarray] = {}  # col_name -> (1000, dim)

    for col in array_cols:
        arrays = []
        for val in df[col]:
            arr = parse_array_string(str(val))
            arrays.append(arr if arr is not None else np.array([]))
        dim = max(a.size for a in arrays)
        mat = np.zeros((n, dim), dtype=float)
        for i, a in enumerate(arrays):
            if a.size > 0:
                mat[i, :a.size] = a[:dim]
        su_arrays[col] = mat

        for j in range(dim):
            r = np.corrcoef(mat[:, j], y)[0, 1]
            all_corrs.append((f"{col}_{j}", r))

    all_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"  共 {len(all_corrs)} 个元素特征")
    print("  Top 20:")
    for name, r in all_corrs[:20]:
        print(f"    {name:35s}  r={r:+.4f}")
    print(f"  最大 |r| = {abs(all_corrs[0][1]):.4f}")

    # ---------- 2. 数组聚合统计量 ----------
    print("\n" + "=" * 70)
    print("2. 数组聚合统计量与 PU_Presence 的相关性")
    print("=" * 70)

    agg_corrs = []
    for col, mat in su_arrays.items():
        # 向量统计量
        feats = {
            f"{col}_l2norm": np.linalg.norm(mat, axis=1),
            f"{col}_mean": mat.mean(axis=1),
            f"{col}_std": mat.std(axis=1),
            f"{col}_max": mat.max(axis=1),
            f"{col}_min": mat.min(axis=1),
            f"{col}_range": mat.max(axis=1) - mat.min(axis=1),
            f"{col}_median": np.median(mat, axis=1),
            f"{col}_energy": (mat ** 2).sum(axis=1),
            f"{col}_skew": _safe_skew(mat),
            f"{col}_kurtosis": _safe_kurtosis(mat),
        }
        for fname, fvals in feats.items():
            r = np.corrcoef(fvals, y)[0, 1]
            agg_corrs.append((fname, r))

    agg_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    for name, r in agg_corrs[:30]:
        print(f"  {name:45s}  r={r:+.4f}")

    # ---------- 3. 跨 SU 特征 ----------
    print("\n" + "=" * 70)
    print("3. 跨 SU 关系特征与 PU_Presence 的相关性")
    print("=" * 70)

    cross_corrs = []
    cov_cols = [c for c in array_cols if "cov_flat" in c]
    temp_cols = [c for c in array_cols if "temporal_cov" in c]

    # SU 间余弦相似度
    for i in range(len(cov_cols)):
        for j in range(i + 1, len(cov_cols)):
            a, b = su_arrays[cov_cols[i]], su_arrays[cov_cols[j]]
            cos_sim = _row_cosine(a, b)
            name = f"cos({cov_cols[i]},{cov_cols[j]})"
            r = np.corrcoef(cos_sim, y)[0, 1]
            cross_corrs.append((name, r))

    for i in range(len(temp_cols)):
        for j in range(i + 1, len(temp_cols)):
            a, b = su_arrays[temp_cols[i]], su_arrays[temp_cols[j]]
            cos_sim = _row_cosine(a, b)
            name = f"cos({temp_cols[i]},{temp_cols[j]})"
            r = np.corrcoef(cos_sim, y)[0, 1]
            cross_corrs.append((name, r))

    # SU 间 L2 距离
    for i in range(len(cov_cols)):
        for j in range(i + 1, len(cov_cols)):
            a, b = su_arrays[cov_cols[i]], su_arrays[cov_cols[j]]
            dist = np.linalg.norm(a - b, axis=1)
            name = f"l2dist({cov_cols[i]},{cov_cols[j]})"
            r = np.corrcoef(dist, y)[0, 1]
            cross_corrs.append((name, r))

    # SU 统计量均值/方差（3 个 SU 的一致性）
    for stat_name, stat_fn in [("l2norm", lambda m: np.linalg.norm(m, axis=1)),
                                ("energy", lambda m: (m**2).sum(axis=1)),
                                ("std", lambda m: m.std(axis=1))]:
        cov_vals = np.column_stack([stat_fn(su_arrays[c]) for c in cov_cols])
        temp_vals = np.column_stack([stat_fn(su_arrays[c]) for c in temp_cols])
        for prefix, vals in [("cov", cov_vals), ("temp", temp_vals)]:
            mu = vals.mean(axis=1)
            sd = vals.std(axis=1)
            r_mu = np.corrcoef(mu, y)[0, 1]
            r_sd = np.corrcoef(sd, y)[0, 1]
            cross_corrs.append((f"cross_{prefix}_{stat_name}_mean", r_mu))
            cross_corrs.append((f"cross_{prefix}_{stat_name}_std", r_sd))

    cross_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    for name, r in cross_corrs:
        print(f"  {name:55s}  r={r:+.4f}")

    # ---------- 4. 尝试把 cov_flat (8) 视为 2x4 矩阵 ----------
    print("\n" + "=" * 70)
    print("4. 将 cov_flat(8) 视为不同矩阵结构的特征值分析")
    print("=" * 70)

    for col in cov_cols:
        mat = su_arrays[col]  # (1000, 8)
        # 尝试 2x4
        for shape_name, shape in [("2x4", (2, 4)), ("4x2", (4, 2))]:
            feat_corrs = []
            reshaped = mat.reshape(n, *shape)
            # 奇异值
            for i in range(n):
                pass  # 太慢，用批量方法
            # 直接用 gram matrix 特征值
            if shape == (2, 4):
                gram = np.einsum("nij,nik->njk", reshaped, reshaped)  # (n,4,4)
            else:
                gram = np.einsum("nij,nik->njk", reshaped, reshaped)  # (n,2,2)
            # 取 trace 和 max eigenvalue
            traces = np.trace(gram, axis1=1, axis2=2)
            r = np.corrcoef(traces, y)[0, 1]
            feat_corrs.append((f"{col}_{shape_name}_gram_trace", r))

            if gram.shape[1] <= 4:
                eigs = np.linalg.eigvalsh(gram)
                for ei in range(eigs.shape[1]):
                    r = np.corrcoef(eigs[:, ei], y)[0, 1]
                    feat_corrs.append((f"{col}_{shape_name}_eig_{ei}", r))
                # MME ratio
                mme = eigs[:, -1] / np.maximum(np.abs(eigs[:, 0]), 1e-15)
                r = np.corrcoef(mme, y)[0, 1]
                feat_corrs.append((f"{col}_{shape_name}_mme", r))

            feat_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
            for name, r in feat_corrs[:5]:
                print(f"  {name:50s}  r={r:+.4f}")

    # ---------- 5. temporal_cov (32) 视为 4x8 或 8x4 ----------
    print("\n" + "=" * 70)
    print("5. 将 temporal_cov(32) 视为矩阵结构的特征分析")
    print("=" * 70)

    for col in temp_cols:
        mat = su_arrays[col]  # (1000, 32)
        for shape_name, shape in [("4x8", (4, 8)), ("8x4", (8, 4))]:
            reshaped = mat.reshape(n, *shape)
            feat_corrs = []
            # Frobenius norm
            frob = np.linalg.norm(reshaped.reshape(n, -1), axis=1)
            r = np.corrcoef(frob, y)[0, 1]
            feat_corrs.append((f"{col}_{shape_name}_frob", r))
            # 行方差均值
            row_vars = reshaped.var(axis=2).mean(axis=1)
            r = np.corrcoef(row_vars, y)[0, 1]
            feat_corrs.append((f"{col}_{shape_name}_row_var_mean", r))
            # Gram matrix
            if shape[0] <= shape[1]:
                gram = np.einsum("nij,nik->njk", reshaped, reshaped)
            else:
                gram = np.einsum("nji,nki->njk", reshaped, reshaped)
            traces = np.trace(gram, axis1=1, axis2=2)
            r = np.corrcoef(traces, y)[0, 1]
            feat_corrs.append((f"{col}_{shape_name}_gram_trace", r))

            g_small = gram if gram.shape[1] <= 8 else gram[:, :4, :4]
            eigs = np.linalg.eigvalsh(g_small)
            mme = eigs[:, -1] / np.maximum(np.abs(eigs[:, 0]), 1e-15)
            r = np.corrcoef(mme, y)[0, 1]
            feat_corrs.append((f"{col}_{shape_name}_mme", r))

            feat_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
            for name, r in feat_corrs:
                print(f"  {name:50s}  r={r:+.4f}")

    # ---------- 6. 全局最终总结 ----------
    print("\n" + "=" * 70)
    print("6. 全部特征相关性排名 Top 30（含所有衍生特征）")
    print("=" * 70)

    # 收集标量特征
    scalar_cols = [c for c in df.columns if c != "PU_Presence" and pd.api.types.is_numeric_dtype(df[c])]
    final_corrs = []
    for col in scalar_cols:
        v = df[col].values.astype(float)
        r = np.corrcoef(v, y)[0, 1]
        final_corrs.append((col, r))
    final_corrs.extend(all_corrs)
    final_corrs.extend(agg_corrs)
    final_corrs.extend(cross_corrs)
    final_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    for name, r in final_corrs[:30]:
        print(f"  {name:55s}  r={r:+.4f}")

    max_r = abs(final_corrs[0][1])
    print(f"\n  >>> 全局最大 |r| = {max_r:.4f}")
    if max_r < 0.10:
        print("  >>> 结论: 数据中几乎没有与 PU_Presence 相关的线性信号！")
        print("  >>> 需要考虑非线性交互特征或重新审视数据生成过程。")


def _safe_skew(mat: np.ndarray) -> np.ndarray:
    mu = mat.mean(axis=1, keepdims=True)
    s = mat.std(axis=1, keepdims=True)
    s = np.where(s < 1e-15, 1.0, s)
    return ((mat - mu) ** 3).mean(axis=1) / (s.ravel() ** 3)


def _safe_kurtosis(mat: np.ndarray) -> np.ndarray:
    mu = mat.mean(axis=1, keepdims=True)
    s = mat.std(axis=1, keepdims=True)
    s = np.where(s < 1e-15, 1.0, s)
    return ((mat - mu) ** 4).mean(axis=1) / (s.ravel() ** 4) - 3.0


def _row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dot = (a * b).sum(axis=1)
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    return dot / np.maximum(na * nb, 1e-15)


if __name__ == "__main__":
    main()
