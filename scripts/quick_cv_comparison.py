#!/usr/bin/env python3
"""
快速交叉验证对比实验

用 5-fold stratified CV 可靠评估：
1. 原始展平特征（当前 baseline）
2. 原始 + 前缀聚合（hybrid）
3. 原始 + 物理特征
4. 原始 + 前缀聚合 + 物理特征
5. 只用物理特征
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from emc_diag.data_pipeline import load_local_data
from emc_diag.feature_engineering import (
    _augment_tabular_with_prefix_aggregates,
    _extract_cognitive_radio_physics_features,
)

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "Cognitive Radio Spectrum Sensing Dataset.csv"

# Columns to drop for PU_Presence task
DROP_LEAKAGE_COLS = ["time_index", "PU_bandwidth", "PU_burst_duration", "PU_drift_type"]


def main() -> None:
    print("加载数据...")
    result = load_local_data(
        CSV_PATH,
        schema="tabular",
        target_column="PU_Presence",
        drop_columns=DROP_LEAKAGE_COLS,
    )
    X_raw = result["X"]
    y = result["y"]
    feature_names = result["feature_names"]
    print(f"样本数: {len(y)}, 原始特征数: {X_raw.shape[1]}")
    print(f"类别分布: {dict(zip(*np.unique(y, return_counts=True)))}")

    # 先标准化（CV 内部还会再标准化，这里只是为了给特征工程提供标准化输入）
    scaler_raw = StandardScaler()
    X_std = scaler_raw.fit_transform(X_raw)

    # --- 特征方案 ---
    configs = {}

    # 1. 原始展平特征
    configs["raw_flat"] = X_std

    # 2. 原始 + 前缀聚合 (hybrid)
    X_aug, _, _, aug_names = _augment_tabular_with_prefix_aggregates(
        X_std, X_std, X_std, feature_names
    )
    configs["hybrid_aug"] = X_aug

    # 3. 只用物理特征
    phys_feats, phys_names = _extract_cognitive_radio_physics_features(X_std, feature_names)
    if phys_feats.shape[1] > 0:
        configs["physics_only"] = phys_feats
        print(f"物理特征数: {phys_feats.shape[1]}")

    # 4. 原始 + 物理特征
    configs["raw_plus_physics"] = np.hstack([X_std, phys_feats])

    # 5. hybrid + 物理
    configs["hybrid_plus_physics"] = np.hstack([X_aug, phys_feats])

    # --- 模型 ---
    models = {
        "LogisticRegression": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, C=1.0))]),
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n" + "=" * 80)
    print("5-Fold Stratified CV Results (accuracy)")
    print("=" * 80)
    print(f"{'Feature Config':<30} {'Model':<25} {'Mean':>8} {'Std':>8}")
    print("-" * 80)

    best_score = 0.0
    best_config = ""

    for config_name, X in configs.items():
        for model_name, model in models.items():
            scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
            mean_score = scores.mean()
            std_score = scores.std()
            marker = " ***" if mean_score > best_score else ""
            if mean_score > best_score:
                best_score = mean_score
                best_config = f"{config_name} + {model_name}"
            print(f"{config_name:<30} {model_name:<25} {mean_score:>8.4f} {std_score:>8.4f}{marker}")

    print("-" * 80)
    print(f"\n最佳组合: {best_config} (accuracy={best_score:.4f})")
    print(f"随机基线: {max(np.bincount(y)) / len(y):.4f}")


if __name__ == "__main__":
    main()
