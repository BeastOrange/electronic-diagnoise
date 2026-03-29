# Thesis Asset Pipeline

这份文档固定了“能毕业”的双主线实验交付路径，目标不是继续盲目堆模型，而是稳定产出论文级实验资产。

## 1. Cognitive 主线 benchmark

```bash
uv run python -m emc_diag benchmark \
  --configs configs/cognitive_radio_thesis_benchmark.yaml \
  --device auto \
  --theme paper-bar
```

输出重点：

- `artifacts/benchmarks/benchmark-*/benchmark_metrics.csv`
- `artifacts/benchmarks/benchmark-*/summary.md`
- `artifacts/benchmarks/benchmark-*/figures/*.png`

## 2. VSB 第二主线 benchmark

```bash
uv run python -m emc_diag benchmark \
  --configs configs/vsb_fault_thesis_benchmark.yaml \
  --device auto \
  --theme paper-bar
```

输出重点：

- `artifacts/benchmarks/benchmark-*/benchmark_metrics.csv`
- `artifacts/benchmarks/benchmark-*/summary.md`
- `artifacts/benchmarks/benchmark-*/figures/*.png`

## 3. 单实验的探索记录

任意配置在 `prepare` 和 `extract-features` 后都会在对应 `prepared_dir` 下产出探索资产：

- `figures/dataset_summary.png`
- `figures/feature_importance.png`
- `tables/class_distribution.csv`
- `tables/dataset_statistics.csv`
- `tables/selected_features.csv`
- `exploration_summary.md`

推荐命令：

```bash
uv run python -m emc_diag prepare --config configs/cognitive_radio_spectrum.yaml
uv run python -m emc_diag extract-features --config configs/cognitive_radio_spectrum.yaml
uv run python -m emc_diag train --config configs/cognitive_radio_spectrum.yaml --device auto
```

## 4. 论文级总资产汇总

在收集到主线 `run_dir` 和 `benchmark_dir` 后，执行：

```bash
uv run python -m emc_diag thesis-assets \
  --run-dirs \
    artifacts/runs/<run-a> \
    artifacts/runs/<run-b> \
  --benchmark-dirs \
    artifacts/benchmarks/<benchmark-a> \
    artifacts/benchmarks/<benchmark-b> \
  --output-dir artifacts/reports/thesis-assets-final \
  --title "Final Summary"
```

汇总命令会生成：

- `final_metrics.csv`
- `benchmark_metrics_merged.csv`
- `thesis_figures_manifest.csv`
- `final_summary.md`
- `figures/dataset_comparison.png`
- `figures/task_comparison.png`
- `figures/ml_vs_cnn_comparison.png`

## 5. 建议的最终交付顺序

1. 先跑 Cognitive 主线 benchmark，确认 Presence 与辅助任务的主表结果。
2. 再跑 VSB benchmark，补第二主线与泛化验证。
3. 对关键单 run 执行 `export-report`，确保训练历史、混淆矩阵、binary curves 等都完整。
4. 最后用 `thesis-assets` 汇总成论文级总资产目录。
