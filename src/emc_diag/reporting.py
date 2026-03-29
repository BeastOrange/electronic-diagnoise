from __future__ import annotations

from pathlib import Path


def format_metric_value(value: float | int | str) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def collect_run_artifacts(run_dir: Path) -> dict[str, list[str]]:
    figures_dir = run_dir / "figures"
    tables_dir = run_dir / "tables"
    figures = sorted(path.name for path in figures_dir.glob("*.png")) if figures_dir.exists() else []
    tables = sorted(path.name for path in tables_dir.glob("*.csv")) if tables_dir.exists() else []
    return {"figures": figures, "tables": tables}


def summarize_cv_aggregates(cv_records: list[dict[str, float | int | str]]) -> dict[str, float]:
    if not cv_records:
        return {"best_cv_mean": 0.0, "mean_cv_std": 0.0}

    means = [float(record["cv_mean"]) for record in cv_records if "cv_mean" in record]
    stds = [float(record["cv_std"]) for record in cv_records if "cv_std" in record]
    if not means:
        return {"best_cv_mean": 0.0, "mean_cv_std": 0.0}
    return {
        "best_cv_mean": max(means),
        "mean_cv_std": (sum(stds) / len(stds)) if stds else 0.0,
    }


def build_overfitting_risk_note(
    max_gap: float,
    warning_threshold: float = 0.05,
) -> str:
    if max_gap >= warning_threshold:
        return (
            f"High overfitting risk detected: max gap={max_gap:.4f} "
            f"(threshold={warning_threshold:.4f})."
        )
    return (
        f"Overfitting risk acceptable: max gap={max_gap:.4f} "
        f"(threshold={warning_threshold:.4f})."
    )


def build_ablation_notes(ablation_records: list[dict[str, float | int | str]]) -> list[str]:
    if not ablation_records:
        return []
    baseline = float(ablation_records[0]["score"]) if "score" in ablation_records[0] else 0.0
    notes: list[str] = []
    for record in ablation_records:
        variant = str(record.get("variant", "unknown"))
        score = float(record.get("score", 0.0))
        notes.append(f"{variant}: {score:.4f} (delta {score - baseline:+.4f} vs baseline)")
    return notes


def summarize_cross_dataset_benchmark(
    benchmark_records: list[dict[str, float | int | str]],
) -> dict[str, float | int | str]:
    if not benchmark_records:
        return {
            "dataset_count": 0,
            "best_dataset": "N/A",
            "best_accuracy": 0.0,
            "mean_accuracy": 0.0,
            "worst_accuracy": 0.0,
        }

    valid_records = [
        record
        for record in benchmark_records
        if "dataset" in record and "accuracy" in record
    ]
    if not valid_records:
        return {
            "dataset_count": 0,
            "best_dataset": "N/A",
            "best_accuracy": 0.0,
            "mean_accuracy": 0.0,
            "worst_accuracy": 0.0,
        }

    accuracies = [float(record["accuracy"]) for record in valid_records]
    best_index = int(max(range(len(valid_records)), key=lambda idx: float(valid_records[idx]["accuracy"])))
    best_dataset = str(valid_records[best_index]["dataset"])
    return {
        "dataset_count": len(valid_records),
        "best_dataset": best_dataset,
        "best_accuracy": max(accuracies),
        "mean_accuracy": sum(accuracies) / len(accuracies),
        "worst_accuracy": min(accuracies),
    }


def build_cross_dataset_benchmark_notes(
    benchmark_records: list[dict[str, float | int | str]],
) -> list[str]:
    notes: list[str] = []
    for record in benchmark_records:
        dataset = str(record.get("dataset", "unknown"))
        accuracy = float(record.get("accuracy", 0.0))
        f1 = float(record.get("f1", 0.0))
        notes.append(f"{dataset}: accuracy={accuracy:.4f}, f1={f1:.4f}")
    return notes


def _format_dict_entries(values: dict[str, str]) -> list[str]:
    return [f"{key}: {values[key]}" for key in sorted(values)]


def summarize_benchmark_highlights(
    benchmark_records: list[dict[str, float | int | str]],
    top_n: int = 2,
) -> list[str]:
    if not benchmark_records:
        return []

    def _score(record: dict[str, float | int | str], key: str) -> float:
        return float(record.get(key, 0.0))

    sorted_by_accuracy = sorted(benchmark_records, key=lambda record: _score(record, "accuracy"), reverse=True)
    sorted_by_macro = sorted(benchmark_records, key=lambda record: _score(record, "macro_f1"), reverse=True)
    sorted_by_minority = sorted(benchmark_records, key=lambda record: _score(record, "minority_f1"), reverse=True)

    unique_tasks = {record.get("task_name") for record in benchmark_records if record.get("task_name")}
    records: list[str] = []
    if sorted_by_minority and float(sorted_by_minority[0].get("minority_f1", 0.0)) > 0.0:
        top = sorted_by_minority[0]
        records.append(
            f"Top minority_f1: {top.get('model_name', top.get('model', 'model'))} on {top.get('task_name', 'task')} "
            f"({format_metric_value(float(top.get('minority_f1', 0.0)))})."
        )
    if sorted_by_accuracy:
        top = sorted_by_accuracy[0]
        records.append(
            f"Top accuracy: {top.get('model_name', top.get('model', 'model'))} on {top.get('task_name', 'task')} "
            f"({format_metric_value(float(top.get('accuracy', 0.0)))})."
        )
    if sorted_by_macro:
        top = sorted_by_macro[0]
        records.append(
            f"Top macro_f1: {top.get('model_name', top.get('model', 'model'))} on {top.get('task_name', 'task')} "
            f"({format_metric_value(float(top.get('macro_f1', top.get('f1', 0.0))))})."
        )
    records.append(f"Tasks represented: {len(unique_tasks)}")
    if len(benchmark_records) > top_n:
        records.append(f"Total benchmark runs: {len(benchmark_records)}")
    return records


def build_benchmark_markdown_table(
    benchmark_records: list[dict[str, float | int | str]],
    max_rows: int = 5,
) -> list[str]:
    if not benchmark_records:
        return []

    columns = ["rank", "model_name", "task_name", "model_family", "accuracy", "macro_f1", "run_dir"]
    sorted_records = sorted(benchmark_records, key=lambda record: float(record.get("accuracy", 0.0)), reverse=True)
    table_lines: list[str] = ["## Benchmark Table"]
    header = "| " + " | ".join(columns) + " |"
    table_lines.append(header)
    delimiter = "| " + " | ".join("---" for _ in columns) + " |"
    table_lines.append(delimiter)
    for rank, record in enumerate(sorted_records[:max_rows], start=1):
        row: list[str] = [str(rank)]
        for column in columns[1:]:
            value = record.get(column, "")
            if isinstance(value, (int, float)) and column in {"accuracy", "macro_f1"}:
                value = format_metric_value(float(value))
            row.append(str(value))
        table_lines.append("| " + " | ".join(row) + " |")
    return table_lines


def _format_dict_entries(values: dict[str, str]) -> list[str]:
    return [f"{key}: {values[key]}" for key in sorted(values)]


def summarize_benchmark_highlights(
    benchmark_records: list[dict[str, float | int | str]],
    top_n: int = 2,
) -> list[str]:
    if not benchmark_records:
        return []

    def _score(record: dict[str, float | int | str], key: str) -> float:
        return float(record.get(key, 0.0))

    sorted_by_accuracy = sorted(benchmark_records, key=lambda record: _score(record, "accuracy"), reverse=True)
    sorted_by_macro = sorted(benchmark_records, key=lambda record: _score(record, "macro_f1"), reverse=True)

    unique_tasks = {record.get("task_name") for record in benchmark_records if record.get("task_name")}
    records: list[str] = []
    if sorted_by_accuracy:
        top = sorted_by_accuracy[0]
        records.append(
            f"Top accuracy: {top.get('model_name', top.get('model', 'model'))} on {top.get('task_name', 'task')} "
            f"({format_metric_value(float(top.get('accuracy', 0.0)))})."
        )
    if sorted_by_macro:
        top = sorted_by_macro[0]
        records.append(
            f"Top macro_f1: {top.get('model_name', top.get('model', 'model'))} on {top.get('task_name', 'task')} "
            f"({format_metric_value(float(top.get('macro_f1', top.get('f1', 0.0))))})."
        )
    records.append(f"Tasks represented: {len(unique_tasks)}")
    if len(benchmark_records) > top_n:
        records.append(f"Total benchmark runs: {len(benchmark_records)}")
    return records


def build_benchmark_markdown_table(
    benchmark_records: list[dict[str, float | int | str]],
    max_rows: int = 5,
) -> list[str]:
    if not benchmark_records:
        return []

    columns = ["rank", "model_name", "task_name", "model_family", "accuracy", "macro_f1", "run_dir"]
    sorted_records = sorted(benchmark_records, key=lambda record: float(record.get("accuracy", 0.0)), reverse=True)
    table_lines: list[str] = ["## Benchmark Table"]
    header = "| " + " | ".join(columns) + " |"
    table_lines.append(header)
    delimiter = "| " + " | ".join("---" for _ in columns) + " |"
    table_lines.append(delimiter)
    for rank, record in enumerate(sorted_records[:max_rows], start=1):
        row: list[str] = [str(rank)]
        for column in columns[1:]:
            value = record.get(column, "")
            if isinstance(value, (int, float)) and column in {"accuracy", "macro_f1"}:
                value = format_metric_value(float(value))
            row.append(str(value))
        table_lines.append("| " + " | ".join(row) + " |")
    return table_lines


def build_run_summary(
    run_name: str,
    output_dir: Path,
    metrics: dict[str, float | int | str],
    figures: list[str],
    tables: list[str],
    task_metadata: dict[str, str] | None = None,
    run_metadata: dict[str, str] | None = None,
    transfer_notes: list[str] | None = None,
    pretrain_notes: list[str] | None = None,
    feature_group_summary: str | None = None,
    robustness_notes: list[str] | None = None,
    cv_aggregates: dict[str, float] | None = None,
    ablation_notes: list[str] | None = None,
    overfitting_risk_note: str | None = None,
    cross_dataset_benchmark_summary: dict[str, float | int | str] | None = None,
    cross_dataset_benchmark_notes: list[str] | None = None,
    benchmark_highlights: list[str] | None = None,
    benchmark_table: list[str] | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.md"

    lines: list[str] = [f"# Run Summary: {run_name}", ""]
    if task_metadata:
        lines.append("## Task")
        for key in sorted(task_metadata):
            lines.append(f"- {key}: {task_metadata[key]}")
        lines.append("")
    if run_metadata:
        lines.append("## Run Metadata")
        lines.extend(_format_dict_entries(run_metadata))
        lines.append("")
    if transfer_notes or pretrain_notes:
        lines.append("## Transfer / Pretrain")
        if transfer_notes:
            lines.extend(transfer_notes)
        if pretrain_notes:
            lines.extend(pretrain_notes)
        lines.append("")
    if feature_group_summary:
        lines.append("## Feature Groups")
        lines.append(f"- {feature_group_summary}")
        lines.append("")
    if robustness_notes:
        lines.append("## Robustness Notes")
        for note in robustness_notes:
            lines.append(f"- {note}")
        lines.append("")

    lines.append("## Metrics")
    for key in sorted(metrics):
        lines.append(f"- {key}: {format_metric_value(metrics[key])}")
    lines.append("")

    lines.append("## Figures")
    for figure in figures:
        lines.append(f"- {figure}")
    lines.append("")

    lines.append("## Tables")
    for table in tables:
        lines.append(f"- {table}")
    lines.append("")

    if cv_aggregates:
        lines.append("## CV Aggregates")
        for key in sorted(cv_aggregates):
            lines.append(f"- {key}: {format_metric_value(cv_aggregates[key])}")
        lines.append("")

    if ablation_notes:
        lines.append("## Ablation Notes")
        for note in ablation_notes:
            lines.append(f"- {note}")
        lines.append("")

    if overfitting_risk_note:
        lines.append("## Overfitting/Risk Note")
        lines.append(f"- {overfitting_risk_note}")
        lines.append("")

    if cross_dataset_benchmark_summary:
        lines.append("## Cross-Dataset Benchmark Summary")
        for key in sorted(cross_dataset_benchmark_summary):
            lines.append(f"- {key}: {format_metric_value(cross_dataset_benchmark_summary[key])}")
        lines.append("")

    if cross_dataset_benchmark_notes:
        lines.append("## Cross-Dataset Benchmark Notes")
        for note in cross_dataset_benchmark_notes:
            lines.append(f"- {note}")
        lines.append("")

    if benchmark_highlights:
        lines.append("## Benchmark Highlights")
        for highlight in benchmark_highlights:
            lines.append(f"- {highlight}")
        lines.append("")

    if benchmark_table:
        lines.extend(benchmark_table)
        lines.append("")

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def build_prepared_summary(
    prepared_dir: Path,
    dataset_metadata: dict[str, str],
    statistics: dict[str, float | int | str | dict[str, int] | list[float]],
    figures: list[str],
    tables: list[str],
    feature_summary: dict[str, str] | None = None,
) -> Path:
    prepared_dir.mkdir(parents=True, exist_ok=True)
    summary_path = prepared_dir / "exploration_summary.md"

    lines: list[str] = ["# Dataset Exploration Summary", ""]
    lines.append("## Dataset")
    for key in sorted(dataset_metadata):
        lines.append(f"- {key}: {dataset_metadata[key]}")
    lines.append("")

    lines.append("## Statistics")
    class_counts = statistics.get("class_counts", {})
    if isinstance(class_counts, dict):
        for label, count in class_counts.items():
            lines.append(f"- class_count[{label}]: {count}")
    for key in ["missing_values", "missing_ratio", "num_samples", "num_features"]:
        if key in statistics:
            lines.append(f"- {key}: {format_metric_value(statistics[key])}")
    for key in ["feature_mean_head", "feature_std_head"]:
        if key in statistics:
            lines.append(f"- {key}: {statistics[key]}")
    lines.append("")

    if feature_summary:
        lines.append("## Feature Analysis")
        for key in sorted(feature_summary):
            lines.append(f"- {key}: {feature_summary[key]}")
        lines.append("")

    lines.append("## Figures")
    for figure in figures:
        lines.append(f"- {figure}")
    lines.append("")

    lines.append("## Tables")
    for table in tables:
        lines.append(f"- {table}")
    lines.append("")

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def build_thesis_asset_summary(
    output_dir: Path,
    title: str,
    overview_metrics: dict[str, float | int | str],
    key_findings: list[str],
    dataset_notes: list[str],
    output_files: list[str],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "final_summary.md"

    lines: list[str] = [f"# {title}", ""]
    lines.append("## Overview")
    for key in sorted(overview_metrics):
        lines.append(f"- {key}: {format_metric_value(overview_metrics[key])}")
    lines.append("")

    lines.append("## Key Findings")
    for finding in key_findings:
        lines.append(f"- {finding}")
    lines.append("")

    lines.append("## Dataset Notes")
    for note in dataset_notes:
        lines.append(f"- {note}")
    lines.append("")

    lines.append("## Output Files")
    for file_name in output_files:
        lines.append(f"- {file_name}")
    lines.append("")

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path
