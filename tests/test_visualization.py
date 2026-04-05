from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from emc_diag.reporting import (
    build_ablation_notes,
    build_overfitting_risk_note,
    build_run_summary,
    collect_run_artifacts,
    format_metric_value,
    build_cross_dataset_benchmark_notes,
    summarize_cv_aggregates,
    summarize_cross_dataset_benchmark,
)
from emc_diag.visualization import (
    expected_thesis_figure_stems,
    has_thesis_figure_stems,
    paper_bar_theme,
    plot_ablation_comparison,
    plot_candidate_comparison,
    plot_cv_comparison,
    plot_ml_vs_cnn_comparison,
    plot_noise_robustness_curve,
    plot_feature_group_ablation,
    plot_overfitting_gap,
    plot_per_class_metrics,
    plot_spectrum_stft_summary,
    plot_task_comparison,
    plot_transfer_vs_scratch,
    plot_multitask_vs_single_task,
    plot_waveform_overview,
    plot_confusion_matrix,
    plot_dataset_summary,
    plot_feature_importance,
    plot_metrics_bar,
    plot_training_curves,
)


def test_paper_bar_theme_has_required_style_keys() -> None:
    theme = paper_bar_theme()
    assert "palette" in theme
    assert "font_family" in theme
    assert "dpi" in theme
    assert len(theme["palette"]) >= 5


def test_stage_plots_export_png_svg_and_csv(tmp_path: Path) -> None:
    # dataset summary
    dataset_df = pd.DataFrame(
        {"class": ["A", "A", "B", "C"], "value": [1.0, 1.1, 0.3, -0.2]}
    )
    summary_paths = plot_dataset_summary(dataset_df, tmp_path, theme_name="paper-bar")
    assert summary_paths["png"].exists()
    assert summary_paths["svg"].exists()
    assert summary_paths["csv"].exists()

    # feature importance
    feature_df = pd.DataFrame({"feature": ["f1", "f2"], "importance": [0.6, 0.4]})
    fi_paths = plot_feature_importance(feature_df, tmp_path, theme_name="paper-bar")
    assert fi_paths["png"].exists()
    assert fi_paths["svg"].exists()
    assert fi_paths["csv"].exists()

    # training curves
    curve_df = pd.DataFrame(
        {
            "epoch": [1, 2, 3],
            "train_loss": [0.8, 0.5, 0.3],
            "val_loss": [0.9, 0.6, 0.35],
            "accuracy": [0.5, 0.7, 0.85],
            "f1": [0.45, 0.68, 0.82],
        }
    )
    tc_paths = plot_training_curves(curve_df, tmp_path, theme_name="paper-bar")
    assert tc_paths["png"].exists()
    assert tc_paths["svg"].exists()
    assert tc_paths["csv"].exists()

    # confusion matrix
    cm = np.array([[8, 1], [2, 9]])
    cm_paths = plot_confusion_matrix(
        cm, labels=["normal", "fault"], output_dir=tmp_path, theme_name="paper-bar"
    )
    assert cm_paths["png"].exists()
    assert cm_paths["svg"].exists()
    assert cm_paths["csv"].exists()

    # metrics bar
    metric_df = pd.DataFrame(
        {"model": ["rf", "cnn"], "accuracy": [0.72, 0.81], "f1": [0.7, 0.8]}
    )
    mb_paths = plot_metrics_bar(metric_df, tmp_path, theme_name="paper-bar")
    assert mb_paths["png"].exists()
    assert mb_paths["svg"].exists()
    assert mb_paths["csv"].exists()

    candidate_df = pd.DataFrame(
        {"model": ["rf", "svc"], "accuracy": [0.72, 0.84], "f1": [0.71, 0.83]}
    )
    cc_paths = plot_candidate_comparison(candidate_df, tmp_path, theme_name="paper-bar")
    assert cc_paths["png"].exists()
    assert cc_paths["svg"].exists()
    assert cc_paths["csv"].exists()

    per_class_df = pd.DataFrame(
        {
            "label_name": ["A", "B"],
            "precision": [0.8, 0.9],
            "recall": [0.75, 0.88],
            "f1": [0.77, 0.89],
            "support": [12, 10],
        }
    )
    pcm_paths = plot_per_class_metrics(per_class_df, tmp_path, theme_name="paper-bar")
    assert pcm_paths["png"].exists()
    assert pcm_paths["svg"].exists()
    assert pcm_paths["csv"].exists()

    task_df = pd.DataFrame(
        {"task_name": ["presence", "band"], "macro_f1": [0.55, 0.98], "accuracy": [0.56, 0.98]}
    )
    tcmp_paths = plot_task_comparison(task_df, tmp_path, theme_name="paper-bar")
    assert tcmp_paths["png"].exists()
    assert tcmp_paths["svg"].exists()
    assert tcmp_paths["csv"].exists()

    risk_curve = pd.DataFrame(
        {
            "epoch": [1, 2, 3],
            "train_score": [0.72, 0.85, 0.93],
            "val_score": [0.7, 0.8, 0.81],
        }
    )
    gap_paths = plot_overfitting_gap(risk_curve, tmp_path, theme_name="paper-bar")
    assert gap_paths["png"].exists()
    assert gap_paths["svg"].exists()
    assert gap_paths["csv"].exists()

    cv_df = pd.DataFrame(
        {
            "model": ["rf", "cnn"],
            "cv_mean": [0.88, 0.91],
            "cv_std": [0.02, 0.03],
        }
    )
    cv_paths = plot_cv_comparison(cv_df, tmp_path, theme_name="paper-bar")
    assert cv_paths["png"].exists()
    assert cv_paths["svg"].exists()
    assert cv_paths["csv"].exists()

    ablation_df = pd.DataFrame(
        {
            "variant": ["baseline", "w/o entropy", "w/o cov"],
            "score": [0.9, 0.84, 0.8],
        }
    )
    ablation_paths = plot_ablation_comparison(ablation_df, tmp_path, theme_name="paper-bar")
    assert ablation_paths["png"].exists()
    assert ablation_paths["svg"].exists()
    assert ablation_paths["csv"].exists()

    ml_vs_cnn_df = pd.DataFrame(
        {
            "family": ["ml", "cnn"],
            "accuracy": [0.88, 0.91],
            "f1": [0.87, 0.9],
        }
    )
    mvc_paths = plot_ml_vs_cnn_comparison(ml_vs_cnn_df, tmp_path, theme_name="paper-bar")
    assert mvc_paths["png"].exists()
    assert mvc_paths["svg"].exists()
    assert mvc_paths["csv"].exists()

    waveform_df = pd.DataFrame(
        {
            "sample_index": np.arange(128),
            "signal": np.sin(np.linspace(0, 8 * np.pi, 128)),
        }
    )
    waveform_paths = plot_waveform_overview(waveform_df, tmp_path, theme_name="paper-bar")
    assert waveform_paths["png"].exists()
    assert waveform_paths["svg"].exists()
    assert waveform_paths["csv"].exists()

    spectrum_paths = plot_spectrum_stft_summary(waveform_df, tmp_path, theme_name="paper-bar")
    assert spectrum_paths["png"].exists()
    assert spectrum_paths["svg"].exists()
    assert spectrum_paths["csv"].exists()


def test_feature_importance_plot_limits_rows_for_readability(tmp_path: Path) -> None:
    feature_df = pd.DataFrame(
        {
            "feature": [f"very_long_feature_name_{index}" for index in range(20)],
            "importance": np.linspace(0.99, 0.1, 20),
        }
    )

    paths = plot_feature_importance(feature_df, tmp_path, theme_name="paper-bar")
    plotted = pd.read_csv(paths["csv"])

    assert len(plotted) <= 12
    assert plotted.iloc[0]["feature"] == "very_long_feature_name_0"
    assert plotted.iloc[-1]["importance"] <= plotted.iloc[0]["importance"]


def test_new_visualization_plots_create_artifacts(tmp_path: Path) -> None:
    # noise robustness
    noise_df = pd.DataFrame({"noise_sigma": [0.0, 0.05, 0.1], "macro_f1": [0.72, 0.68, 0.55]})
    noise_paths = plot_noise_robustness_curve(noise_df, tmp_path)
    assert noise_paths["png"].exists()
    assert noise_paths["svg"].exists()
    assert noise_paths["csv"].exists()

    # feature group ablation
    ablation_df = pd.DataFrame({"variant": ["baseline", "no_entropy"], "score": [0.9, 0.86]})
    fg_paths = plot_feature_group_ablation(ablation_df, tmp_path)
    assert fg_paths["png"].exists()
    assert fg_paths["svg"].exists()
    assert fg_paths["csv"].exists()

    # transfer vs scratch
    transfer_df = pd.DataFrame(
        {
            "model_name": ["cnn", "cnn"],
            "strategy": ["scratch", "transfer"],
            "macro_f1": [0.71, 0.81],
        }
    )
    transfer_paths = plot_transfer_vs_scratch(transfer_df, tmp_path)
    assert transfer_paths["png"].exists()
    assert transfer_paths["svg"].exists()
    assert transfer_paths["csv"].exists()

    transfer_nan_df = pd.DataFrame(
        {
            "model_name": ["cnn", "rf"],
            "strategy": [np.nan, np.nan],
            "macro_f1": [0.71, 0.81],
        }
    )
    transfer_nan_paths = plot_transfer_vs_scratch(transfer_nan_df, tmp_path)
    transfer_nan_csv = pd.read_csv(transfer_nan_paths["csv"])
    assert not transfer_nan_csv.empty
    assert set(transfer_nan_csv["strategy"]) == {"scratch"}

    # multitask vs single task
    multi_df = pd.DataFrame(
        {
            "task_name": ["presence", "presence", "band", "band"],
            "mode": ["single_task", "multitask", "single_task", "multitask"],
            "macro_f1": [0.75, 0.82, 0.63, 0.72],
        }
    )
    multitask_paths = plot_multitask_vs_single_task(multi_df, tmp_path)
    assert multitask_paths["png"].exists()
    assert multitask_paths["svg"].exists()
    assert multitask_paths["csv"].exists()


def test_thesis_figure_naming_convention_alignment(tmp_path: Path) -> None:
    dataset_df = pd.DataFrame({"class": ["present", "absent", "present"]})
    feature_df = pd.DataFrame(
        {"feature": ["power_dB", "spectral_entropy"], "importance": [0.73, 0.27]}
    )
    metric_df = pd.DataFrame(
        {"model": ["rf"], "accuracy": [0.88], "f1": [0.87], "precision": [0.86]}
    )
    curve_df = pd.DataFrame(
        {
            "epoch": [1, 2],
            "train_loss": [0.7, 0.4],
            "val_loss": [0.72, 0.45],
            "accuracy": [0.65, 0.82],
            "f1": [0.62, 0.8],
        }
    )
    cm = np.array([[12, 3], [2, 11]])

    paths = {
        "dataset_summary": plot_dataset_summary(dataset_df, tmp_path),
        "feature_importance": plot_feature_importance(feature_df, tmp_path),
        "training_curves": plot_training_curves(curve_df, tmp_path),
        "confusion_matrix": plot_confusion_matrix(cm, ["0", "1"], tmp_path),
        "metrics_overview": plot_metrics_bar(metric_df, tmp_path),
    }

    for expected_stem, artifact_paths in paths.items():
        assert artifact_paths["png"].stem == expected_stem
        assert artifact_paths["svg"].stem == expected_stem
        assert artifact_paths["csv"].stem == expected_stem
        assert artifact_paths["png"].exists()
        assert artifact_paths["svg"].exists()
        assert artifact_paths["csv"].exists()


def test_build_run_summary_outputs_markdown_file(tmp_path: Path) -> None:
    summary = build_run_summary(
        run_name="smoke_run",
        output_dir=tmp_path,
        metrics={"accuracy": 0.83, "f1": 0.81},
        figures=["dataset_summary", "metrics_overview"],
        tables=["metrics_table"],
    )
    assert summary.exists()
    text = summary.read_text(encoding="utf-8")
    assert "# Run Summary: smoke_run" in text
    assert "accuracy: 0.8300" in text
    assert "dataset_summary" in text


def test_expected_thesis_figure_stems_helper() -> None:
    expected = expected_thesis_figure_stems()
    assert "dataset_summary" in expected
    assert "waveform_overview" in expected
    assert "spectrum_stft_summary" in expected
    assert "candidate_comparison" in expected
    assert "cv_comparison" in expected
    assert "ablation_comparison" in expected
    assert "ml_vs_cnn_comparison" in expected
    assert "overfitting_gap" in expected
    assert "per_class_metrics" in expected
    assert "task_comparison" in expected
    assert has_thesis_figure_stems(list(expected))
    assert not has_thesis_figure_stems(["dataset_summary", "feature_importance"])


def test_reporting_helpers_collect_artifacts_and_format_metrics(tmp_path: Path) -> None:
    (tmp_path / "figures").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tables").mkdir(parents=True, exist_ok=True)
    (tmp_path / "figures" / "dataset_summary.png").write_text("x", encoding="utf-8")
    (tmp_path / "figures" / "metrics_overview.png").write_text("x", encoding="utf-8")
    (tmp_path / "tables" / "feature_importance.csv").write_text("x", encoding="utf-8")

    artifacts = collect_run_artifacts(tmp_path)
    assert artifacts["figures"] == ["dataset_summary.png", "metrics_overview.png"]
    assert artifacts["tables"] == ["feature_importance.csv"]

    assert format_metric_value(0.123456) == "0.1235"
    assert format_metric_value(3) == "3"
    assert format_metric_value("cpu") == "cpu"


def test_reporting_risk_helpers_and_summary_sections(tmp_path: Path) -> None:
    cv_aggregates = summarize_cv_aggregates(
        [
            {"model": "rf", "cv_mean": 0.87, "cv_std": 0.02},
            {"model": "cnn", "cv_mean": 0.9, "cv_std": 0.03},
        ]
    )
    assert cv_aggregates["best_cv_mean"] == 0.9
    assert cv_aggregates["mean_cv_std"] == 0.025

    ablation_notes = build_ablation_notes(
        [
            {"variant": "baseline", "score": 0.91},
            {"variant": "w/o entropy", "score": 0.85},
            {"variant": "w/o cov", "score": 0.8},
        ]
    )
    assert "baseline: 0.9100" in ablation_notes[0]
    assert "w/o entropy: 0.8500" in ablation_notes[1]

    risk_note = build_overfitting_risk_note(max_gap=0.12, warning_threshold=0.05)
    assert "High overfitting risk" in risk_note

    cross_dataset_records = [
        {"dataset": "CognitiveRadio", "accuracy": 0.91, "f1": 0.9},
        {"dataset": "ElectricalFault", "accuracy": 0.88, "f1": 0.87},
        {"dataset": "VSB", "accuracy": 0.84, "f1": 0.82},
    ]
    benchmark_summary = summarize_cross_dataset_benchmark(cross_dataset_records)
    benchmark_notes = build_cross_dataset_benchmark_notes(cross_dataset_records)
    assert benchmark_summary["dataset_count"] == 3
    assert benchmark_summary["best_dataset"] == "CognitiveRadio"
    assert benchmark_summary["best_accuracy"] == 0.91
    assert "VSB: accuracy=0.8400, f1=0.8200" in benchmark_notes

    summary = build_run_summary(
        run_name="risk_run",
        output_dir=tmp_path,
        metrics={"accuracy": 0.83, "f1": 0.81},
        figures=["overfitting_gap", "cv_comparison", "ablation_comparison"],
        tables=["cv_table", "ablation_table"],
        cv_aggregates=cv_aggregates,
        ablation_notes=ablation_notes,
        overfitting_risk_note=risk_note,
        cross_dataset_benchmark_summary=benchmark_summary,
        cross_dataset_benchmark_notes=benchmark_notes,
    )
    text = summary.read_text(encoding="utf-8")
    assert "## CV Aggregates" in text
    assert "best_cv_mean: 0.9000" in text
    assert "## Ablation Notes" in text
    assert "w/o entropy: 0.8500" in text
    assert "## Overfitting/Risk Note" in text
    assert "High overfitting risk" in text
    assert "## Cross-Dataset Benchmark Summary" in text
    assert "best_dataset: CognitiveRadio" in text
    assert "## Cross-Dataset Benchmark Notes" in text
    assert "VSB: accuracy=0.8400, f1=0.8200" in text
