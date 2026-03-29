from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pandas as pd

from emc_diag.visualization import plot_dataset_summary, plot_feature_importance


ROOT = Path(__file__).resolve().parents[1]
THESIS_DIR = ROOT / "artifacts" / "reports" / "thesis-assets-final"
OUTPUT_DIR = ROOT / "artifacts" / "reports" / "paper-handoff-package"

REPORT_SOURCE = THESIS_DIR / "final_paper_report.md"
REPORT_TARGET = OUTPUT_DIR / "01_final_paper_report.md"
README_TARGET = OUTPUT_DIR / "00_README.md"
EXPLORATION_INDEX = OUTPUT_DIR / "02_exploration_gallery.md"

COPY_FILES = [
    ("final_summary.md", "03_final_summary.md"),
    ("final_metrics.csv", "04_final_metrics.csv"),
    ("benchmark_metrics_merged.csv", "05_benchmark_metrics_merged.csv"),
    ("thesis_figures_manifest.csv", "06_thesis_figures_manifest.csv"),
]

PREPARED_DIRS = [
    ROOT / "artifacts" / "prepared_cognitive_presence",
    ROOT / "artifacts" / "prepared_cognitive_burst_duration",
    ROOT / "artifacts" / "prepared_cognitive_drift_type",
    ROOT / "artifacts" / "prepared_cognitive_frequency_band",
    ROOT / "artifacts" / "prepared_vsb_fault_detection",
    ROOT / "artifacts" / "prepared_vsb_fault_cnn_lstm",
]


def _slugify(value: str) -> str:
    normalized = "".join(char.lower() if char.isalnum() else "-" for char in value)
    compact = "-".join(part for part in normalized.split("-") if part)
    return compact or "item"


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_dataset_frame(class_counts: dict[str, int]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for label, count in class_counts.items():
        rows.extend({"class": str(label)} for _ in range(int(count)))
    return pd.DataFrame(rows or [{"class": "unknown"}])


def _build_exploration_assets(prepared_dir: Path) -> tuple[str, list[str]]:
    metadata = _load_json(prepared_dir / "metadata.json")
    statistics = _load_json(prepared_dir / "statistics.json")
    feature_df = pd.read_csv(prepared_dir / "feature_importance.csv")

    task_name = str(metadata.get("task_name", prepared_dir.name))
    slug = _slugify(task_name)
    target_dir = OUTPUT_DIR / "figures" / "exploration" / slug
    target_dir.mkdir(parents=True, exist_ok=True)

    dataset_paths = plot_dataset_summary(_build_dataset_frame(statistics.get("class_counts", {})), target_dir)
    feature_paths = plot_feature_importance(feature_df, target_dir)

    _copy(prepared_dir / "metadata.json", target_dir / "metadata.json")
    _copy(prepared_dir / "statistics.json", target_dir / "statistics.json")
    _copy(prepared_dir / "feature_importance.csv", target_dir / "feature_importance_source.csv")

    lines = [
        f"### {task_name}",
        "",
        f"- Source: `{prepared_dir.relative_to(ROOT)}`",
        "",
        f"![{task_name} 数据分布](figures/exploration/{slug}/{dataset_paths['png'].name})",
        "",
        f"![{task_name} 特征重要性](figures/exploration/{slug}/{feature_paths['png'].name})",
        "",
    ]
    return task_name, lines


def _rewrite_report_with_local_images() -> None:
    report_text = REPORT_SOURCE.read_text(encoding="utf-8")
    image_pattern = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

    def replace(match: re.Match[str]) -> str:
        alt_text = match.group(1)
        raw_path = match.group(2)
        source_path = (REPORT_SOURCE.parent / raw_path).resolve()
        folder_name = "report"
        if source_path.parent == THESIS_DIR / "figures":
            folder_name = "final"
        target_name = _slugify(source_path.parent.parent.name) + "-" + source_path.name if "runs" in raw_path else source_path.name
        target_path = OUTPUT_DIR / "figures" / folder_name / target_name
        _copy(source_path, target_path)
        relative_path = target_path.relative_to(OUTPUT_DIR).as_posix()
        return f"![{alt_text}]({relative_path})"

    rewritten = image_pattern.sub(replace, report_text)
    REPORT_TARGET.write_text(rewritten, encoding="utf-8")


def build_package() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for source_name, target_name in COPY_FILES:
        _copy(THESIS_DIR / source_name, OUTPUT_DIR / target_name)

    _rewrite_report_with_local_images()

    exploration_sections: list[str] = [
        "# 数据探索图册",
        "",
        "本文件汇总了论文写作需要用到的训练前探索图，避免在 prepared 目录和 runs 目录之间来回查找。",
        "",
    ]
    collected_tasks: list[str] = []
    for prepared_dir in PREPARED_DIRS:
        if not prepared_dir.exists():
            continue
        required = [
            prepared_dir / "metadata.json",
            prepared_dir / "statistics.json",
            prepared_dir / "feature_importance.csv",
        ]
        if not all(path.exists() for path in required):
            continue
        task_name, section_lines = _build_exploration_assets(prepared_dir)
        collected_tasks.append(task_name)
        exploration_sections.extend(section_lines)
    EXPLORATION_INDEX.write_text("\n".join(exploration_sections), encoding="utf-8")

    readme_lines = [
        "# 论文移交包",
        "",
        "该目录用于向论文撰写者移交最终可用材料，不需要再去翻 prepared、runs 和 thesis-assets-final 目录。",
        "",
        "## 目录说明",
        "",
        "- `01_final_paper_report.md`：最终研究报告，已嵌入关键图表。",
        "- `02_exploration_gallery.md`：训练前数据探索图册。",
        "- `03_final_summary.md`：摘要版总结。",
        "- `04_final_metrics.csv`：最终指标总表。",
        "- `05_benchmark_metrics_merged.csv`：合并后的 benchmark 结果。",
        "- `06_thesis_figures_manifest.csv`：全图索引。",
        "- `figures/final/`：最终总览图。",
        "- `figures/report/`：报告正文引用的训练/评估图。",
        "- `figures/exploration/`：数据探索图。",
        "",
        "## 已收录探索任务",
        "",
    ]
    readme_lines.extend(f"- {task}" for task in collected_tasks)
    readme_lines.append("")
    README_TARGET.write_text("\n".join(readme_lines), encoding="utf-8")


if __name__ == "__main__":
    build_package()
