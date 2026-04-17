from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


RAW_PREFIX = "Electrical Fault detection and classification"
OUTPUT_DIRNAME = "electrical_fault_detection"


def _find_raw_dataset_dir(project_root: Path) -> Path:
    candidates = [
        path
        for path in project_root.iterdir()
        if path.is_dir() and path.name.startswith(RAW_PREFIX)
    ]
    if not candidates:
        raise FileNotFoundError(
            "未找到 Electrical Fault 原始目录。请确认下载后的目录仍位于仓库根目录。"
        )
    return sorted(candidates, key=lambda item: len(item.name.strip()))[0]


def _clean_detect_dataset(raw_dir: Path) -> pd.DataFrame:
    detect_path = raw_dir / "detect_dataset.csv"
    if not detect_path.exists():
        raise FileNotFoundError(f"缺少 detect_dataset.csv: {detect_path}")
    detect_df = pd.read_csv(detect_path)
    drop_columns = [
        column
        for column in detect_df.columns
        if str(column).startswith("Unnamed:") and detect_df[column].isna().all()
    ]
    if drop_columns:
        detect_df = detect_df.drop(columns=drop_columns)
    return detect_df


def _build_fault_code_dataset(raw_dir: Path) -> pd.DataFrame:
    class_path = raw_dir / "classData.csv"
    if not class_path.exists():
        raise FileNotFoundError(f"缺少 classData.csv: {class_path}")
    class_df = pd.read_csv(class_path)
    indicator_columns = ["G", "C", "B", "A"]
    missing_columns = [column for column in indicator_columns if column not in class_df.columns]
    if missing_columns:
        raise ValueError(f"classData.csv 缺少故障编码列: {missing_columns}")

    encoded = (
        class_df[indicator_columns]
        .fillna(0)
        .astype(int)
        .astype(str)
        .agg("".join, axis=1)
    )
    fault_df = class_df.copy()
    fault_df["fault_code"] = encoded.map(lambda item: f"fault_{item}")
    return fault_df


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = _find_raw_dataset_dir(project_root)
    output_dir = project_root / "data" / OUTPUT_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)

    detect_df = _clean_detect_dataset(raw_dir)
    fault_df = _build_fault_code_dataset(raw_dir)

    detect_out = output_dir / "detect_dataset.csv"
    fault_out = output_dir / "classData_fault_code.csv"
    metadata_out = output_dir / "dataset_summary.json"

    detect_df.to_csv(detect_out, index=False)
    fault_df.to_csv(fault_out, index=False)

    summary = {
        "raw_source_dir": str(raw_dir),
        "normalized_dir": str(output_dir),
        "detect_dataset": {
            "rows": int(len(detect_df)),
            "columns": list(detect_df.columns),
            "label_column": "Output (S)",
            "label_counts": {
                str(label): int(count)
                for label, count in detect_df["Output (S)"].value_counts().sort_index().items()
            },
        },
        "classData_fault_code": {
            "rows": int(len(fault_df)),
            "columns": list(fault_df.columns),
            "label_column": "fault_code",
            "label_counts": {
                str(label): int(count)
                for label, count in fault_df["fault_code"].value_counts().sort_index().items()
            },
        },
    }
    metadata_out.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[electrical_fault] raw_dir={raw_dir}")
    print(f"[electrical_fault] normalized_dir={output_dir}")
    print(f"[electrical_fault] wrote {detect_out.name} rows={len(detect_df)}")
    print(f"[electrical_fault] wrote {fault_out.name} rows={len(fault_df)}")
    print(f"[electrical_fault] wrote {metadata_out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
