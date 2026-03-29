from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DatasetInfo:
    id: str
    name: str
    schema: str
    source_url: str
    description: str
    download_hint: str


_DATASETS = (
    DatasetInfo(
        id="cognitive_radio_spectrum",
        name="Cognitive Radio Spectrum Sensing Dataset",
        schema="tabular",
        source_url="https://www.kaggle.com/datasets/sunilsinghcse/cognitive-radio-spectrum-sensing-dataset",
        description="Primary project dataset used for the main thesis experiments, including multi-task spectrum sensing, transfer learning, and model comparison.",
        download_hint="Place the downloaded CSV under data/Cognitive Radio Spectrum Sensing Dataset.csv",
    ),
    DatasetInfo(
        id="emi_uci",
        name="Electromagnetic Interference Dataset (UCI/Kaggle mirror)",
        schema="tabular",
        source_url="https://www.kaggle.com/datasets/ucimachinelearning/electromagnetic-interference-dataset",
        description="Primary EMC/EMI dataset for interference identification tasks.",
        download_hint="kaggle datasets download -d ucimachinelearning/electromagnetic-interference-dataset",
    ),
    DatasetInfo(
        id="vsb_power_line_fault",
        name="VSB Power Line Fault Detection",
        schema="waveform",
        source_url="https://www.kaggle.com/competitions/vsb-power-line-fault-detection",
        description="Auxiliary large waveform-style dataset used for supplementary fault-diagnosis and generalization experiments, not the main thesis result line.",
        download_hint="kaggle competitions download -c vsb-power-line-fault-detection",
    ),
    DatasetInfo(
        id="electrical_fault",
        name="Electrical Fault Detection and Classification",
        schema="tabular",
        source_url="https://www.kaggle.com/datasets/esathyaprakash/electrical-fault-detection-and-classification",
        description="Structured multi-class electrical fault dataset for baseline model benchmarking.",
        download_hint="kaggle datasets download -d esathyaprakash/electrical-fault-detection-and-classification",
    ),
)


DATASET_REGISTRY: dict[str, DatasetInfo] = {item.id: item for item in _DATASETS}


def get_dataset_info(dataset_id: str) -> dict[str, Any]:
    if dataset_id not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY))
        raise KeyError(f"Unknown dataset_id='{dataset_id}'. Available: {available}")
    return asdict(DATASET_REGISTRY[dataset_id])


def list_datasets() -> list[dict[str, Any]]:
    return [asdict(item) for item in DATASET_REGISTRY.values()]
