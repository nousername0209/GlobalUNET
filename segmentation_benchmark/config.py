from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str
    model: str
    data_root: Path = Path("data")
    output_dir: Path = Path("runs")
    epochs: int = 50
    batch_size: int = 4
    num_workers: int = 4
    image_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "cuda"
    amp: bool = True
    download_voc: bool = False
    cityscapes_split: str = "val"
    inference_warmup: int = 10
    inference_iters: int = 50

    @property
    def num_classes(self) -> int:
        if self.dataset == "voc2012":
            return 21
        if self.dataset == "cityscapes":
            return 19
        raise ValueError(f"Unknown dataset: {self.dataset}")

    @property
    def ignore_index(self) -> int:
        return 255

    @property
    def run_name(self) -> str:
        return f"{self.dataset}_{self.model}_seed{self.seed}"

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.run_name
