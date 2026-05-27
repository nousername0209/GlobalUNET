from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return _with_run_dir(config)


def _with_run_dir(config: dict) -> dict:
    cfg = deepcopy(config)
    cfg["output"]["run_dir"] = str(Path(cfg["output"]["root"]) / cfg["experiment_name"])
    return cfg
