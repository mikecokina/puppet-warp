from __future__ import annotations

import json
from logging import config
from pathlib import Path
from typing import Any


def _load_conf(path: Path) -> None:
    conf_dict: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    config.dictConfig(conf_dict)


def initialize_logger() -> None:
    config_path = Path(__file__).parent / "logging_schemas" / "default.json"
    _load_conf(config_path)
