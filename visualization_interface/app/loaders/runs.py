from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def list_run_files(results_dir: Path) -> list[Path]:
    if not results_dir.exists():
        return []
    return sorted(results_dir.glob("*.json"))


def load_run_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
