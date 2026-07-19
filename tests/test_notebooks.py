from __future__ import annotations

import json
from pathlib import Path


def test_all_notebooks_are_valid_v4_json() -> None:
    expected = [
        "00_environment_check.ipynb",
        "01_download_and_audit.ipynb",
        "02_preprocess.ipynb",
        "03_train_qlora.ipynb",
        "04_merge_and_export.ipynb",
        "05_evaluate.ipynb",
    ]
    root = Path(__file__).resolve().parents[1] / "notebooks"
    for name in expected:
        value = json.loads((root / name).read_text(encoding="utf-8"))
        assert value["nbformat"] == 4
        assert value["cells"]

