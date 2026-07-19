from __future__ import annotations

import json
from pathlib import Path


def test_all_notebooks_are_valid_v4_json() -> None:
    expected = ["fable_qwen3_8b_colab_l4.ipynb"]
    root = Path(__file__).resolve().parents[1] / "notebooks"
    assert sorted(path.name for path in root.glob("*.ipynb")) == expected
    for name in expected:
        value = json.loads((root / name).read_text(encoding="utf-8"))
        assert value["nbformat"] == 4
        assert value["cells"]
        for index, cell in enumerate(value["cells"]):
            if cell.get("cell_type") == "code":
                compile("".join(cell.get("source", [])), f"{name}:cell-{index}", "exec")
