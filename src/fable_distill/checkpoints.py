from __future__ import annotations

import json
import re
from pathlib import Path


CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


def is_valid_checkpoint(path: Path) -> bool:
    if not path.is_dir() or not CHECKPOINT_RE.match(path.name):
        return False
    state = path / "trainer_state.json"
    adapter = path / "adapter_config.json"
    model = path / "pytorch_model.bin"
    safetensors = path / "model.safetensors"
    if not state.exists() or not (adapter.exists() or model.exists() or safetensors.exists()):
        return False
    try:
        json.loads(state.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return True


def find_latest_checkpoint(output_dir: str | Path) -> str | None:
    root = Path(output_dir)
    candidates: list[tuple[int, Path]] = []
    if not root.exists():
        return None
    for child in root.iterdir():
        match = CHECKPOINT_RE.match(child.name)
        if match and is_valid_checkpoint(child):
            candidates.append((int(match.group(1)), child))
    if not candidates:
        return None
    return str(max(candidates, key=lambda item: item[0])[1])

