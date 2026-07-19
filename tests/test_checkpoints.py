from __future__ import annotations

from pathlib import Path

from fable_distill.checkpoints import find_latest_checkpoint


def create_checkpoint(root: Path, step: int, valid: bool = True) -> None:
    path = root / f"checkpoint-{step}"
    path.mkdir()
    (path / "trainer_state.json").write_text("{}" if valid else "{", encoding="utf-8")
    (path / "adapter_config.json").write_text("{}", encoding="utf-8")


def test_finds_latest_valid_checkpoint(tmp_path: Path) -> None:
    create_checkpoint(tmp_path, 10)
    create_checkpoint(tmp_path, 30, valid=False)
    create_checkpoint(tmp_path, 20)
    assert find_latest_checkpoint(tmp_path) == str(tmp_path / "checkpoint-20")

