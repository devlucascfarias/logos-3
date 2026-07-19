from __future__ import annotations

from pathlib import Path

import pytest

from fable_distill.harness import IsolatedHarness, validate_command


def test_blocks_destructive_and_network_commands() -> None:
    with pytest.raises(ValueError):
        validate_command("rm -rf .")
    with pytest.raises(ValueError):
        validate_command("curl https://example.com")
    validate_command("python -m pytest -q")


def test_runs_in_a_copied_directory(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "value.txt").write_text("original", encoding="utf-8")
    with IsolatedHarness(source, timeout_seconds=5) as harness:
        result = harness.run("python -c \"from pathlib import Path; print(Path('value.txt').read_text())\"")
        assert result.exit_code == 0
        assert "original" in result.stdout
    assert (source / "value.txt").read_text(encoding="utf-8") == "original"


def test_canonical_file_tools_stay_in_workspace(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    with IsolatedHarness(source, timeout_seconds=5) as harness:
        written = harness.execute_tool(
            "write_file", {"path": "src/demo.py", "content": "answer = 42\n"}
        )
        assert written["written"] == "src/demo.py"
        result = harness.execute_tool("search", {"path": "src", "pattern": "answer"})
        assert result["matches"][0]["line"] == 1
        with pytest.raises(ValueError):
            harness.execute_tool("read_file", {"path": "../secret.txt"})
