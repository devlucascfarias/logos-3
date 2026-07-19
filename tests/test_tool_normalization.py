from __future__ import annotations

import json

import pytest

from fable_distill.tools import (
    canonical_tool_call,
    canonical_tool_result,
    normalize_tool_name,
    parse_tool_call,
    truncate_head_tail,
)


@pytest.mark.parametrize(
    ("source", "expected"),
    [("Bash", "shell"), ("Read", "read_file"), ("Edit", "apply_patch"), ("Grep", "search")],
)
def test_aliases(source: str, expected: str) -> None:
    assert normalize_tool_name(source) == expected


def test_tool_call_is_valid_deterministic_json() -> None:
    rendered = canonical_tool_call("Bash", {"timeout_ms": 99, "cmd": "pytest -q"})
    parsed = parse_tool_call(rendered)
    assert parsed == {"name": "shell", "arguments": {"command": "pytest -q"}}
    payload = json.loads(rendered.split("\n")[1])
    assert list(payload["arguments"]) == ["command"]


def test_unknown_tool_is_rejected() -> None:
    with pytest.raises(ValueError):
        canonical_tool_call("delete_everything", {})


def test_invalid_argument_schema_is_rejected() -> None:
    with pytest.raises(ValueError):
        canonical_tool_call("Bash", {})


def test_structured_edit_arguments_are_canonicalized() -> None:
    rendered = canonical_tool_call(
        "Edit",
        {"file_path": "demo.py", "old_string": "old", "new_string": "new"},
    )
    parsed = parse_tool_call(rendered)
    assert parsed["arguments"] == {
        "new_text": "new",
        "old_text": "old",
        "path": "demo.py",
    }


def test_head_tail_truncation_preserves_both_ends() -> None:
    value = truncate_head_tail("A" * 100 + "Z" * 100, max_chars=80)
    assert len(value) == 80
    assert value.startswith("A") and value.endswith("Z")
    assert "<truncated>" in value
    assert "<tool_result" in canonical_tool_result("Bash", value)
