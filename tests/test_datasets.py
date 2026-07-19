from __future__ import annotations

from fable_distill.datasets import extract_messages, parse_source_row
from fable_distill.tools import parse_tool_call


def test_parse_messages_and_openai_tool_call() -> None:
    row = {
        "id": "row-1",
        "session_id": "session-1",
        "repo": "owner/repo",
        "messages": [
            {"role": "user", "content": "Run tests"},
            {
                "role": "assistant",
                "content": "I will verify.",
                "tool_calls": [
                    {"function": {"name": "Bash", "arguments": '{"cmd":"pytest -q"}'}}
                ],
            },
        ],
        "success": True,
    }
    messages = extract_messages(row)
    assert messages[0].role == "system"
    parsed = parse_tool_call(messages[-1].content)
    assert parsed["name"] == "shell"
    example = parse_source_row(
        row,
        source_dataset="unit/source",
        source_revision="abc",
        license_name="test",
        row_index=0,
    )
    assert example is not None
    assert example.repository_id == "owner/repo"
    assert example.verified_success
    assert example.provenance.source_revision == "abc"


def test_parse_separate_reasoning_and_action() -> None:
    messages = extract_messages(
        {
            "prompt": "Inspect",
            "reasoning": "I need to list files.",
            "action": {"tool": "Glob", "args": {"directory": "."}},
        }
    )
    assert "<think>" in messages[-1].content
    assert parse_tool_call(messages[-1].content)["name"] == "list_files"

