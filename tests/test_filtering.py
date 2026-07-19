from __future__ import annotations

from fable_distill.filtering import clean_text, redact_sensitive, sanitize_example
from fable_distill.schemas import CanonicalExample, Message, Provenance


def test_redacts_pii_secrets_and_user_paths() -> None:
    text = (
        "email me at person@example.com; api_key=abcdefghijklmnop; "
        "token ghp_abcdefghijklmnopqrstuvwxyz; C:\\Users\\Alice\\repo; /home/bob/project"
    )
    cleaned = redact_sensitive(text)
    assert "person@example.com" not in cleaned
    assert "abcdefghijklmnop" not in cleaned
    assert "ghp_" not in cleaned
    assert "Alice" not in cleaned and "/home/bob" not in cleaned


def test_tool_output_strips_ansi_and_truncates() -> None:
    cleaned = clean_text("\x1b[31mERROR\x1b[0m" + "x" * 200, max_chars=64, is_tool_output=True)
    assert "\x1b" not in cleaned
    assert "<truncated>" in cleaned
    assert len(cleaned) == 64


def test_empty_invalid_example_is_removed() -> None:
    example = CanonicalExample(
        messages=[Message("assistant", "answer")],
        provenance=Provenance("unit"),
        session_id="s",
        repository_id="r",
    )
    assert sanitize_example(example) is None


def test_invalid_tool_call_example_is_removed() -> None:
    example = CanonicalExample(
        messages=[
            Message("user", "run"),
            Message("assistant", '<tool_call>{"name":"Bash","arguments":{}}</tool_call>'),
        ],
        provenance=Provenance("unit"),
        session_id="s",
        repository_id="r",
    )
    assert sanitize_example(example) is None
