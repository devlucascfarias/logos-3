from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from .filtering import sanitize_example
from .schemas import CanonicalExample, Message, Provenance
from .scoring import trajectory_heuristics
from .tools import canonical_tool_call, canonical_tool_result, normalize_tool_name


ROLE_ALIASES = {
    "human": "user",
    "user": "user",
    "system": "system",
    "assistant": "assistant",
    "gpt": "assistant",
    "model": "assistant",
    "tool": "tool",
    "tool_result": "tool",
    "function": "tool",
}

DEFAULT_SYSTEM = (
    "You are a software engineering agent operating in a repository. "
    "Use tools when necessary and verify changes with tests."
)


def _first(row: dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return default


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _message_from_raw(raw: Any) -> list[Message]:
    if isinstance(raw, str):
        return [Message(role="assistant", content=raw)]
    if not isinstance(raw, dict):
        return []
    raw_role = str(_first(raw, ("role", "from", "speaker", "type"), "assistant")).lower()
    role = ROLE_ALIASES.get(raw_role, "assistant")
    content = _first(raw, ("content", "value", "text", "message", "output"), "")
    name = _first(raw, ("name", "tool_name", "function_name"))

    tool_calls = raw.get("tool_calls")
    if role == "assistant" and tool_calls:
        calls = tool_calls if isinstance(tool_calls, list) else [tool_calls]
        rendered: list[str] = []
        for call in calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function") if isinstance(call.get("function"), dict) else call
            try:
                rendered.append(
                    canonical_tool_call(
                        str(_first(function, ("name", "tool_name"), "")),
                        _first(function, ("arguments", "args", "input"), {}),
                    )
                )
            except ValueError:
                continue
        if rendered:
            prefix = _as_text(content).strip()
            content = "\n".join(([prefix] if prefix else []) + rendered)

    if role == "tool":
        raw_name = str(name or _first(raw, ("tool",), "shell"))
        try:
            content = canonical_tool_result(raw_name, content)
            name = normalize_tool_name(raw_name)
        except ValueError:
            name = "shell"
            content = canonical_tool_result("shell", content)
    return [Message(role=role, content=_as_text(content), name=str(name) if name else None)]


def extract_messages(row: dict[str, Any]) -> list[Message]:
    candidate = _first(
        row,
        ("messages", "conversations", "conversation", "trajectory", "trace", "transcript"),
    )
    messages: list[Message] = []
    if isinstance(candidate, dict):
        candidate = _first(candidate, ("messages", "steps", "events"), [])
    if isinstance(candidate, list):
        for raw in candidate:
            messages.extend(_message_from_raw(raw))

    if not messages:
        system = _first(row, ("system", "system_prompt"))
        prompt = _first(row, ("prompt", "instruction", "task", "question", "input"))
        reasoning = _first(row, ("reasoning", "analysis", "thinking"))
        response = _first(row, ("response", "answer", "completion", "output"))
        action = _first(row, ("action", "tool_call"))
        tool_result = _first(row, ("tool_result", "observation"))
        if system:
            messages.append(Message("system", _as_text(system)))
        if prompt:
            messages.append(Message("user", _as_text(prompt)))
        if response or reasoning or action:
            assistant_parts: list[str] = []
            if reasoning:
                assistant_parts.append(f"<think>\n{_as_text(reasoning)}\n</think>")
            if action and isinstance(action, dict):
                try:
                    assistant_parts.append(
                        canonical_tool_call(
                            str(_first(action, ("name", "tool", "tool_name"), "")),
                            _first(action, ("arguments", "args", "input"), {}),
                        )
                    )
                except ValueError:
                    pass
            if response:
                assistant_parts.append(_as_text(response))
            messages.append(Message("assistant", "\n".join(assistant_parts)))
        if tool_result:
            messages.append(Message("tool", _as_text(tool_result), name="shell"))

    if messages and messages[0].role != "system":
        messages.insert(0, Message("system", DEFAULT_SYSTEM))
    return messages


def parse_source_row(
    row: dict[str, Any],
    *,
    source_dataset: str,
    source_revision: str,
    license_name: str,
    row_index: int,
    max_tool_output_chars: int = 12000,
) -> CanonicalExample | None:
    messages = extract_messages(row)
    if not messages:
        return None
    source_session_id = str(
        _first(row, ("session_id", "trajectory_id", "conversation_id", "run_id"), row_index)
    )
    source_example_id = str(_first(row, ("id", "example_id", "uuid"), row_index))
    repository_id = str(
        _first(row, ("repository_id", "repo_id", "repository", "repo", "project"), "unknown")
    )
    score = trajectory_heuristics([message.to_dict() for message in messages], row)
    example = CanonicalExample(
        session_id=f"{source_dataset}:{source_session_id}",
        repository_id=repository_id,
        task_family=str(_first(row, ("task_family", "category", "task_type"), "unknown")),
        language=str(_first(row, ("language", "programming_language", "lang"), "unknown")),
        provenance=Provenance(
            source_dataset=source_dataset,
            source_revision=source_revision,
            source_session_id=source_session_id,
            source_example_id=source_example_id,
            license=license_name,
        ),
        messages=messages,
        verified_success=bool(
            _first(row, ("verified_success", "success", "tests_passed", "passed"), False)
        ),
        quality_score=score,
        metadata={"raw_row_index": row_index},
    )
    return sanitize_example(example, max_tool_output_chars=max_tool_output_chars)
