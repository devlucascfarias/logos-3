from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any

from .io import stable_json


class ToolName(str, Enum):
    LIST_FILES = "list_files"
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    APPLY_PATCH = "apply_patch"
    SEARCH = "search"
    SHELL = "shell"


TOOL_ALIASES = {
    "bash": ToolName.SHELL.value,
    "shell": ToolName.SHELL.value,
    "terminal": ToolName.SHELL.value,
    "run_command": ToolName.SHELL.value,
    "execute": ToolName.SHELL.value,
    "read": ToolName.READ_FILE.value,
    "readfile": ToolName.READ_FILE.value,
    "read_file": ToolName.READ_FILE.value,
    "write": ToolName.WRITE_FILE.value,
    "writefile": ToolName.WRITE_FILE.value,
    "write_file": ToolName.WRITE_FILE.value,
    "edit": ToolName.APPLY_PATCH.value,
    "str_replace": ToolName.APPLY_PATCH.value,
    "applypatch": ToolName.APPLY_PATCH.value,
    "apply_patch": ToolName.APPLY_PATCH.value,
    "grep": ToolName.SEARCH.value,
    "ripgrep": ToolName.SEARCH.value,
    "search": ToolName.SEARCH.value,
    "glob": ToolName.LIST_FILES.value,
    "ls": ToolName.LIST_FILES.value,
    "listfiles": ToolName.LIST_FILES.value,
    "list_files": ToolName.LIST_FILES.value,
}

ARGUMENT_ALIASES = {
    ToolName.SHELL.value: {"cmd": "command", "script": "command"},
    ToolName.READ_FILE.value: {"file": "path", "file_path": "path"},
    ToolName.WRITE_FILE.value: {"file": "path", "file_path": "path", "text": "content"},
    ToolName.APPLY_PATCH.value: {
        "patch_text": "patch",
        "diff": "patch",
        "file": "path",
        "file_path": "path",
        "old_string": "old_text",
        "new_string": "new_text",
    },
    ToolName.SEARCH.value: {"query": "pattern", "regex": "pattern"},
    ToolName.LIST_FILES.value: {"directory": "path", "dir": "path"},
}

ANSI_RE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def normalize_tool_name(name: str) -> str:
    key = re.sub(r"[^a-z0-9_]+", "", str(name).strip().lower())
    try:
        return TOOL_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported tool name: {name!r}") from exc


def normalize_arguments(name: str, arguments: Any) -> dict[str, Any]:
    if arguments is None:
        result: dict[str, Any] = {}
    elif isinstance(arguments, str):
        try:
            decoded = json.loads(arguments)
        except json.JSONDecodeError:
            decoded = {"command": arguments} if name == ToolName.SHELL.value else {"value": arguments}
        result = decoded if isinstance(decoded, dict) else {"value": decoded}
    elif isinstance(arguments, dict):
        result = dict(arguments)
    else:
        result = {"value": arguments}

    aliases = ARGUMENT_ALIASES.get(name, {})
    normalized: dict[str, Any] = {}
    for key, value in result.items():
        clean_key = aliases.get(str(key), str(key))
        if clean_key in {"timeout_ms", "request_id", "call_id", "runtime", "metadata"}:
            continue
        normalized[clean_key] = value
    return dict(sorted(normalized.items()))


def validate_tool_arguments(name: str, arguments: dict[str, Any]) -> None:
    required_any = {
        ToolName.SHELL.value: (("command",),),
        ToolName.READ_FILE.value: (("path",),),
        ToolName.WRITE_FILE.value: (("path", "content"),),
        ToolName.APPLY_PATCH.value: (("patch",), ("path", "old_text", "new_text")),
        ToolName.SEARCH.value: (("pattern",),),
        ToolName.LIST_FILES.value: ((),),
    }
    alternatives = required_any[name]
    if not any(all(key in arguments for key in alternative) for alternative in alternatives):
        rendered = " or ".join("+".join(keys) for keys in alternatives)
        raise ValueError(f"Invalid arguments for {name}: requires {rendered}")
    for key in ("command", "path", "pattern", "patch"):
        if key in arguments and not isinstance(arguments[key], str):
            raise ValueError(f"Invalid arguments for {name}: {key} must be a string")


def canonical_tool_call(name: str, arguments: Any) -> str:
    normalized_name = normalize_tool_name(name)
    normalized_arguments = normalize_arguments(normalized_name, arguments)
    validate_tool_arguments(normalized_name, normalized_arguments)
    payload = {
        "name": normalized_name,
        "arguments": normalized_arguments,
    }
    return f"<tool_call>\n{stable_json(payload)}\n</tool_call>"


def parse_tool_call(text: str) -> dict[str, Any]:
    match = TOOL_CALL_RE.search(text)
    if not match:
        raise ValueError("No <tool_call> JSON object found")
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid tool-call JSON: {exc}") from exc
    if not isinstance(payload, dict) or "name" not in payload:
        raise ValueError("Tool call must be an object with a name")
    name = normalize_tool_name(str(payload["name"]))
    arguments = normalize_arguments(name, payload.get("arguments", {}))
    validate_tool_arguments(name, arguments)
    return {"name": name, "arguments": arguments}


def truncate_head_tail(text: str, max_chars: int = 12000) -> str:
    clean = ANSI_RE.sub("", str(text))
    if max_chars < 32:
        raise ValueError("max_chars must be at least 32")
    if len(clean) <= max_chars:
        return clean
    marker = "\n<truncated>\n"
    available = max_chars - len(marker)
    head = available // 2
    tail = available - head
    return clean[:head] + marker + clean[-tail:]


def canonical_tool_result(name: str, content: Any, max_chars: int = 12000) -> str:
    normalized_name = normalize_tool_name(name)
    result = truncate_head_tail(str(content), max_chars=max_chars)
    return f'<tool_result name="{normalized_name}">\n{result}\n</tool_result>'


def normalize_embedded_tool_calls(content: str) -> str:
    """Normalize already tagged tool calls, leaving ordinary prose untouched."""

    def replace(match: re.Match[str]) -> str:
        try:
            payload = json.loads(match.group(1))
            return canonical_tool_call(payload.get("name", ""), payload.get("arguments", {}))
        except (ValueError, json.JSONDecodeError, TypeError):
            return match.group(0)

    return TOOL_CALL_RE.sub(replace, str(content))


def validate_embedded_tool_calls(content: str) -> bool:
    value = str(content)
    if "<tool_call>" not in value and "</tool_call>" not in value:
        return True
    matches = list(TOOL_CALL_RE.finditer(value))
    if len(matches) != value.count("<tool_call>") or len(matches) != value.count("</tool_call>"):
        return False
    try:
        for match in matches:
            parse_tool_call(match.group(0))
    except ValueError:
        return False
    return True
