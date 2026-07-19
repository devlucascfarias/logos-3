from __future__ import annotations

import re
from dataclasses import replace

from .schemas import CanonicalExample, Message
from .tools import (
    ANSI_RE,
    normalize_embedded_tool_calls,
    truncate_head_tail,
    validate_embedded_tool_calls,
)


EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
API_KEY_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|secret|password)\b"
    r"(\s*[:=]\s*)"
    r"([\"']?)[A-Za-z0-9_./+=-]{8,}\3"
)
KNOWN_TOKEN_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]{20,}|"
    r"hf_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16})\b"
)
WINDOWS_USER_RE = re.compile(r"(?i)\b[A-Z]:\\Users\\[^\\\s]+")
POSIX_HOME_RE = re.compile(r"(?<![\w/])/(?:home|Users)/[^/\s]+")
RATE_LIMIT_RE = re.compile(r"(?im)^.*(?:rate limit|too many requests|http 429).*$\n?")
RUNTIME_BANNER_RE = re.compile(
    r"(?im)^.*(?:welcome to google colab|nvidia-smi \d|cuda version:).*$\n?"
)
MINIFIED_LINE_RE = re.compile(r"^.{20000,}$", re.MULTILINE)


def redact_sensitive(text: str) -> str:
    value = EMAIL_RE.sub("<redacted_email>", str(text))
    value = API_KEY_RE.sub(r"\1\2<redacted_secret>", value)
    value = KNOWN_TOKEN_RE.sub("<redacted_token>", value)
    value = WINDOWS_USER_RE.sub(lambda _: r"C:\Users\<redacted_user>", value)
    value = POSIX_HOME_RE.sub("/home/<redacted_user>", value)
    return value


def clean_text(text: str, max_chars: int = 12000, is_tool_output: bool = False) -> str:
    value = ANSI_RE.sub("", str(text).replace("\x00", ""))
    value = redact_sensitive(value)
    if is_tool_output:
        value = RATE_LIMIT_RE.sub("", value)
        value = RUNTIME_BANNER_RE.sub("", value)
        value = MINIFIED_LINE_RE.sub("<minified_or_binary_content_removed>", value)
        value = truncate_head_tail(value, max_chars=max_chars)
    return value.strip()


def sanitize_example(
    example: CanonicalExample,
    max_tool_output_chars: int = 12000,
) -> CanonicalExample | None:
    cleaned: list[Message] = []
    for message in example.messages:
        is_tool = message.role == "tool"
        content = clean_text(
            message.content,
            max_chars=max_tool_output_chars,
            is_tool_output=is_tool,
        )
        if message.role == "assistant":
            content = normalize_embedded_tool_calls(content)
            if not validate_embedded_tool_calls(content):
                return None
        if not content and message.role in {"user", "assistant"}:
            continue
        cleaned.append(Message(role=message.role, content=content, name=message.name))

    if not cleaned or not any(item.role == "assistant" for item in cleaned):
        return None
    if not any(item.role == "user" for item in cleaned):
        return None
    return replace(example, messages=cleaned, example_id="")
