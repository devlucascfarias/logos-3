from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any


ROLE_ALIASES = {
    "human": "user",
    "user": "user",
    "system": "system",
    "assistant": "assistant",
    "gpt": "assistant",
    "model": "assistant",
    "tool": "tool",
    "function": "tool",
    "tool_result": "tool",
}

KNOWN_SECRET_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{16,}|gh[oprsu]_[A-Za-z0-9]{20,}|"
    r"hf_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16})\b"
)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
WINDOWS_HOME_RE = re.compile(r"(?i)\b[A-Z]:\\Users\\[^\\\s]+")
POSIX_HOME_RE = re.compile(r"(?<![\w/])/(?:home|Users)/[^/\s]+")
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
THINK_RE = re.compile(r"<think>(.*?)</think>", re.I | re.S)
DIFF_RE = re.compile(r"(?m)^(?:diff --git |@@\s+-\d|---\s+\S+\n\+\+\+\s+\S+)")


def _stable_hash(*parts: Any) -> str:
    value = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalized_message_marker(message: dict[str, str]) -> str:
    content = re.sub(r"\s+", " ", message["content"]).strip()
    return f"{message['role']}:{content}"


def _json_or_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        blocks: list[str] = []
        for item in value:
            if isinstance(item, dict) and "text" in item:
                blocks.append(str(item["text"]))
            else:
                blocks.append(_text(item))
        return "\n".join(block for block in blocks if block)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _render_tool_calls(value: Any) -> str:
    parsed = _json_or_value(value)
    calls = parsed if isinstance(parsed, list) else [parsed]
    rendered: list[str] = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if isinstance(function, dict):
            call = function
        name = call.get("name") or call.get("tool_name")
        arguments = _json_or_value(
            call.get("arguments", call.get("args", call.get("input", {})))
        )
        if not name or not isinstance(arguments, (dict, list)):
            continue
        payload = json.dumps(
            {"name": str(name), "arguments": arguments},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        rendered.append(f"<tool_call>\n{payload}\n</tool_call>")
    return "\n".join(rendered)


def _first(row: dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, "", [], {}):
            return value
    return default


def _raw_messages(row: dict[str, Any]) -> list[Any]:
    for key in (
        "messages",
        "messages_json",
        "conversations",
        "conversation",
        "trajectory",
        "trace",
        "transcript",
        "chatml",
    ):
        value = _json_or_value(row.get(key))
        if isinstance(value, dict):
            value = _first(value, ("messages", "steps", "events"), [])
        if isinstance(value, list):
            return value

    input_value = _json_or_value(row.get("input"))
    if isinstance(input_value, list):
        messages = list(input_value)
        output = _first(
            row,
            ("output", "response", "answer", "completion", "assistant_response"),
        )
        if output not in (None, ""):
            messages.append({"role": "assistant", "content": output})
        return messages
    return []


def normalize_messages(
    row: dict[str, Any], system_prompt: str
) -> list[dict[str, str]]:
    raw_messages = _raw_messages(row)
    normalized: list[dict[str, str]] = []
    for raw in raw_messages:
        if not isinstance(raw, dict):
            continue
        raw_role = str(
            _first(raw, ("role", "from", "speaker", "type"), "assistant")
        ).lower()
        role = ROLE_ALIASES.get(raw_role)
        if role is None:
            continue
        content = _text(
            _first(raw, ("content", "value", "text", "message", "output"), "")
        ).strip()
        reasoning = _text(raw.get("reasoning_content")).strip()
        if role == "assistant" and reasoning and "<think>" not in content.lower():
            content = f"<think>{reasoning}</think>\n{content}".strip()

        tool_calls = raw.get("tool_calls")
        if role == "assistant" and tool_calls:
            rendered = _render_tool_calls(tool_calls)
            if rendered:
                content = f"{content}\n{rendered}".strip()
        if role == "tool" and raw.get("name"):
            content = f"[tool={raw['name']}]\n{content}".strip()
        if content:
            normalized.append({"role": role, "content": content})

    if not normalized:
        system = _first(row, ("system", "system_prompt"))
        prompt = _first(
            row,
            (
                "prompt",
                "instruction",
                "task",
                "question",
                "user_prompt",
                "input",
            ),
        )
        response = _first(
            row,
            ("response", "answer", "completion", "output", "assistant_response"),
        )
        reasoning = _first(row, ("reasoning_content", "analysis", "thinking"))
        if system:
            normalized.append({"role": "system", "content": _text(system)})
        if prompt:
            normalized.append({"role": "user", "content": _text(prompt)})
        if response or reasoning:
            assistant = _text(response)
            if reasoning:
                assistant = f"<think>{_text(reasoning)}</think>\n{assistant}".strip()
            normalized.append({"role": "assistant", "content": assistant})

    if normalized and normalized[0]["role"] != "system":
        normalized.insert(0, {"role": "system", "content": system_prompt})
    return normalized


def contains_secret(messages: Sequence[dict[str, str]]) -> bool:
    return bool(KNOWN_SECRET_RE.search("\n".join(item["content"] for item in messages)))


def _clean_content(content: str, *, tool_output: bool, max_chars: int) -> str:
    value = ANSI_RE.sub("", content.replace("\x00", ""))
    value = EMAIL_RE.sub("<redacted_email>", value)
    value = WINDOWS_HOME_RE.sub(
        lambda _: r"C:\Users\<redacted_user>", value
    )
    value = POSIX_HOME_RE.sub("/home/<redacted_user>", value)
    if tool_output and len(value) > max_chars:
        half = max(1, (max_chars - 64) // 2)
        value = (
            value[:half]
            + "\n<tool_output_truncated_head_tail>\n"
            + value[-half:]
        )
    return value.strip()


def sanitize_messages(
    messages: Sequence[dict[str, str]], max_tool_output_chars: int
) -> list[dict[str, str]]:
    cleaned: list[dict[str, str]] = []
    for message in messages:
        content = _clean_content(
            message["content"],
            tool_output=message["role"] == "tool",
            max_chars=max_tool_output_chars,
        )
        if content:
            cleaned.append({"role": message["role"], "content": content})
    return cleaned


def _max_identical_run(messages: Sequence[dict[str, str]]) -> int:
    best = 0
    current = 0
    previous = None
    for message in messages:
        marker = (
            message["role"],
            re.sub(r"\s+", " ", message["content"]).strip().lower(),
        )
        current = current + 1 if marker == previous else 1
        previous = marker
        best = max(best, current)
    return best


def _row_success(row: dict[str, Any]) -> bool:
    for key in (
        "verified_success",
        "success",
        "resolved",
        "tests_passed",
        "passed",
        "patch_applied",
    ):
        value = row.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value > 0:
            return True
        if isinstance(value, str) and value.strip().lower() in {
            "true",
            "yes",
            "passed",
            "success",
            "resolved",
        }:
            return True
    return False


def _row_id(row: dict[str, Any], row_index: int) -> str:
    value = _first(
        row,
        (
            "id",
            "uuid",
            "example_id",
            "question_id",
            "instance_id",
            "session_id",
            "request_id",
        ),
        row_index,
    )
    return str(value)


def _group_id(row: dict[str, Any], source_name: str, source_id: str) -> str:
    value = _first(
        row,
        (
            "repository_id",
            "repo_id",
            "repository",
            "repo",
            "project",
            "session_id",
            "conversation_id",
            "request_id",
        ),
    )
    return f"{source_name}:{value if value is not None else source_id}"


def reasoning_band(
    messages: Sequence[dict[str, str]],
    text_token_counter: Callable[[str], int] | None = None,
) -> str:
    reasoning = "\n".join(
        match.group(1)
        for message in messages
        if message["role"] == "assistant"
        for match in THINK_RE.finditer(message["content"])
    ).strip()
    if not reasoning:
        return "direct"
    count = (
        text_token_counter(reasoning)
        if text_token_counter is not None
        else len(reasoning.split())
    )
    if count <= 256:
        return "short"
    if count <= 1024:
        return "medium"
    return "long"


def semantic_segments(
    messages: Sequence[dict[str, str]],
    token_counter: Callable[[Sequence[dict[str, str]]], int],
    max_tokens: int,
) -> list[list[dict[str, str]]]:
    if token_counter(messages) <= max_tokens:
        return [list(messages)]

    system = [message for message in messages[:1] if message["role"] == "system"]
    first_user = next(
        (message for message in messages if message["role"] == "user"), None
    )
    if first_user is None:
        return []
    base = system + [first_user]
    remainder = list(messages[messages.index(first_user) + 1 :])
    segments: list[list[dict[str, str]]] = []
    current = list(base)

    for message in remainder:
        candidate = current + [message]
        if token_counter(candidate) <= max_tokens:
            current = candidate
            continue
        if any(item["role"] == "assistant" for item in current):
            segments.append(current)
        current = list(base)
        if token_counter(current + [message]) <= max_tokens:
            current.append(message)

    if any(item["role"] == "assistant" for item in current):
        segments.append(current)
    return segments


def _has_patch(row: dict[str, Any], messages: Sequence[dict[str, str]]) -> bool:
    raw_patch = _first(row, ("gitdiff", "patch", "diff"))
    if raw_patch and _text(raw_patch).strip():
        return True
    return bool(DIFF_RE.search("\n".join(item["content"] for item in messages)))


def build_candidates(
    row: dict[str, Any],
    *,
    row_index: int,
    source: dict[str, Any],
    system_prompt: str,
    data_config: dict[str, Any],
    token_counter: Callable[[Sequence[dict[str, str]]], int],
    text_token_counter: Callable[[str], int] | None = None,
    successful_only: bool = False,
    holdout_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    if source.get("reject_empty_response") and row.get("has_empty_response") is True:
        return [], "empty_response"

    messages = normalize_messages(row, system_prompt)
    if not messages:
        return [], "missing_messages"
    if contains_secret(messages):
        return [], "secret"
    messages = sanitize_messages(
        messages, int(data_config.get("max_tool_output_chars", 12000))
    )
    if not any(message["role"] == "user" for message in messages):
        return [], "missing_user"
    if not any(message["role"] == "assistant" for message in messages):
        return [], "missing_assistant"
    if sum(message["role"] == "tool" for message in messages) > int(
        data_config.get("max_tool_messages", 48)
    ):
        return [], "tool_loop"
    if _max_identical_run(messages) > int(
        data_config.get("max_repeated_message_run", 2)
    ):
        return [], "repeated_messages"

    combined = "\n".join(message["content"] for message in messages)
    lowered = combined.lower()
    if any(
        marker.lower() in lowered
        for marker in data_config.get("blocked_benchmark_markers", [])
    ):
        return [], "benchmark_marker"

    source_name = str(source["name"])
    source_key = ":".join(
        (
            source_name,
            str(source.get("config_name", "default")),
            str(source.get("split", "train")),
        )
    )
    source_id = _row_id(row, row_index)
    identifiers = {
        source_id,
        *(
            str(row[key])
            for key in ("question_id", "instance_id", "session_id", "repository_id")
            if row.get(key) not in (None, "")
        ),
    }
    if holdout_ids and identifiers & holdout_ids:
        return [], "evaluation_holdout"
    if source.get("require_patch") and not _has_patch(row, messages):
        return [], "missing_patch"

    verified = _row_success(row) or bool(source.get("trusted_curated", False))
    if successful_only and not verified:
        return [], "not_verified"

    segments = semantic_segments(
        messages, token_counter, int(data_config["max_seq_length"])
    )
    if not segments:
        return [], "too_long"

    candidates: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(segments):
        num_tokens = int(token_counter(segment))
        if num_tokens < int(data_config.get("min_tokens", 32)):
            continue
        if num_tokens > int(data_config["max_seq_length"]):
            continue
        assistant_chars = sum(
            len(message["content"])
            for message in segment
            if message["role"] == "assistant"
        )
        user_chars = sum(
            len(message["content"])
            for message in segment
            if message["role"] == "user"
        )
        if user_chars < 240 and assistant_chars > 24000:
            continue
        fingerprint = _stable_hash(
            *(_normalized_message_marker(message) for message in segment)
        )
        candidates.append(
            {
                "id": _stable_hash(source_key, source_id, segment_index, fingerprint),
                "group_id": _group_id(row, source_key, source_id),
                "source": source_name,
                "source_config": str(source.get("config_name", "default")),
                "source_split": str(source.get("split", "train")),
                "source_id": source_id,
                "source_revision": str(source.get("revision", "main")),
                "license": str(source.get("license", "unknown")),
                "category": str(source["category"]),
                "reasoning_band": reasoning_band(segment, text_token_counter),
                "verified": verified,
                "num_tokens": num_tokens,
                "fingerprint": fingerprint,
                "messages": segment,
            }
        )
    return candidates, None if candidates else "length_filter"


def load_holdout_ids(path: str | Path | None) -> set[str]:
    if not path:
        return set()
    target = Path(path)
    if not target.exists():
        return set()
    return {
        line.strip()
        for line in target.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def deduplicate(examples: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for example in examples:
        fingerprint = str(example["fingerprint"])
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique.append(example)
    return unique


def _ordered(
    examples: Iterable[dict[str, Any]], seed: int, namespace: str
) -> list[dict[str, Any]]:
    return sorted(
        examples,
        key=lambda example: _stable_hash(seed, namespace, example["id"]),
    )


def _take_until(
    pool: Sequence[dict[str, Any]],
    budget: int,
    selected_ids: set[str],
) -> tuple[list[dict[str, Any]], int]:
    chosen: list[dict[str, Any]] = []
    tokens = 0
    for example in pool:
        if example["id"] in selected_ids:
            continue
        if tokens >= budget:
            break
        chosen.append(example)
        selected_ids.add(str(example["id"]))
        tokens += int(example["num_tokens"])
    return chosen, tokens


def mix_by_tokens(
    examples: Iterable[dict[str, Any]],
    *,
    token_budget: int,
    category_weights: dict[str, float],
    reasoning_weights: dict[str, float],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    unique = deduplicate(examples)
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for example in unique:
        buckets[(example["category"], example["reasoning_band"])].append(example)
    for key, values in list(buckets.items()):
        buckets[key] = _ordered(values, seed, f"{key[0]}:{key[1]}")

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    category_tokens: Counter[str] = Counter()

    for category, category_weight in category_weights.items():
        for band, reasoning_weight in reasoning_weights.items():
            target = round(token_budget * category_weight * reasoning_weight)
            chosen, tokens = _take_until(
                buckets.get((category, band), []), target, selected_ids
            )
            selected.extend(chosen)
            category_tokens[category] += tokens

    for category, category_weight in category_weights.items():
        target = round(token_budget * category_weight)
        deficit = max(0, target - category_tokens[category])
        if not deficit:
            continue
        pool = _ordered(
            (
                example
                for example in unique
                if example["category"] == category
                and example["id"] not in selected_ids
            ),
            seed,
            f"{category}:spill",
        )
        chosen, tokens = _take_until(pool, deficit, selected_ids)
        selected.extend(chosen)
        category_tokens[category] += tokens

    total_tokens = sum(int(example["num_tokens"]) for example in selected)
    if total_tokens < token_budget:
        pool = _ordered(
            (example for example in unique if example["id"] not in selected_ids),
            seed,
            "global-spill",
        )
        chosen, _ = _take_until(pool, token_budget - total_tokens, selected_ids)
        selected.extend(chosen)

    selected = _ordered(selected, seed, "final-shuffle")
    actual_total = sum(int(example["num_tokens"]) for example in selected)
    actual_categories: Counter[str] = Counter()
    actual_reasoning: Counter[str] = Counter()
    for example in selected:
        actual_categories[str(example["category"])] += int(example["num_tokens"])
        actual_reasoning[str(example["reasoning_band"])] += int(
            example["num_tokens"]
        )

    def distribution(counter: Counter[str]) -> dict[str, dict[str, float | int]]:
        return {
            key: {
                "tokens": value,
                "fraction": round(value / actual_total, 6) if actual_total else 0.0,
            }
            for key, value in sorted(counter.items())
        }

    report = {
        "requested_tokens": int(token_budget),
        "selected_tokens": actual_total,
        "selected_examples": len(selected),
        "available_unique_examples": len(unique),
        "budget_reached": actual_total >= token_budget,
        "category_distribution": distribution(actual_categories),
        "reasoning_distribution": distribution(actual_reasoning),
    }
    return selected, report


def split_by_group(
    examples: Sequence[dict[str, Any]],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction deve estar no intervalo [0, 1).")
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    threshold = int(validation_fraction * 10_000)
    for example in examples:
        bucket = int(_stable_hash(seed, example["group_id"])[:8], 16) % 10_000
        (validation if bucket < threshold else train).append(example)

    if validation_fraction and len(examples) > 1 and not validation:
        candidate_group = examples[-1]["group_id"]
        validation = [
            example for example in train if example["group_id"] == candidate_group
        ]
        train = [
            example for example in train if example["group_id"] != candidate_group
        ]
    if not train and validation:
        candidate_group = validation[0]["group_id"]
        train = [
            example
            for example in validation
            if example["group_id"] == candidate_group
        ]
        validation = [
            example
            for example in validation
            if example["group_id"] != candidate_group
        ]
    return train, validation


def training_row(example: dict[str, Any]) -> dict[str, Any]:
    row = {
        "messages": example["messages"],
        "id": example["id"],
        "group_id": example["group_id"],
        "source": example["source"],
        "category": example["category"],
        "reasoning_band": example["reasoning_band"],
        "verified": example["verified"],
        "num_tokens": example["num_tokens"],
    }
    if "source_config" in example:
        row["source_config"] = example["source_config"]
    if "source_split" in example:
        row["source_split"] = example["source_split"]
    return row
