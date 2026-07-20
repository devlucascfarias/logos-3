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


def _assistant_quality_rejection(
    messages: Sequence[dict[str, str]], data_config: dict[str, Any]
) -> str | None:
    min_line_chars = int(data_config.get("repeated_line_min_chars", 12))
    max_line_occurrences = int(
        data_config.get("max_repeated_line_occurrences", 3)
    )
    ngram_size = int(data_config.get("repeated_ngram_size", 8))
    max_ngram_occurrences = int(
        data_config.get("max_repeated_ngram_occurrences", 6)
    )
    reject_unclosed = bool(data_config.get("reject_unclosed_blocks", True))

    for message in messages:
        if message["role"] != "assistant":
            continue
        content = message["content"]
        if reject_unclosed:
            if content.count("```") % 2:
                return "unclosed_block"
            for tag in ("think", "tool_call"):
                if content.lower().count(f"<{tag}>") != content.lower().count(
                    f"</{tag}>"
                ):
                    return "unclosed_block"

        lines = [
            re.sub(r"\s+", " ", line).strip().lower()
            for line in content.splitlines()
            if len(re.sub(r"\s+", " ", line).strip()) >= min_line_chars
        ]
        if lines and max(Counter(lines).values()) > max_line_occurrences:
            return "repeated_lines"

        tokens = re.findall(r"\w+|[^\w\s]", content.lower(), flags=re.UNICODE)
        if ngram_size > 0 and len(tokens) >= ngram_size:
            ngrams = Counter(
                tuple(tokens[index : index + ngram_size])
                for index in range(len(tokens) - ngram_size + 1)
            )
            if ngrams and max(ngrams.values()) > max_ngram_occurrences:
                return "repeated_ngram"
    return None


def _near_fingerprint(messages: Sequence[dict[str, str]]) -> str:
    markers: list[str] = []
    for message in messages:
        content = re.sub(r"\b\d+(?:\.\d+)?\b", "<num>", message["content"])
        content = re.sub(r"\s+", " ", content).strip().lower()
        markers.append(f"{message['role']}:{content}")
    return _stable_hash(*markers)


def _without_thinking(
    messages: Sequence[dict[str, str]],
) -> list[dict[str, str]]:
    converted: list[dict[str, str]] = []
    for message in messages:
        content = message["content"]
        if message["role"] == "assistant":
            content = THINK_RE.sub("", content).strip()
        if content:
            converted.append({"role": message["role"], "content": content})
    return converted


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
    quality_rejection = _assistant_quality_rejection(messages, data_config)
    if quality_rejection:
        return [], quality_rejection

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
        category = str(source["category"])
        direct_fractions = data_config.get(
            "direct_conversion_fraction_by_category", {}
        )
        direct_fraction = float(direct_fractions.get(category, 0.0))
        conversion_key = _stable_hash(source_key, source_id, segment_index)
        conversion_value = int(conversion_key[:8], 16) / 0xFFFFFFFF
        if direct_fraction and conversion_value < direct_fraction:
            converted = _without_thinking(segment)
            if converted != segment and any(
                message["role"] == "assistant" for message in converted
            ):
                segment = converted

        quality_rejection = _assistant_quality_rejection(segment, data_config)
        if quality_rejection:
            continue
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
        if assistant_chars > int(
            data_config.get("max_assistant_chars", 20000)
        ):
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
                "category": category,
                "reasoning_band": reasoning_band(segment, text_token_counter),
                "verified": verified,
                "num_tokens": num_tokens,
                "fingerprint": fingerprint,
                "near_fingerprint": _near_fingerprint(segment),
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
    seen_near: set[str] = set()
    unique: list[dict[str, Any]] = []
    for example in examples:
        fingerprint = str(example["fingerprint"])
        if fingerprint in seen:
            continue
        near_fingerprint = str(example.get("near_fingerprint", fingerprint))
        if near_fingerprint in seen_near:
            continue
        seen.add(fingerprint)
        seen_near.add(near_fingerprint)
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


def _allocate_token_matrix(
    buckets: dict[tuple[str, str], list[dict[str, Any]]],
    category_targets: dict[str, int],
    reasoning_targets: dict[str, int],
) -> tuple[dict[tuple[str, str], int], int]:
    source = ("source", "")
    sink = ("sink", "")
    residual: dict[tuple[tuple[str, str], tuple[str, str]], int] = {}
    neighbors: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)

    def add_edge(
        start: tuple[str, str], end: tuple[str, str], capacity: int
    ) -> None:
        residual[(start, end)] = capacity
        residual[(end, start)] = 0
        neighbors[start].append(end)
        neighbors[end].append(start)

    for category, target in category_targets.items():
        category_node = ("category", category)
        add_edge(source, category_node, target)
        for band in reasoning_targets:
            available = sum(
                int(example["num_tokens"])
                for example in buckets.get((category, band), [])
            )
            if available:
                add_edge(category_node, ("band", band), available)
    for band, target in reasoning_targets.items():
        add_edge(("band", band), sink, target)

    total_flow = 0
    while True:
        parent: dict[tuple[str, str], tuple[str, str] | None] = {source: None}
        queue = [source]
        queue_index = 0
        while queue_index < len(queue) and sink not in parent:
            node = queue[queue_index]
            queue_index += 1
            for neighbor in neighbors[node]:
                if neighbor in parent or residual[(node, neighbor)] <= 0:
                    continue
                parent[neighbor] = node
                queue.append(neighbor)
        if sink not in parent:
            break

        path_capacity = sum(category_targets.values())
        node = sink
        while parent[node] is not None:
            previous = parent[node]
            path_capacity = min(path_capacity, residual[(previous, node)])
            node = previous
        node = sink
        while parent[node] is not None:
            previous = parent[node]
            residual[(previous, node)] -= path_capacity
            residual[(node, previous)] += path_capacity
            node = previous
        total_flow += path_capacity

    allocations: dict[tuple[str, str], int] = {}
    for category in category_targets:
        category_node = ("category", category)
        for band in reasoning_targets:
            band_node = ("band", band)
            if (category_node, band_node) not in residual:
                continue
            available = sum(
                int(example["num_tokens"])
                for example in buckets.get((category, band), [])
            )
            flow = available - residual[(category_node, band_node)]
            if flow:
                allocations[(category, band)] = flow
    return allocations, total_flow


def mix_by_tokens(
    examples: Iterable[dict[str, Any]],
    *,
    token_budget: int,
    category_weights: dict[str, float],
    reasoning_weights: dict[str, float],
    seed: int,
    max_examples_per_group: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    unique = deduplicate(examples)
    if max_examples_per_group > 0:
        limited: list[dict[str, Any]] = []
        group_counts: Counter[str] = Counter()
        for example in _ordered(unique, seed, "group-limit"):
            group_id = str(example["group_id"])
            if group_counts[group_id] >= max_examples_per_group:
                continue
            limited.append(example)
            group_counts[group_id] += 1
        unique = limited
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for example in unique:
        buckets[(example["category"], example["reasoning_band"])].append(example)
    for key, values in list(buckets.items()):
        buckets[key] = _ordered(values, seed, f"{key[0]}:{key[1]}")

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    category_tokens: Counter[str] = Counter()
    reasoning_tokens: Counter[str] = Counter()

    def add(chosen: Sequence[dict[str, Any]]) -> None:
        selected.extend(chosen)
        for example in chosen:
            tokens = int(example["num_tokens"])
            category_tokens[str(example["category"])] += tokens
            reasoning_tokens[str(example["reasoning_band"])] += tokens

    category_targets = {
        category: round(token_budget * weight)
        for category, weight in category_weights.items()
    }
    reasoning_targets = {
        band: round(token_budget * weight)
        for band, weight in reasoning_weights.items()
    }

    allocations, maximum_feasible_tokens = _allocate_token_matrix(
        buckets, category_targets, reasoning_targets
    )
    for (category, band), target in allocations.items():
        chosen, _ = _take_until(
            buckets[(category, band)], target, selected_ids
        )
        add(chosen)

    # Corrige pequenos déficits causados pela granularidade dos exemplos sem
    # permitir spill para uma categoria ou faixa que já atingiu a própria meta.
    category_order = sorted(
        category_weights,
        key=lambda category: (
            sum(
                bool(buckets.get((category, band)))
                for band in reasoning_weights
            ),
            list(category_weights).index(category),
        ),
    )
    for category in category_order:
        category_deficit = max(
            0, category_targets[category] - category_tokens[category]
        )
        while category_deficit > 0:
            available_bands = [
                band
                for band in reasoning_weights
                if buckets.get((category, band))
                and any(
                    str(example["id"]) not in selected_ids
                    for example in buckets[(category, band)]
                )
                and reasoning_targets[band] - reasoning_tokens[band] > 0
            ]
            if not available_bands:
                break
            band = max(
                available_bands,
                key=lambda name: (
                    (
                        reasoning_targets[name] - reasoning_tokens[name]
                    )
                    / max(reasoning_targets[name], 1),
                    reasoning_targets[name] - reasoning_tokens[name],
                    reasoning_weights[name],
                ),
            )
            reasoning_deficit = max(
                0, reasoning_targets[band] - reasoning_tokens[band]
            )
            chosen, _ = _take_until(
                buckets.get((category, band), []),
                min(category_deficit, reasoning_deficit),
                selected_ids,
            )
            add(chosen)
            category_deficit = max(
                0, category_targets[category] - category_tokens[category]
            )

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

    def matrix(
        values: Sequence[dict[str, Any]],
    ) -> dict[str, dict[str, dict[str, int]]]:
        result = {
            category: {
                band: {"examples": 0, "tokens": 0}
                for band in reasoning_weights
            }
            for category in category_weights
        }
        for example in values:
            category = str(example["category"])
            band = str(example["reasoning_band"])
            if category not in result or band not in result[category]:
                continue
            result[category][band]["examples"] += 1
            result[category][band]["tokens"] += int(example["num_tokens"])
        return result

    actual_category_distribution = distribution(actual_categories)
    actual_reasoning_distribution = distribution(actual_reasoning)
    category_deviation = {
        category: round(
            float(
                actual_category_distribution.get(category, {}).get(
                    "fraction", 0.0
                )
            )
            - weight,
            6,
        )
        for category, weight in category_weights.items()
    }
    reasoning_deviation = {
        band: round(
            float(actual_reasoning_distribution.get(band, {}).get("fraction", 0.0))
            - weight,
            6,
        )
        for band, weight in reasoning_weights.items()
    }
    report = {
        "requested_tokens": int(token_budget),
        "selected_tokens": actual_total,
        "budget_fraction": round(actual_total / token_budget, 6),
        "selected_examples": len(selected),
        "available_unique_examples": len(unique),
        "available_unique_groups": len(
            {str(example["group_id"]) for example in unique}
        ),
        "budget_reached": actual_total >= token_budget,
        "maximum_feasible_tokens": maximum_feasible_tokens,
        "matrix_feasible": maximum_feasible_tokens >= token_budget,
        "category_distribution": actual_category_distribution,
        "category_targets": {
            category: {
                "tokens": category_targets[category],
                "fraction": float(weight),
            }
            for category, weight in category_weights.items()
        },
        "category_deviation": category_deviation,
        "max_category_deviation": max(
            (abs(value) for value in category_deviation.values()),
            default=0.0,
        ),
        "reasoning_distribution": actual_reasoning_distribution,
        "reasoning_targets": {
            band: {
                "tokens": reasoning_targets[band],
                "fraction": float(weight),
            }
            for band, weight in reasoning_weights.items()
        },
        "reasoning_deviation": reasoning_deviation,
        "max_reasoning_deviation": max(
            (abs(value) for value in reasoning_deviation.values()),
            default=0.0,
        ),
        "candidate_matrix": matrix(unique),
        "selected_matrix": matrix(selected),
    }
    return selected, report


def split_by_group(
    examples: Sequence[dict[str, Any]],
    *,
    validation_fraction: float,
    seed: int,
    min_validation_examples: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction deve estar no intervalo [0, 1).")
    if min_validation_examples < 0:
        raise ValueError("min_validation_examples não pode ser negativo.")
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    threshold = int(validation_fraction * 10_000)
    for example in examples:
        bucket = int(_stable_hash(seed, example["group_id"])[:8], 16) % 10_000
        (validation if bucket < threshold else train).append(example)

    target_min = min(min_validation_examples, max(0, len(examples) - 1))
    if validation_fraction and len(examples) > 1:
        target_min = max(target_min, 1)
    if len(validation) < target_min and train:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for example in train:
            groups[str(example["group_id"])].append(example)
        ordered_groups = sorted(
            groups,
            key=lambda group_id: _stable_hash(
                seed, "validation-minimum", group_id
            ),
        )
        for group_id in ordered_groups:
            group = groups[group_id]
            if len(train) == len(group):
                continue
            validation.extend(group)
            group_ids = {str(example["id"]) for example in group}
            train = [
                example
                for example in train
                if str(example["id"]) not in group_ids
            ]
            if len(validation) >= target_min:
                break
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
