from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from typing import Any, Iterable

from .schemas import CanonicalExample, Message, ReasoningMode, TargetType


IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
ASSISTANT_BLOCK_RE = re.compile(
    rf"{re.escape(IM_START)}assistant\n(.*?){re.escape(IM_END)}",
    re.DOTALL,
)
DEFAULT_REASONING_MODE_WEIGHTS = {
    ReasoningMode.LONG.value: 0.30,
    ReasoningMode.COMPRESSED.value: 0.50,
    ReasoningMode.HIDDEN.value: 0.20,
}


def render_message(message: Message | dict[str, Any]) -> str:
    item = message if isinstance(message, Message) else Message.from_dict(message)
    role = item.role if item.role != "tool" else f"tool {item.name or 'unknown'}"
    return f"{IM_START}{role}\n{item.content}{IM_END}\n"


def render_messages(
    messages: Iterable[Message | dict[str, Any]],
    add_generation_prompt: bool = False,
) -> str:
    rendered = "".join(render_message(message) for message in messages)
    if add_generation_prompt:
        rendered += f"{IM_START}assistant\n"
    return rendered


def flatten_for_training(example: CanonicalExample | dict[str, Any]) -> dict[str, Any]:
    canonical = (
        example if isinstance(example, CanonicalExample) else CanonicalExample.from_dict(example)
    )
    last_assistant = max(
        index for index, message in enumerate(canonical.messages) if message.role == "assistant"
    )
    target_indices = [
        int(index)
        for index in canonical.metadata.get("loss_assistant_indices", [last_assistant])
    ]
    first_target = min(target_indices)
    prompt_messages = canonical.messages[:first_target]
    target_messages = canonical.messages[first_target:]
    if target_indices == [last_assistant] and first_target == len(canonical.messages) - 1:
        prompt = render_messages(prompt_messages, add_generation_prompt=True)
        completion = target_messages[0].content
    else:
        prompt = render_messages(prompt_messages)
        completion = render_messages(target_messages)
    metadata = {
        **canonical.metadata,
        "example_id": canonical.example_id,
        "session_id": canonical.session_id,
        "repository_id": canonical.repository_id,
        "source_dataset": canonical.provenance.source_dataset,
        "target_type": canonical.target_type,
        "reasoning_mode": canonical.reasoning_mode,
        "quality_score": canonical.quality_score,
    }
    value = canonical.to_dict()
    value.update(
        {
            "prompt": prompt,
            "completion": completion,
            "prompt_messages": [message.to_dict() for message in prompt_messages],
            "completion_messages": [message.to_dict() for message in target_messages],
            "assistant_loss_mask": "derived_at_tokenization",
            "assistant_target_indices": target_indices,
            "enable_thinking": reasoning_mode_uses_thinking(canonical.reasoning_mode),
            "metadata": metadata,
        }
    )
    return value


def reasoning_mode_uses_thinking(mode: str) -> bool:
    if mode not in {item.value for item in ReasoningMode}:
        raise ValueError(f"Unknown reasoning mode: {mode}")
    return mode != ReasoningMode.HIDDEN.value


def _template_messages(
    messages: Iterable[Message | dict[str, Any]],
    *,
    assistant_target_indices: set[int] | None = None,
    reasoning_mode: str | None = None,
) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for index, raw_message in enumerate(messages):
        message = (
            raw_message
            if isinstance(raw_message, Message)
            else Message.from_dict(raw_message)
        )
        value = message.to_dict()
        if (
            reasoning_mode == ReasoningMode.HIDDEN.value
            and assistant_target_indices is not None
            and index in assistant_target_indices
            and message.role == "assistant"
        ):
            direct_content = hide_reasoning(message.content)
            value["content"] = f"<think>\n\n</think>\n\n{direct_content}"
        values.append(value)
    return values


def render_with_chat_template(
    tokenizer: Any,
    messages: Iterable[Message | dict[str, Any]],
    *,
    add_generation_prompt: bool = False,
    reasoning_mode: str = ReasoningMode.COMPRESSED.value,
    assistant_target_indices: Iterable[int] | None = None,
) -> str:
    """Render messages with the tokenizer-owned Qwen chat template.

    Hidden targets receive Qwen3's empty thinking prefix during training. At
    generation time the official template injects that prefix through
    ``enable_thinking=False``.
    """

    targets = (
        set(int(index) for index in assistant_target_indices)
        if assistant_target_indices is not None
        else None
    )
    prepared = _template_messages(
        messages,
        assistant_target_indices=targets,
        reasoning_mode=reasoning_mode,
    )
    rendered = tokenizer.apply_chat_template(
        prepared,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=reasoning_mode_uses_thinking(reasoning_mode),
    )
    if not isinstance(rendered, str):
        raise TypeError("tokenizer.apply_chat_template(..., tokenize=False) must return text")
    return rendered


def generation_messages_from_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    prompt_messages = row.get("prompt_messages")
    if isinstance(prompt_messages, list) and prompt_messages:
        return [Message.from_dict(item).to_dict() for item in prompt_messages]
    raw_messages = row.get("messages")
    if isinstance(raw_messages, list) and raw_messages:
        messages = [Message.from_dict(item) for item in raw_messages]
        raw_targets = row.get("assistant_target_indices")
        if isinstance(raw_targets, list) and raw_targets:
            messages = messages[: min(int(index) for index in raw_targets)]
        return [message.to_dict() for message in messages]
    prompt = str(row.get("prompt") or row.get("instruction") or "").strip()
    if not prompt:
        raise ValueError("Generation row needs prompt_messages, messages, prompt, or instruction")
    return [{"role": "user", "content": prompt}]


def render_generation_prompt(
    tokenizer: Any,
    row: dict[str, Any],
    *,
    default_reasoning_mode: str = ReasoningMode.COMPRESSED.value,
) -> tuple[str, str]:
    mode = str(row.get("reasoning_mode") or default_reasoning_mode)
    if mode == "thinking":
        mode = ReasoningMode.COMPRESSED.value
    elif mode in {"non-thinking", "non_thinking"}:
        mode = ReasoningMode.HIDDEN.value
    prompt = render_with_chat_template(
        tokenizer,
        generation_messages_from_row(row),
        add_generation_prompt=True,
        reasoning_mode=mode,
    )
    return prompt, mode


def _target_content_spans(
    rendered: str,
    messages: list[Message],
    target_indices: set[int],
    reasoning_mode: str,
) -> list[tuple[int, int]]:
    assistant_indices = [
        index for index, message in enumerate(messages) if message.role == "assistant"
    ]
    blocks = list(ASSISTANT_BLOCK_RE.finditer(rendered))
    if len(blocks) != len(assistant_indices):
        raise ValueError(
            "Qwen chat template produced an unexpected number of assistant blocks: "
            f"{len(blocks)} for {len(assistant_indices)} assistant messages"
        )
    spans: list[tuple[int, int]] = []
    for message_index, block in zip(assistant_indices, blocks):
        if message_index not in target_indices:
            continue
        start, end = block.span(1)
        if reasoning_mode == ReasoningMode.HIDDEN.value:
            closing = rendered.find("</think>", start, end)
            if closing >= 0:
                start = closing + len("</think>")
                while start < end and rendered[start] in "\r\n":
                    start += 1
        if start < end:
            spans.append((start, end))
    return spans


def tokenize_with_assistant_mask(
    tokenizer: Any,
    messages: Iterable[Message | dict[str, Any]],
    max_length: int,
    assistant_target_indices: Iterable[int] | None = None,
    reasoning_mode: str = ReasoningMode.COMPRESSED.value,
) -> dict[str, list[int]]:
    """Apply the official template and label only selected assistant content."""

    items = [
        raw_message if isinstance(raw_message, Message) else Message.from_dict(raw_message)
        for raw_message in messages
    ]
    targets = (
        set(int(index) for index in assistant_target_indices)
        if assistant_target_indices is not None
        else {index for index, message in enumerate(items) if message.role == "assistant"}
    )
    rendered = render_with_chat_template(
        tokenizer,
        items,
        reasoning_mode=reasoning_mode,
        assistant_target_indices=targets,
    )
    spans = _target_content_spans(rendered, items, targets, reasoning_mode)
    try:
        encoded = tokenizer(
            rendered,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except (NotImplementedError, TypeError) as exc:
        raise RuntimeError(
            "Assistant-only masking requires a fast tokenizer with offset mappings"
        ) from exc
    input_ids = list(encoded["input_ids"])
    offsets = list(encoded.get("offset_mapping") or [])
    if len(offsets) != len(input_ids):
        raise RuntimeError("Tokenizer did not return one offset mapping per token")
    labels = [-100] * len(input_ids)
    for token_index, (token_start, token_end) in enumerate(offsets):
        if token_end <= token_start:
            continue
        if any(token_start < span_end and token_end > span_start for span_start, span_end in spans):
            labels[token_index] = input_ids[token_index]

    if len(input_ids) > max_length:
        input_ids = input_ids[-max_length:]
        labels = labels[-max_length:]
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def normalized_prompt_hash(messages: Iterable[Message | dict[str, Any]]) -> str:
    prompt_parts: list[str] = []
    for raw_message in messages:
        message = (
            raw_message
            if isinstance(raw_message, Message)
            else Message.from_dict(raw_message)
        )
        if message.role == "assistant":
            break
        normalized = re.sub(r"\s+", " ", message.content).strip().lower()
        prompt_parts.append(f"{message.role}:{normalized}")
    return hashlib.sha256("\n".join(prompt_parts).encode("utf-8")).hexdigest()


def compress_reasoning(content: str) -> str:
    think_match = re.search(
        r"<(?:think|plan)>(.*?)</(?:think|plan)>",
        content,
        re.DOTALL | re.IGNORECASE,
    )
    if not think_match:
        return content
    raw = think_match.group(1)
    sentences = [
        sentence.strip(" -\n\t")
        for sentence in re.split(r"(?:\n+|(?<=[.!?])\s+)", raw)
        if sentence.strip()
    ]
    selected: list[str] = []
    keywords = re.compile(
        r"\b(inspect|read|search|reproduce|test|build|fix|patch|verify|check|run|locate)\b",
        re.IGNORECASE,
    )
    for sentence in sentences:
        if keywords.search(sentence) and sentence not in selected:
            selected.append(sentence)
        if len(selected) == 4:
            break
    if not selected:
        selected = sentences[:3]
    plan = "<think>\n" + "\n".join(
        f"{index}. {sentence}" for index, sentence in enumerate(selected, 1)
    ) + "\n</think>"
    return content[: think_match.start()] + plan + content[think_match.end() :]


def hide_reasoning(content: str) -> str:
    return re.sub(
        r"<(?:think|plan)>.*?</(?:think|plan)>\s*",
        "",
        content,
        flags=re.DOTALL | re.IGNORECASE,
    ).lstrip()


def apply_reasoning_mode(content: str, mode: str) -> str:
    if mode == ReasoningMode.LONG.value:
        return re.sub(r"<(/?)plan>", r"<\1think>", content, flags=re.IGNORECASE)
    if mode == ReasoningMode.COMPRESSED.value:
        return compress_reasoning(content)
    if mode == ReasoningMode.HIDDEN.value:
        return hide_reasoning(content)
    raise ValueError(f"Unknown reasoning mode: {mode}")


def _select_reasoning_mode(
    session_id: str,
    turn_number: int,
    weights: dict[str, float],
) -> str:
    valid = {item.value for item in ReasoningMode}
    normalized = {
        str(mode): max(0.0, float(weight))
        for mode, weight in weights.items()
        if str(mode) in valid
    }
    total = sum(normalized.values())
    if total <= 0:
        raise ValueError("At least one reasoning mode must have a positive weight")
    digest = hashlib.sha256(f"{session_id}:{turn_number}".encode("utf-8")).digest()
    position = int.from_bytes(digest[:8], "big") / 2**64
    cumulative = 0.0
    for mode in (
        ReasoningMode.LONG.value,
        ReasoningMode.COMPRESSED.value,
        ReasoningMode.HIDDEN.value,
    ):
        cumulative += normalized.get(mode, 0.0) / total
        if position < cumulative:
            return mode
    return ReasoningMode.HIDDEN.value


def _effective_reasoning_mode(content: str, requested_mode: str) -> str:
    if requested_mode == ReasoningMode.HIDDEN.value:
        return requested_mode
    if re.search(r"<think>.*?</think>", content, re.DOTALL | re.IGNORECASE):
        return requested_mode
    return ReasoningMode.HIDDEN.value


def build_training_examples(
    trajectory: CanonicalExample,
    *,
    max_chunk_assistant_turns: int = 2,
    reasoning_mode_weights: dict[str, float] | None = None,
) -> list[CanonicalExample]:
    """Expand one clean trajectory into next-action, chunk and final-answer examples."""

    assistant_indices = [
        index for index, message in enumerate(trajectory.messages) if message.role == "assistant"
    ]
    if not assistant_indices:
        return []
    examples: list[CanonicalExample] = []
    mode_weights = reasoning_mode_weights or DEFAULT_REASONING_MODE_WEIGHTS
    for turn_number, assistant_index in enumerate(assistant_indices):
        requested_mode = _select_reasoning_mode(
            trajectory.session_id,
            turn_number,
            mode_weights,
        )
        transformed_content = apply_reasoning_mode(
            trajectory.messages[assistant_index].content,
            requested_mode,
        )
        mode = _effective_reasoning_mode(transformed_content, requested_mode)
        if mode == ReasoningMode.HIDDEN.value:
            transformed_content = hide_reasoning(transformed_content)
        messages = [
            replace(message, content=transformed_content)
            if index == assistant_index
            else message
            for index, message in enumerate(trajectory.messages[: assistant_index + 1])
        ]
        examples.append(
            replace(
                trajectory,
                messages=messages,
                target_type=TargetType.NEXT_ACTION.value,
                reasoning_mode=mode,
                example_id="",
                metadata={**trajectory.metadata, "assistant_turn": turn_number},
            )
        )
        examples[-1].metadata["loss_assistant_indices"] = [assistant_index]

    for start_turn in range(0, len(assistant_indices), max_chunk_assistant_turns):
        end_turn = min(start_turn + max_chunk_assistant_turns - 1, len(assistant_indices) - 1)
        start_index = 0 if start_turn == 0 else assistant_indices[start_turn - 1] + 1
        end_index = (
            assistant_indices[end_turn + 1]
            if end_turn + 1 < len(assistant_indices)
            else len(trajectory.messages)
        )
        prefix = trajectory.messages[:start_index]
        chunk = trajectory.messages[start_index:end_index]
        if prefix and not any(message.role == "user" for message in prefix):
            prefix = trajectory.messages[: assistant_indices[start_turn]]
        transformed_chunk: list[Message] = []
        target_original_indices = set(assistant_indices[start_turn : end_turn + 1])
        target_thinking_flags: list[bool] = []
        for index, message in enumerate(chunk):
            if (start_index + index) in target_original_indices:
                transformed = apply_reasoning_mode(
                    message.content,
                    ReasoningMode.COMPRESSED.value,
                )
                target_thinking_flags.append(
                    bool(
                        re.search(
                            r"<think>.*?</think>",
                            transformed,
                            re.DOTALL | re.IGNORECASE,
                        )
                    )
                )
                transformed_chunk.append(replace(message, content=transformed))
            else:
                transformed_chunk.append(message)
        chunk_mode = (
            ReasoningMode.COMPRESSED.value
            if target_thinking_flags and all(target_thinking_flags)
            else ReasoningMode.HIDDEN.value
        )
        if chunk_mode == ReasoningMode.HIDDEN.value:
            transformed_chunk = [
                replace(message, content=hide_reasoning(message.content))
                if (start_index + index) in target_original_indices
                else message
                for index, message in enumerate(transformed_chunk)
            ]
        combined = list(prefix) + transformed_chunk
        target_indices = [
            index
            for index, message in enumerate(combined)
            if message.role == "assistant"
            and index >= len(prefix)
        ]
        examples.append(
            replace(
                trajectory,
                messages=combined,
                target_type=TargetType.TRAJECTORY.value,
                reasoning_mode=chunk_mode,
                example_id="",
                metadata={
                    **trajectory.metadata,
                    "chunk_turns": [start_turn, end_turn],
                    "loss_assistant_indices": target_indices,
                },
            )
        )

    final_index = assistant_indices[-1]
    if "<tool_call>" not in trajectory.messages[final_index].content:
        final_messages = [
            replace(message, content=hide_reasoning(message.content))
            if index == final_index
            else message
            for index, message in enumerate(trajectory.messages[: final_index + 1])
        ]
        examples.append(
            replace(
                trajectory,
                messages=final_messages,
                target_type=TargetType.FINAL_ANSWER.value,
                reasoning_mode=ReasoningMode.HIDDEN.value,
                example_id="",
                metadata={
                    **trajectory.metadata,
                    "final_turn": len(assistant_indices) - 1,
                    "loss_assistant_indices": [final_index],
                },
            )
        )
    return examples
