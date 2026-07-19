from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from typing import Any, Iterable

from .schemas import CanonicalExample, Message, ReasoningMode, TargetType


IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


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
            "assistant_loss_mask": "derived_at_tokenization",
            "assistant_target_indices": target_indices,
            "metadata": metadata,
        }
    )
    return value


def tokenize_with_assistant_mask(
    tokenizer: Any,
    messages: Iterable[Message | dict[str, Any]],
    max_length: int,
    assistant_target_indices: Iterable[int] | None = None,
) -> dict[str, list[int]]:
    """Tokenize ChatML while labeling assistant content only.

    Each fragment is encoded separately without extra special tokens, which
    makes the boundary deterministic across fast and slow tokenizers.
    Left truncation preserves the most recent action/answer.
    """

    input_ids: list[int] = []
    labels: list[int] = []
    targets = set(assistant_target_indices) if assistant_target_indices is not None else None
    for message_index, raw_message in enumerate(messages):
        message = raw_message if isinstance(raw_message, Message) else Message.from_dict(raw_message)
        role = message.role if message.role != "tool" else f"tool {message.name or 'unknown'}"
        prefix = f"{IM_START}{role}\n"
        suffix = f"{IM_END}\n"
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        content_ids = tokenizer.encode(message.content, add_special_tokens=False)
        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        input_ids.extend(prefix_ids)
        labels.extend([-100] * len(prefix_ids))
        input_ids.extend(content_ids)
        if message.role == "assistant" and (targets is None or message_index in targets):
            labels.extend(content_ids)
        else:
            labels.extend([-100] * len(content_ids))
        input_ids.extend(suffix_ids)
        labels.extend([-100] * len(suffix_ids))

    if len(input_ids) > max_length:
        input_ids = input_ids[-max_length:]
        labels = labels[-max_length:]
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def normalized_prompt_hash(messages: Iterable[Message | dict[str, Any]]) -> str:
    prompt_parts: list[str] = []
    for raw_message in messages:
        message = raw_message if isinstance(raw_message, Message) else Message.from_dict(raw_message)
        if message.role == "assistant":
            break
        normalized = re.sub(r"\s+", " ", message.content).strip().lower()
        prompt_parts.append(f"{message.role}:{normalized}")
    return hashlib.sha256("\n".join(prompt_parts).encode("utf-8")).hexdigest()


def compress_reasoning(content: str) -> str:
    think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL | re.IGNORECASE)
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
    plan = "<plan>\n" + "\n".join(
        f"{index}. {sentence}" for index, sentence in enumerate(selected, 1)
    ) + "\n</plan>"
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
        return content
    if mode == ReasoningMode.COMPRESSED.value:
        return compress_reasoning(content)
    if mode == ReasoningMode.HIDDEN.value:
        return hide_reasoning(content)
    raise ValueError(f"Unknown reasoning mode: {mode}")


def build_training_examples(
    trajectory: CanonicalExample,
    *,
    max_chunk_assistant_turns: int = 2,
    reasoning_modes: tuple[str, ...] = (
        "long",
        "long",
        "long",
        "long",
        "long",
        "long",
        "compressed",
        "compressed",
        "compressed",
        "hidden",
    ),
) -> list[CanonicalExample]:
    """Expand one clean trajectory into next-action, chunk and final-answer examples."""

    assistant_indices = [
        index for index, message in enumerate(trajectory.messages) if message.role == "assistant"
    ]
    if not assistant_indices:
        return []
    examples: list[CanonicalExample] = []
    mode_offset = int(
        hashlib.sha256(trajectory.session_id.encode("utf-8")).hexdigest()[:8],
        16,
    )
    for turn_number, assistant_index in enumerate(assistant_indices):
        mode = reasoning_modes[(mode_offset + turn_number) % len(reasoning_modes)]
        messages = [
            replace(message, content=apply_reasoning_mode(message.content, mode))
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
        combined = list(prefix) + [
            replace(message, content=apply_reasoning_mode(message.content, "compressed"))
            if (start_index + index) in assistant_indices[start_turn : end_turn + 1]
            else message
            for index, message in enumerate(chunk)
        ]
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
                reasoning_mode=ReasoningMode.COMPRESSED.value,
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
