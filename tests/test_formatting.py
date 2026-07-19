from __future__ import annotations

from fable_distill.formatting import (
    apply_reasoning_mode,
    build_training_examples,
    flatten_for_training,
    render_generation_prompt,
    render_messages,
    render_with_chat_template,
    tokenize_with_assistant_mask,
)
from fable_distill.schemas import CanonicalExample, Message, Provenance


class CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    eos_token = "<eos>"
    pad_token = "<pad>"

    def __init__(self) -> None:
        self.template_calls: list[dict] = []

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(character) + 2 for character in text]

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict:
        value = {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}
        if return_offsets_mapping:
            value["offset_mapping"] = [
                (index, index + 1) for index in range(len(text))
            ]
        return value

    def apply_chat_template(
        self,
        messages: list[dict],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        assert not tokenize
        self.template_calls.append(
            {
                "add_generation_prompt": add_generation_prompt,
                "enable_thinking": enable_thinking,
            }
        )
        rendered = ""
        for message in messages:
            role = message["role"]
            if role == "tool":
                rendered += (
                    "<|im_start|>user\n<tool_response>\n"
                    f"{message['content']}\n</tool_response><|im_end|>\n"
                )
            else:
                rendered += (
                    f"<|im_start|>{role}\n{message['content']}<|im_end|>\n"
                )
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
            if not enable_thinking:
                rendered += "<think>\n\n</think>\n\n"
        return rendered


def example() -> CanonicalExample:
    return CanonicalExample(
        messages=[
            Message("system", "system"),
            Message("user", "question"),
            Message("assistant", "<think>inspect</think>\nanswer"),
        ],
        provenance=Provenance("unit/source"),
        session_id="session-1",
        repository_id="repo-1",
        quality_score=0.8,
    )


def test_chatml_format_and_flatten() -> None:
    value = flatten_for_training(example())
    assert value["prompt"].endswith("<|im_start|>assistant\n")
    assert value["completion"] == "<think>inspect</think>\nanswer"
    assert render_messages(example().messages).count("<|im_start|>") == 3
    assert value["prompt_messages"][-1]["role"] == "user"
    assert value["enable_thinking"]


def test_official_template_controls_thinking_mode() -> None:
    tokenizer = CharacterTokenizer()
    prompt, mode = render_generation_prompt(
        tokenizer,
        {"instruction": "question", "reasoning_mode": "hidden"},
    )
    assert mode == "hidden"
    assert prompt.endswith("<think>\n\n</think>\n\n")
    assert tokenizer.template_calls[-1]["enable_thinking"] is False
    rendered = render_with_chat_template(
        tokenizer,
        example().messages,
        reasoning_mode="compressed",
    )
    assert "<|im_start|>assistant" in rendered
    assert tokenizer.template_calls[-1]["enable_thinking"] is True


def test_loss_mask_labels_only_assistant_content() -> None:
    tokenizer = CharacterTokenizer()
    messages = example().messages
    encoded = tokenize_with_assistant_mask(tokenizer, messages, max_length=10_000)
    labeled = [token for token in encoded["labels"] if token != -100]
    assert labeled == tokenizer.encode(messages[-1].content)
    assert len(encoded["input_ids"]) == len(encoded["labels"])


def test_explicit_target_indices_mask_previous_assistant_turns() -> None:
    tokenizer = CharacterTokenizer()
    messages = [
        Message("user", "first"),
        Message("assistant", "old answer"),
        Message("user", "next"),
        Message("assistant", "new answer"),
    ]
    encoded = tokenize_with_assistant_mask(
        tokenizer,
        messages,
        max_length=10_000,
        assistant_target_indices=[3],
    )
    labeled = [token for token in encoded["labels"] if token != -100]
    assert labeled == tokenizer.encode("new answer")


def test_left_truncation_keeps_latest_target() -> None:
    tokenizer = CharacterTokenizer()
    encoded = tokenize_with_assistant_mask(tokenizer, example().messages, max_length=12)
    assert len(encoded["input_ids"]) == 12
    assert any(label != -100 for label in encoded["labels"])


def test_reasoning_modes() -> None:
    text = "<think>Inspect files. Run tests. Patch bug. Verify tests.</think>\nDone."
    compressed = apply_reasoning_mode(text, "compressed")
    hidden = apply_reasoning_mode(text, "hidden")
    assert "<think>" in compressed and "<plan>" not in compressed
    assert hidden == "Done."


def test_hidden_mode_masks_qwen_empty_thinking_prefix() -> None:
    tokenizer = CharacterTokenizer()
    messages = [Message("user", "question"), Message("assistant", "direct answer")]
    encoded = tokenize_with_assistant_mask(
        tokenizer,
        messages,
        max_length=10_000,
        assistant_target_indices=[1],
        reasoning_mode="hidden",
    )
    labeled = [token for token in encoded["labels"] if token != -100]
    assert labeled == tokenizer.encode("direct answer")


def test_configured_reasoning_weights_are_used() -> None:
    trajectory = CanonicalExample(
        messages=[
            Message("user", "task"),
            Message("assistant", "<think>Inspect files.</think>\nDone"),
        ],
        provenance=Provenance("unit/source"),
        session_id="weighted",
        repository_id="r",
    )
    hidden = build_training_examples(
        trajectory,
        reasoning_mode_weights={"hidden": 1.0},
    )
    next_action = next(item for item in hidden if item.target_type == "next_action")
    assert next_action.reasoning_mode == "hidden"
    assert "<think>" not in next_action.messages[-1].content


def test_built_example_exports_loss_target_indices() -> None:
    trajectory = CanonicalExample(
        messages=[
            Message("user", "task"),
            Message("assistant", "<think>Inspect files.</think>\n<tool_call>{}</tool_call>"),
            Message("tool", "result", name="shell"),
            Message("assistant", "<think>Verify.</think>\nDone"),
        ],
        provenance=Provenance("unit/source"),
        session_id="s",
        repository_id="r",
    )
    built = build_training_examples(trajectory)
    flattened = [flatten_for_training(item) for item in built]
    for row in flattened:
        assert row["assistant_target_indices"]
    final = next(row for row in flattened if row["target_type"] == "final_answer")
    assert final["assistant_target_indices"] == [3]
    assert "<think>" not in final["completion"]
