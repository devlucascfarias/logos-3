from __future__ import annotations

from fable_distill.formatting import (
    apply_reasoning_mode,
    build_training_examples,
    flatten_for_training,
    render_messages,
    tokenize_with_assistant_mask,
)
from fable_distill.schemas import CanonicalExample, Message, Provenance


class CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    eos_token = "<eos>"
    pad_token = "<pad>"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(character) + 2 for character in text]


def example() -> CanonicalExample:
    return CanonicalExample(
        messages=[
            Message("system", "system"),
            Message("user", "question"),
            Message("assistant", "<plan>inspect</plan>\nanswer"),
        ],
        provenance=Provenance("unit/source"),
        session_id="session-1",
        repository_id="repo-1",
        quality_score=0.8,
    )


def test_chatml_format_and_flatten() -> None:
    value = flatten_for_training(example())
    assert value["prompt"].endswith("<|im_start|>assistant\n")
    assert value["completion"] == "<plan>inspect</plan>\nanswer"
    assert render_messages(example().messages).count("<|im_start|>") == 3


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
    assert "<plan>" in compressed and "<think>" not in compressed
    assert hidden == "Done."


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
