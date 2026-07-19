from __future__ import annotations

from typing import Any

from .formatting import tokenize_with_assistant_mask


class AssistantOnlyDataCollator:
    """Pads examples and masks system/user/tool/padding tokens with -100."""

    def __init__(self, tokenizer: Any, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("PyTorch is required for the training collator") from exc

        encoded = [
            tokenize_with_assistant_mask(
                self.tokenizer,
                feature["messages"],
                max_length=self.max_length,
                assistant_target_indices=feature.get("assistant_target_indices"),
            )
            for feature in features
        ]
        max_size = max(len(item["input_ids"]) for item in encoded)
        batch: dict[str, list[list[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in encoded:
            padding = max_size - len(item["input_ids"])
            batch["input_ids"].append(item["input_ids"] + [self.tokenizer.pad_token_id] * padding)
            batch["attention_mask"].append(item["attention_mask"] + [0] * padding)
            batch["labels"].append(item["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}
