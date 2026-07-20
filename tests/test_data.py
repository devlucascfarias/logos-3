from collections import Counter

from qwen_sft.data import (
    build_candidates,
    mix_by_tokens,
    normalize_messages,
    semantic_segments,
    split_by_group,
)


SYSTEM = "Sistema de teste."
DATA_CONFIG = {
    "max_seq_length": 64,
    "max_tool_output_chars": 100,
    "max_tool_messages": 8,
    "max_repeated_message_run": 2,
    "repeated_line_min_chars": 12,
    "max_repeated_line_occurrences": 3,
    "repeated_ngram_size": 8,
    "max_repeated_ngram_occurrences": 6,
    "reject_unclosed_blocks": True,
    "max_assistant_chars": 20_000,
    "min_tokens": 3,
    "blocked_benchmark_markers": ["humaneval"],
}


def count_messages(messages):
    return sum(len(message["content"].split()) + 1 for message in messages)


def count_text(text):
    return len(text.split())


def test_normalizes_open_code_reasoning_shape():
    messages = normalize_messages(
        {"input": "Corrija a função.", "output": "```python\npass\n```"},
        SYSTEM,
    )
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
    ]
    assert "pass" in messages[-1]["content"]


def test_normalizes_claude_json_and_reasoning_content():
    row = {
        "messages_json": (
            '[{"role":"user","content":"bug"},'
            '{"role":"assistant","reasoning_content":"causa curta",'
            '"content":"patch"}]'
        )
    }
    messages = normalize_messages(row, SYSTEM)
    assert messages[-1]["content"].startswith("<think>causa curta</think>")


def test_tool_calls_use_qwen_canonical_tags():
    messages = normalize_messages(
        {
            "messages": [
                {"role": "user", "content": "liste"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "list_files",
                                "arguments": "{\"path\":\".\"}",
                            }
                        }
                    ],
                },
            ]
        },
        SYSTEM,
    )
    assert "<tool_call>" in messages[-1]["content"]
    assert '"name":"list_files"' in messages[-1]["content"]
    assert "<tool_calls>" not in messages[-1]["content"]


def test_rejects_secrets_and_benchmark_contamination():
    source = {
        "name": "test/source",
        "category": "verified_code",
        "trusted_curated": True,
    }
    secret, reason = build_candidates(
        {
            "messages": [
                {"role": "user", "content": "use hf_abcdefghijklmnopqrstuvwxyz"},
                {"role": "assistant", "content": "não"},
            ]
        },
        row_index=0,
        source=source,
        system_prompt=SYSTEM,
        data_config=DATA_CONFIG,
        token_counter=count_messages,
        text_token_counter=count_text,
    )
    assert secret == []
    assert reason == "secret"

    benchmark, reason = build_candidates(
        {"input": "Resolva HumanEval 12.", "output": "resposta"},
        row_index=1,
        source=source,
        system_prompt=SYSTEM,
        data_config=DATA_CONFIG,
        token_counter=count_messages,
        text_token_counter=count_text,
    )
    assert benchmark == []
    assert reason == "benchmark_marker"


def test_rejects_repetition_and_unclosed_blocks():
    source = {
        "name": "test/source",
        "category": "verified_code",
        "trusted_curated": True,
    }
    repeated, reason = build_candidates(
        {
            "input": "Explique.",
            "output": "\n".join(["esta linha se repete"] * 4),
        },
        row_index=0,
        source=source,
        system_prompt=SYSTEM,
        data_config=DATA_CONFIG,
        token_counter=count_messages,
        text_token_counter=count_text,
    )
    assert repeated == []
    assert reason == "repeated_lines"

    unclosed, reason = build_candidates(
        {"input": "Corrija.", "output": "```python\nprint('x')"},
        row_index=1,
        source=source,
        system_prompt=SYSTEM,
        data_config=DATA_CONFIG,
        token_counter=count_messages,
        text_token_counter=count_text,
    )
    assert unclosed == []
    assert reason == "unclosed_block"


def test_can_deterministically_convert_reasoning_to_direct_answer():
    source = {
        "name": "test/source",
        "category": "verified_code",
        "trusted_curated": True,
    }
    data_config = {
        **DATA_CONFIG,
        "direct_conversion_fraction_by_category": {"verified_code": 1.0},
    }

    candidates, reason = build_candidates(
        {
            "id": "reasoning-example",
            "input": "Resolva.",
            "output": "<think>raciocínio interno</think>\nResposta final.",
        },
        row_index=0,
        source=source,
        system_prompt=SYSTEM,
        data_config=data_config,
        token_counter=count_messages,
        text_token_counter=count_text,
    )

    assert reason is None
    assert candidates[0]["reasoning_band"] == "direct"
    assert "<think>" not in candidates[0]["messages"][-1]["content"]


def test_semantic_segmentation_keeps_assistant_targets():
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "tarefa"},
        {"role": "assistant", "content": "a " * 20},
        {"role": "tool", "content": "resultado " * 20},
        {"role": "assistant", "content": "b " * 20},
    ]
    segments = semantic_segments(messages, count_messages, max_tokens=28)
    assert len(segments) == 2
    assert all(any(item["role"] == "assistant" for item in part) for part in segments)
    assert all(count_messages(part) <= 28 for part in segments)


def _example(category, band, index, tokens=100):
    return {
        "id": f"{category}-{band}-{index}",
        "group_id": f"group-{category}-{index}",
        "source": "fixture",
        "category": category,
        "reasoning_band": band,
        "verified": True,
        "num_tokens": tokens,
        "fingerprint": f"fp-{category}-{band}-{index}",
        "messages": [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "y"},
        ],
    }


def test_mix_is_by_tokens_and_group_split_has_no_overlap():
    categories = {"verified_code": 0.6, "reasoning": 0.4}
    bands = {"short": 0.55, "medium": 0.25, "long": 0.10, "direct": 0.10}
    examples = [
        _example(category, band, index)
        for category in categories
        for band in bands
        for index in range(70)
    ]
    selected, report = mix_by_tokens(
        examples,
        token_budget=20_000,
        category_weights=categories,
        reasoning_weights=bands,
        seed=42,
    )
    assert report["budget_reached"]
    token_counts = Counter()
    for example in selected:
        token_counts[example["category"]] += example["num_tokens"]
    assert token_counts["verified_code"] == 12_000
    assert token_counts["reasoning"] == 8_000
    assert report["max_category_deviation"] <= 0.01
    assert report["max_reasoning_deviation"] <= 0.01
    assert report["candidate_matrix"]["verified_code"]["short"]["examples"] == 70
    assert report["selected_matrix"]["reasoning"]["direct"]["tokens"] >= 0

    train, validation = split_by_group(
        selected, validation_fraction=0.2, seed=42
    )
    assert train
    assert validation
    assert not (
        {example["group_id"] for example in train}
        & {example["group_id"] for example in validation}
    )


def test_mix_does_not_fill_a_shortage_with_an_overrepresented_band():
    examples = [
        *[_example("verified_code", "short", index) for index in range(2)],
        *[_example("verified_code", "long", index) for index in range(20)],
    ]
    selected, report = mix_by_tokens(
        examples,
        token_budget=1000,
        category_weights={"verified_code": 1.0},
        reasoning_weights={"short": 0.5, "long": 0.5},
        seed=42,
    )

    assert not report["budget_reached"]
    assert sum(
        item["num_tokens"]
        for item in selected
        if item["reasoning_band"] == "long"
    ) == 500


def test_mix_solves_crossed_category_and_reasoning_constraints():
    examples = [
        *[
            _example("first", "shared", index)
            for index in range(10)
        ],
        *[
            _example("first", "first_only", index)
            for index in range(10)
        ],
        *[
            _example("second", "shared", index)
            for index in range(10)
        ],
        *[
            _example("second", "second_only", index)
            for index in range(10)
        ],
    ]
    selected, report = mix_by_tokens(
        examples,
        token_budget=2_000,
        category_weights={"first": 0.5, "second": 0.5},
        reasoning_weights={
            "shared": 0.5,
            "first_only": 0.25,
            "second_only": 0.25,
        },
        seed=42,
    )

    assert sum(example["num_tokens"] for example in selected) == 2_000
    assert report["matrix_feasible"]
    assert report["maximum_feasible_tokens"] == 2_000
    assert report["max_category_deviation"] == 0
    assert report["max_reasoning_deviation"] == 0


def test_group_split_honors_minimum_validation_size():
    examples = [
        _example("verified_code", "direct", index) for index in range(40)
    ]

    train, validation = split_by_group(
        examples,
        validation_fraction=0.02,
        min_validation_examples=16,
        seed=42,
    )

    assert train
    assert len(validation) >= 16
    assert not (
        {example["group_id"] for example in train}
        & {example["group_id"] for example in validation}
    )


def test_duplicate_fingerprints_are_removed_before_mixing():
    first = _example("verified_code", "short", 1)
    duplicate = dict(first, id="other-id")
    selected, report = mix_by_tokens(
        [first, duplicate],
        token_budget=100,
        category_weights={"verified_code": 1.0},
        reasoning_weights={"short": 1.0},
        seed=1,
    )
    assert len(selected) == 1
    assert report["available_unique_examples"] == 1


def test_near_duplicates_and_group_concentration_are_limited():
    first = {
        **_example("verified_code", "direct", 1),
        "near_fingerprint": "near-same",
    }
    near_duplicate = {
        **_example("verified_code", "direct", 2),
        "near_fingerprint": "near-same",
    }
    same_group = [
        {
            **_example("verified_code", "direct", index),
            "group_id": "shared-group",
        }
        for index in range(3, 8)
    ]
    selected, report = mix_by_tokens(
        [first, near_duplicate, *same_group],
        token_budget=300,
        category_weights={"verified_code": 1.0},
        reasoning_weights={"direct": 1.0},
        seed=1,
        max_examples_per_group=2,
    )

    assert len(selected) == 3
    assert sum(
        example["group_id"] == "shared-group" for example in selected
    ) == 2
    assert report["available_unique_examples"] == 3
    assert report["available_unique_groups"] == 2
