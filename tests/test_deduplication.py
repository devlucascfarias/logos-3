from __future__ import annotations

from fable_distill.deduplication import deduplicate_rows


def row(content: str, example_id: str) -> dict:
    return {
        "example_id": example_id,
        "source_dataset": "unit/source",
        "source_example_id": example_id,
        "messages": [
            {"role": "user", "content": "Fix the parser"},
            {"role": "assistant", "content": content},
        ],
        "prompt": "Fix the parser",
        "completion": content,
        "target_type": "next_action",
    }


def test_exact_and_approximate_deduplication() -> None:
    first = row("Run pytest and inspect the failing parser module.", "1")
    exact = dict(first)
    near = row("Run pytest, then inspect the failing parser module.", "2")
    distinct = row("Implement a Rust lexer using a state machine.", "3")
    kept, report = deduplicate_rows([first, exact, near, distinct], approximate_threshold=0.7)
    assert len(kept) == 2
    assert report["removed_exact"] == 1
    assert report["removed_approximate"] == 1


def test_distinct_targets_from_same_source_are_not_origin_duplicates() -> None:
    first = row("Inspect.", "1")
    second = row("Conclude.", "1")
    second["target_type"] = "final_answer"
    kept, _ = deduplicate_rows([first, second])
    assert len(kept) == 2

