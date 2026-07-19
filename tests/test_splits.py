from __future__ import annotations

from fable_distill.splitting import split_by_connected_groups, validate_splits


def make_row(index: int, repo: str, session: str, prompt: str) -> dict:
    return {
        "example_id": str(index),
        "repository_id": repo,
        "session_id": session,
        "normalized_prompt_hash": prompt,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "answer"},
        ],
    }


def test_connected_identifiers_never_cross_splits() -> None:
    rows = []
    for index in range(30):
        rows.append(make_row(index, f"repo-{index // 3}", f"session-{index}", f"prompt-{index}"))
    # This row connects repo-0 to a session otherwise associated with repo-4.
    rows.append(make_row(31, "repo-0", "session-12", "bridge"))
    splits = split_by_connected_groups(rows, seed=7)
    validate_splits(splits)
    locations = {
        name
        for name, values in splits.items()
        if any(row["repository_id"] == "repo-0" for row in values)
    }
    assert len(locations) == 1
    location = next(iter(locations))
    assert any(row["session_id"] == "session-12" for row in splits[location])
    assert sum(map(len, splits.values())) == len(rows)


def test_split_is_deterministic() -> None:
    rows = [make_row(index, f"repo-{index}", f"s-{index}", f"p-{index}") for index in range(20)]
    left = split_by_connected_groups(rows, seed=42)
    right = split_by_connected_groups(rows, seed=42)
    assert {
        name: [row["example_id"] for row in values] for name, values in left.items()
    } == {
        name: [row["example_id"] for row in values] for name, values in right.items()
    }

