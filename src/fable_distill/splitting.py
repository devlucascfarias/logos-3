from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from .formatting import normalized_prompt_hash


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def _identifier_values(row: dict[str, Any]) -> list[str]:
    prompt_hash = str(
        row.get("normalized_prompt_hash") or normalized_prompt_hash(row.get("messages", []))
    )
    values = [f"prompt:{prompt_hash}"]
    session = str(row.get("session_id", "")).strip()
    repository = str(row.get("repository_id", "")).strip()
    if session and session != "unknown":
        values.append(f"session:{session}")
    if repository and repository != "unknown":
        values.append(f"repository:{repository}")
    return values


def split_by_connected_groups(
    rows: list[dict[str, Any]],
    train_ratio: float = 0.90,
    validation_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    if not rows:
        return {"train": [], "validation": [], "test": []}
    if abs(train_ratio + validation_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("Split ratios must add to 1.0")

    union = _UnionFind(len(rows))
    owners: dict[str, int] = {}
    for index, row in enumerate(rows):
        for value in _identifier_values(row):
            if value in owners:
                union.union(index, owners[value])
            else:
                owners[value] = index

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        components[union.find(index)].append(index)
    groups = list(components.values())
    random.Random(seed).shuffle(groups)
    groups.sort(key=len, reverse=True)

    ratios = {"train": train_ratio, "validation": validation_ratio, "test": test_ratio}
    targets = {name: ratio * len(rows) for name, ratio in ratios.items()}
    assignments: dict[str, list[int]] = {name: [] for name in ratios}
    for group in groups:
        destination = max(
            ratios,
            key=lambda name: (targets[name] - len(assignments[name]), ratios[name]),
        )
        assignments[destination].extend(group)

    result = {
        name: [rows[index] for index in indices]
        for name, indices in assignments.items()
    }
    validate_splits(result)
    return result


def validate_splits(splits: dict[str, list[dict[str, Any]]]) -> None:
    fields = ("session_id", "repository_id")
    names = list(splits)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            for field in fields:
                left_values = {
                    str(row.get(field))
                    for row in splits[left_name]
                    if row.get(field) not in {None, "", "unknown"}
                }
                right_values = {
                    str(row.get(field))
                    for row in splits[right_name]
                    if row.get(field) not in {None, "", "unknown"}
                }
                overlap = left_values & right_values
                if overlap:
                    raise ValueError(
                        f"Leakage in {field} between {left_name} and {right_name}: "
                        f"{sorted(overlap)[:3]}"
                    )
            left_prompts = {
                str(row.get("normalized_prompt_hash") or normalized_prompt_hash(row.get("messages", [])))
                for row in splits[left_name]
            }
            right_prompts = {
                str(row.get("normalized_prompt_hash") or normalized_prompt_hash(row.get("messages", [])))
                for row in splits[right_name]
            }
            overlap = left_prompts & right_prompts
            if overlap:
                raise ValueError(
                    f"Prompt leakage between {left_name} and {right_name}: {len(overlap)} hashes"
                )

