from __future__ import annotations

from collections import Counter

from build_examples import _weighted_take


def test_weighted_take_respects_configured_source_mix() -> None:
    pools = {
        "primary": [{"source_dataset": "primary", "id": index} for index in range(100)],
        "reasoning": [{"source_dataset": "reasoning", "id": index} for index in range(100)],
        "general": [{"source_dataset": "general", "id": index} for index in range(100)],
    }
    selected = _weighted_take(
        pools,
        {"primary": 0.65, "reasoning": 0.20, "general": 0.15},
        total=100,
        seed=42,
    )
    counts = Counter(row["source_dataset"] for row in selected)
    assert counts == {"primary": 65, "reasoning": 20, "general": 15}

