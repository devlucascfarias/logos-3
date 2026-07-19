from __future__ import annotations

from typing import Any


def checkpoint_score(
    metrics: dict[str, Any],
    weights: dict[str, float],
    penalties: dict[str, float],
) -> float:
    missing = (set(weights) | set(penalties)) - set(metrics)
    if missing:
        raise ValueError(f"Métricas ausentes: {sorted(missing)}")
    score = sum(float(weights[key]) * float(metrics[key]) for key in weights)
    score -= sum(
        float(penalties[key]) * float(metrics[key]) for key in penalties
    )
    return score


def rank_checkpoints(
    reports: list[dict[str, Any]],
    weights: dict[str, float],
    penalties: dict[str, float],
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for report in reports:
        metrics = report.get("metrics", report)
        item = dict(report)
        item["score"] = checkpoint_score(metrics, weights, penalties)
        ranked.append(item)
    return sorted(
        ranked,
        key=lambda item: (-float(item["score"]), str(item.get("checkpoint", ""))),
    )
