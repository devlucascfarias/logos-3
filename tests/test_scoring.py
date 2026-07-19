import pytest

from qwen_sft.scoring import checkpoint_score, rank_checkpoints


WEIGHTS = {
    "code_verified": 0.40,
    "debugging": 0.25,
    "swe_tasks": 0.15,
    "tool_use": 0.10,
    "general_quality": 0.10,
}
PENALTIES = {"repetition": 1.0, "overthinking": 1.0}


def test_checkpoint_formula_and_ranking():
    metrics = {
        "code_verified": 1.0,
        "debugging": 1.0,
        "swe_tasks": 1.0,
        "tool_use": 1.0,
        "general_quality": 1.0,
        "repetition": 0.1,
        "overthinking": 0.2,
    }
    assert checkpoint_score(metrics, WEIGHTS, PENALTIES) == pytest.approx(0.7)

    ranked = rank_checkpoints(
        [
            {"checkpoint": "late", "metrics": dict(metrics, repetition=0.4)},
            {"checkpoint": "early", "metrics": metrics},
        ],
        WEIGHTS,
        PENALTIES,
    )
    assert ranked[0]["checkpoint"] == "early"


def test_missing_metrics_fail_closed():
    with pytest.raises(ValueError, match="Métricas ausentes"):
        checkpoint_score({"code_verified": 1.0}, WEIGHTS, PENALTIES)
