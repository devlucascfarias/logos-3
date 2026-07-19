from __future__ import annotations

from fable_distill.scoring import preference_reward, quality_score


def test_quality_formula() -> None:
    assert quality_score(
        verified_success=True,
        completeness=1,
        consistency=1,
        efficiency=1,
        diversity=1,
    ) == 1.0
    assert quality_score(
        verified_success=False,
        build_passed=True,
        completeness=1,
        consistency=1,
        efficiency=1,
        diversity=1,
    ) == 0.825


def test_verified_reward_prefers_passing_small_patch() -> None:
    winner = preference_reward(True, True, 0, 10)
    loser = preference_reward(False, True, 2, 100)
    assert winner > loser

