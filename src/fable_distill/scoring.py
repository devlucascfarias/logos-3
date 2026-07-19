from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def quality_score(
    *,
    verified_success: bool,
    build_passed: bool | None = None,
    completeness: float = 1.0,
    consistency: float = 1.0,
    efficiency: float = 1.0,
    diversity: float = 0.5,
) -> float:
    if verified_success:
        success = 1.0
    elif build_passed:
        success = 0.5
    else:
        success = 0.0
    components = [
        (0.35, success),
        (0.20, completeness),
        (0.15, consistency),
        (0.15, efficiency),
        (0.15, diversity),
    ]
    return round(sum(weight * max(0.0, min(1.0, float(value))) for weight, value in components), 6)


def trajectory_heuristics(messages: Iterable[dict[str, Any]], row: dict[str, Any]) -> float:
    items = list(messages)
    roles = [str(item.get("role", "")).lower() for item in items]
    has_user = "user" in roles
    has_assistant = "assistant" in roles
    has_tool = "tool" in roles
    assistant_count = roles.count("assistant")
    tool_count = roles.count("tool")
    completeness = 1.0 if has_user and has_assistant else 0.0
    if has_user and has_assistant and items[-1].get("role") == "assistant":
        completeness = 1.0
    elif has_user and has_assistant:
        completeness = 0.7
    consistency = 1.0 if tool_count <= assistant_count + 1 else 0.5
    repeated_penalty = max(0, assistant_count - len(set(
        str(item.get("content", "")) for item in items if item.get("role") == "assistant"
    )))
    efficiency = max(0.0, 1.0 - 0.1 * repeated_penalty)
    diversity = float(row.get("diversity_score", 0.5))
    success = bool(
        row.get("verified_success")
        or row.get("success")
        or row.get("tests_passed")
        or row.get("passed")
    )
    build = row.get("build_passed")
    if build is None and has_tool:
        joined = "\n".join(str(item.get("content", "")) for item in items[-4:])
        build = "exit code: 0" in joined.lower() or "passed" in joined.lower()
    return quality_score(
        verified_success=success,
        build_passed=bool(build),
        completeness=completeness,
        consistency=consistency,
        efficiency=efficiency,
        diversity=diversity,
    )


def preference_reward(
    tests_passed: bool,
    build_passed: bool,
    invalid_calls: int,
    normalized_patch_size: float,
    extra_penalty: float = 0.0,
) -> float:
    return (
        2.0 * int(tests_passed)
        + 0.5 * int(build_passed)
        - 0.1 * max(0, invalid_calls)
        - 0.01 * max(0.0, normalized_patch_size)
        - max(0.0, extra_penalty)
    )

