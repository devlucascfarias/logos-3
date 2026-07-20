from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qwen_sft.io import file_sha256, read_jsonl


_FENCE_RE = re.compile(
    r"```(?P<language>[^\n\r`]*)\r?\n(?P<code>.*?)```",
    re.DOTALL,
)
_ALLOWED_PYTHON_LABELS = {"python", "py"}


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExtractedPython:
    valid: bool
    code: str | None
    explanation: str
    fence_count: int
    reason: str | None = None


def extract_single_python_block(response: str) -> ExtractedPython:
    """Extract exactly one fenced Python block and a non-empty explanation."""

    matches = list(_FENCE_RE.finditer(response))
    delimiter_count = response.count("```")
    raw_fence_count = (delimiter_count + 1) // 2
    if len(matches) != 1 or delimiter_count != 2:
        return ExtractedPython(
            False,
            None,
            "",
            max(len(matches), raw_fence_count),
            "expected_exactly_one_fenced_block",
        )
    match = matches[0]
    language = match.group("language").strip().casefold()
    if language not in _ALLOWED_PYTHON_LABELS:
        return ExtractedPython(
            False,
            None,
            "",
            1,
            "fenced_block_is_not_python",
        )
    code = match.group("code").strip()
    explanation = (response[: match.start()] + response[match.end() :]).strip()
    if not code:
        return ExtractedPython(False, None, explanation, 1, "empty_python_block")
    if not explanation:
        return ExtractedPython(False, code, explanation, 1, "missing_explanation")
    return ExtractedPython(True, code, explanation, 1)


def repetition_fraction(text: str, *, ngram_size: int = 4) -> float:
    """Fraction of repeated word n-grams, normalized to [0, 1]."""

    words = re.findall(r"\w+|[^\w\s]", text.casefold(), flags=re.UNICODE)
    if len(words) < ngram_size:
        return 0.0
    ngrams = [tuple(words[i : i + ngram_size]) for i in range(len(words) - ngram_size + 1)]
    repeated = len(ngrams) - len(set(ngrams))
    return repeated / len(ngrams)


def _test_list(task: Mapping[str, Any], *keys: str) -> list[str]:
    value: Any = None
    for key in keys:
        if key in task:
            value = task[key]
            break
    if value is None and isinstance(task.get("evaluation"), Mapping):
        evaluation = task["evaluation"]
        for key in keys:
            if key in evaluation:
                value = evaluation[key]
                break
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not value:
        return []
    tests: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            tests.append(item.strip())
        elif isinstance(item, Mapping):
            code = item.get("code") or item.get("test")
            if isinstance(code, str) and code.strip():
                tests.append(code.strip())
    return tests


def _prediction_is_truncated(prediction: Mapping[str, Any]) -> bool:
    if bool(prediction.get("truncated")):
        return True
    if str(prediction.get("finish_reason", "")).casefold() in {"length", "max_tokens"}:
        return True
    generated = prediction.get("generated_tokens")
    generation = prediction.get("generation")
    maximum = generation.get("max_new_tokens") if isinstance(generation, Mapping) else None
    return (
        isinstance(generated, int)
        and isinstance(maximum, int)
        and generated >= maximum
        and not bool(prediction.get("ended_with_eos"))
    )


def _result_value(result: Any, name: str, default: Any = None) -> Any:
    if isinstance(result, Mapping):
        return result.get(name, default)
    return getattr(result, name, default)


def _sandbox_passed(result: Any) -> bool:
    passed = _result_value(result, "passed")
    if passed is not None:
        return bool(passed)
    ok = _result_value(result, "ok")
    if ok is not None:
        return bool(ok)
    returncode = _result_value(result, "returncode")
    return returncode == 0


SandboxRunner = Callable[[str, str], Any]


def _run_tests(
    code: str,
    tests: Sequence[str],
    runner: SandboxRunner,
    cache: dict[str, Any],
) -> tuple[int, bool, list[str]]:
    passed = 0
    timed_out = False
    statuses: list[str] = []
    for test in tests:
        cache_key = text_sha256(code + "\n# hidden test\n" + test)
        if cache_key not in cache:
            try:
                cache[cache_key] = runner(code, test)
            except ValueError as exc:
                cache[cache_key] = {
                    "passed": False,
                    "status": "rejected",
                    "error": str(exc),
                }
        result = cache[cache_key]
        status = str(_result_value(result, "status", "")).casefold()
        timeout = bool(
            _result_value(result, "timed_out", _result_value(result, "timeout", False))
        ) or status == "timeout"
        success = _sandbox_passed(result) and not timeout
        passed += int(success)
        timed_out = timed_out or timeout
        statuses.append(
            "pass" if success else "timeout" if timeout else status or "fail"
        )
    return passed, timed_out, statuses


def evaluate_response(
    task: Mapping[str, Any],
    prediction: Mapping[str, Any],
    runner: SandboxRunner,
) -> dict[str, Any]:
    response = prediction.get("response")
    if not isinstance(response, str):
        raise ValueError(f"Prediction {prediction.get('id')} has no string response.")
    hidden_tests = _test_list(task, "hidden_tests", "tests_hidden")
    if not hidden_tests:
        raise ValueError(f"Task {task.get('id')} has no hidden tests.")
    contract_tests = _test_list(
        task, "contract_tests", "tests_contract", "contract_checks"
    )
    if not contract_tests:
        contract_tests = hidden_tests

    extracted = extract_single_python_block(response)
    cache: dict[str, Any] = {}
    hidden_passed = 0
    contract_passed = 0
    timed_out = False
    hidden_statuses = ["not_run"] * len(hidden_tests)
    contract_statuses = ["not_run"] * len(contract_tests)
    if extracted.valid and extracted.code is not None:
        hidden_passed, hidden_timeout, hidden_statuses = _run_tests(
            extracted.code, hidden_tests, runner, cache
        )
        contract_passed, contract_timeout, contract_statuses = _run_tests(
            extracted.code, contract_tests, runner, cache
        )
        timed_out = hidden_timeout or contract_timeout

    source_rejected = "rejected" in hidden_statuses or "rejected" in contract_statuses
    format_valid = extracted.valid and not source_rejected
    test_fraction = hidden_passed / len(hidden_tests)
    contract_fraction = contract_passed / len(contract_tests)
    truncated = _prediction_is_truncated(prediction)
    functional_pass = bool(
        format_valid
        and hidden_passed == len(hidden_tests)
        and not timed_out
        and not truncated
    )
    family = str(task.get("family") or task.get("category") or "uncategorized")
    code_hash = text_sha256(extracted.code) if extracted.code is not None else None
    return {
        "id": str(task.get("id")),
        "family": family,
        "checkpoint": str(prediction.get("checkpoint") or prediction.get("model") or "unknown"),
        "functional_pass": functional_pass,
        "test_fraction": test_fraction,
        "contract_fraction": contract_fraction,
        "format": float(format_valid),
        "format_pass": format_valid,
        "format_reason": "sandbox_rejected" if source_rejected else extracted.reason,
        "timeout": timed_out,
        "truncated": truncated,
        "repetition": repetition_fraction(response),
        "hidden_tests": {"passed": hidden_passed, "total": len(hidden_tests), "statuses": hidden_statuses},
        "contract_tests": {
            "passed": contract_passed,
            "total": len(contract_tests),
            "statuses": contract_statuses,
        },
        "response_sha256": text_sha256(response),
        "code_sha256": code_hash,
        "hidden_tests_sha256": canonical_sha256(hidden_tests),
        "contract_tests_sha256": canonical_sha256(contract_tests),
    }


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row[key]) for row in rows) / len(rows)


def aggregate_evaluations(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot aggregate an empty evaluation.")
    by_family: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_family.setdefault(str(row["family"]), []).append(row)

    def aggregate(part: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "task_count": len(part),
            "functional_pass_count": sum(bool(row["functional_pass"]) for row in part),
            "functional_pass": _mean(part, "functional_pass"),
            "test_fraction": _mean(part, "test_fraction"),
            "contract_fraction": _mean(part, "contract_fraction"),
            "format": _mean(part, "format"),
            "timeout": _mean(part, "timeout"),
            "truncated": _mean(part, "truncated"),
            "repetition": _mean(part, "repetition"),
        }

    result = aggregate(rows)
    result["families"] = {
        family: aggregate(family_rows)
        for family, family_rows in sorted(by_family.items())
    }
    return result


def rank_contract_models(
    models: Mapping[str, Mapping[str, Any]],
    eval_losses: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Rank checkpoints lexicographically according to the corrective-v1 gate."""

    losses = eval_losses or {}
    ranked = []
    for checkpoint, metrics in models.items():
        item = {"checkpoint": checkpoint, **dict(metrics)}
        item["eval_loss"] = (
            float(losses[checkpoint]) if checkpoint in losses else None
        )
        ranked.append(item)
    return sorted(
        ranked,
        key=lambda item: (
            -int(item["functional_pass_count"]),
            -float(item["contract_fraction"]),
            -float(item["format"]),
            float(item["timeout"]) + float(item["truncated"]),
            (
                float(item["eval_loss"])
                if item["eval_loss"] is not None
                else math.inf
            ),
            str(item["checkpoint"]),
        ),
    )


def build_contract_report(
    tasks_path: str | Path,
    prediction_paths: Sequence[str | Path],
    runner: SandboxRunner,
    *,
    eval_losses: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    tasks = list(read_jsonl(tasks_path))
    tasks_by_id: dict[str, dict[str, Any]] = {}
    for task in tasks:
        task_id = str(task.get("id", "")).strip()
        if not task_id or task_id in tasks_by_id:
            raise ValueError(f"Invalid or duplicate task id: {task_id!r}")
        tasks_by_id[task_id] = task

    evaluations: list[dict[str, Any]] = []
    prediction_hashes: dict[str, str] = {}
    seen: set[tuple[str, str]] = set()
    for path in prediction_paths:
        prediction_hashes[str(path)] = file_sha256(path)
        for prediction in read_jsonl(path):
            task_id = str(prediction.get("id", ""))
            if task_id not in tasks_by_id:
                raise ValueError(f"Prediction references unknown task id: {task_id!r}")
            checkpoint = str(prediction.get("checkpoint") or prediction.get("model") or "unknown")
            key = (checkpoint, task_id)
            if key in seen:
                raise ValueError(f"Duplicate prediction for {checkpoint!r}/{task_id!r}.")
            seen.add(key)
            evaluations.append(evaluate_response(tasks_by_id[task_id], prediction, runner))

    if not evaluations:
        raise ValueError("No predictions were evaluated.")

    checkpoints = {row["checkpoint"] for row in evaluations}
    for checkpoint in checkpoints:
        missing = set(tasks_by_id) - {
            row["id"] for row in evaluations if row["checkpoint"] == checkpoint
        }
        if missing:
            raise ValueError(f"Checkpoint {checkpoint!r} is missing tasks: {sorted(missing)}")
    models = {
        checkpoint: aggregate_evaluations(
            [row for row in evaluations if row["checkpoint"] == checkpoint]
        )
        for checkpoint in sorted(checkpoints)
    }
    ranking = rank_contract_models(models, eval_losses)
    evaluation_spec = {
        "format": "exactly_one_python_fence_with_explanation",
        "functional_pass": "format_and_all_hidden_tests_and_not_timeout_or_truncated",
        "repetition_ngram_size": 4,
        "ranking": [
            "functional_pass_count_desc",
            "contract_fraction_desc",
            "format_desc",
            "timeout_plus_truncated_asc",
            "eval_loss_asc",
        ],
    }
    return {
        "schema_version": 1,
        "evaluation_spec": evaluation_spec,
        "evaluation_spec_sha256": canonical_sha256(evaluation_spec),
        "evaluator_sha256": file_sha256(Path(__file__)),
        "tasks": str(tasks_path),
        "tasks_sha256": file_sha256(tasks_path),
        "predictions_sha256": prediction_hashes,
        "task_count": len(tasks),
        "models": models,
        "ranking": ranking,
        "top_three": [item["checkpoint"] for item in ranking[:3]],
        "evaluations": evaluations,
    }


def aggregate_manual_ratings(
    ratings: Sequence[Mapping[str, Any]],
    mapping: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    mapping_by_id: dict[str, Mapping[str, Any]] = {}
    for item in mapping:
        task_id = str(item.get("id", ""))
        if not task_id or task_id in mapping_by_id:
            raise ValueError(f"Invalid or duplicate mapping id: {task_id!r}")
        mapping_by_id[task_id] = item
    totals: dict[str, dict[str, int]] = {}
    seen_ratings: set[str] = set()
    for item in ratings:
        task_id = str(item.get("id"))
        if task_id in seen_ratings:
            raise ValueError(f"Duplicate rating id: {task_id}")
        seen_ratings.add(task_id)
        if task_id not in mapping_by_id:
            raise ValueError(f"Rating without mapping: {task_id}")
        labels = item.get("ratings")
        if not isinstance(labels, Mapping):
            raise ValueError(f"Invalid ratings for {task_id}")
        for label, scores in labels.items():
            identity = mapping_by_id[task_id].get(label)
            if not isinstance(identity, str) or not isinstance(scores, Mapping):
                raise ValueError(f"Invalid label {label!r} for {task_id}")
            bucket = totals.setdefault(
                identity,
                {
                    "task_count": 0,
                    "correctness": 0,
                    "instruction_following": 0,
                    "explanation_quality": 0,
                    "total": 0,
                },
            )
            bucket["task_count"] += 1
            for metric in ("correctness", "instruction_following", "explanation_quality"):
                value = scores.get(metric)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not 0 <= value <= 5
                    or not float(value).is_integer()
                ):
                    raise ValueError(f"Invalid {metric} rating for {task_id}/{label}: {value!r}")
                bucket[metric] += int(value)
                bucket["total"] += int(value)
    return totals


def promotion_gate(
    hidden_models: Mapping[str, Mapping[str, Any]],
    regression_totals: Mapping[str, Mapping[str, int]],
    *,
    candidate: str,
    champion: str,
    base: str,
    candidate_rating_identity: str | None = None,
    base_rating_identity: str | None = None,
) -> dict[str, Any]:
    for identity in (candidate, champion):
        if identity not in hidden_models:
            raise ValueError(f"Hidden report is missing model {identity!r}.")
    candidate_rating_identity = candidate_rating_identity or candidate
    base_rating_identity = base_rating_identity or base
    for identity in (candidate_rating_identity, base_rating_identity):
        if identity not in regression_totals:
            raise ValueError(f"Regression ratings are missing identity {identity!r}.")
    candidate_hidden = hidden_models[candidate]
    champion_hidden = hidden_models[champion]
    candidate_regression = regression_totals[candidate_rating_identity]
    base_regression = regression_totals[base_rating_identity]

    families = set(candidate_hidden.get("families", {})) | set(champion_hidden.get("families", {}))
    family_drops = {
        family: int(candidate_hidden.get("families", {}).get(family, {}).get("functional_pass_count", 0))
        - int(champion_hidden.get("families", {}).get(family, {}).get("functional_pass_count", 0))
        for family in sorted(families)
    }
    checks = {
        "hidden_has_33_tasks": int(candidate_hidden.get("task_count", 0)) == 33
        and int(champion_hidden.get("task_count", 0)) == 33,
        "hidden_plus_three": int(candidate_hidden["functional_pass_count"])
        >= int(champion_hidden["functional_pass_count"]) + 3,
        "contract_not_worse": float(candidate_hidden["contract_fraction"])
        >= float(champion_hidden["contract_fraction"]),
        "format_not_worse": float(candidate_hidden["format"])
        >= float(champion_hidden["format"]),
        "family_drop_at_most_one": all(delta >= -1 for delta in family_drops.values()),
        "regression_has_13_tasks": int(candidate_regression.get("task_count", 0)) == 13
        and int(base_regression.get("task_count", 0)) == 13,
        "regression_total_at_least_84": int(candidate_regression["total"]) >= 84,
        "regression_correctness_at_least_37": int(candidate_regression["correctness"]) >= 37,
        "regression_instruction_at_least_29": int(candidate_regression["instruction_following"]) >= 29,
        "regression_explanation_at_least_17": int(candidate_regression["explanation_quality"]) >= 17,
        "regression_beats_base": int(candidate_regression["total"]) > int(base_regression["total"]),
    }
    return {
        "promote": all(checks.values()),
        "candidate": candidate,
        "champion": champion,
        "base": base,
        "candidate_rating_identity": candidate_rating_identity,
        "base_rating_identity": base_rating_identity,
        "checks": checks,
        "family_functional_deltas": family_drops,
        "hidden_functional_delta": int(candidate_hidden["functional_pass_count"])
        - int(champion_hidden["functional_pass_count"]),
        "regression": {"candidate": dict(candidate_regression), "base": dict(base_regression)},
    }
