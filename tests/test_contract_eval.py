import json
from pathlib import Path

import pytest

from qwen_sft.contract_eval import (
    aggregate_manual_ratings,
    build_contract_report,
    evaluate_response,
    extract_single_python_block,
    promotion_gate,
    rank_contract_models,
    repetition_fraction,
)


def _runner(code: str, test: str):
    namespace = {}
    try:
        exec(compile(code + "\n" + test, "<test>", "exec"), namespace, namespace)
    except Exception:
        return {"passed": False, "status": "failed"}
    return {"passed": True, "status": "passed"}


def _model(passed: int, *, contracts: float = 1.0, format_score: float = 1.0):
    return {
        "task_count": 33,
        "functional_pass_count": passed,
        "functional_pass": passed / 33,
        "test_fraction": passed / 33,
        "contract_fraction": contracts,
        "format": format_score,
        "timeout": 0.0,
        "truncated": 0.0,
        "repetition": 0.0,
        "families": {
            "retry": {"functional_pass_count": 3},
            "parser": {"functional_pass_count": 3},
        },
    }


def _regression(total=84, correctness=37, instructions=29, explanation=18):
    assert correctness + instructions + explanation == total
    return {
        "task_count": 13,
        "total": total,
        "correctness": correctness,
        "instruction_following": instructions,
        "explanation_quality": explanation,
    }


def test_extract_requires_one_python_block_and_explanation():
    valid = extract_single_python_block("Breve.\n```python\ndef f():\n    return 1\n```")
    assert valid.valid
    assert "def f" in valid.code

    assert not extract_single_python_block("```python\nx = 1\n```").valid
    assert not extract_single_python_block(
        "Texto\n```python\nx = 1\n```\n```text\nextra\n```"
    ).valid
    assert not extract_single_python_block("Texto\n```javascript\nx = 1\n```").valid
    assert not extract_single_python_block(
        "Texto\n```python\nx = 1\n```\n```"
    ).valid


def test_evaluate_response_metrics_and_truncation():
    task = {
        "id": "increment",
        "family": "collections",
        "hidden_tests": ["assert inc(1) == 2", "assert inc(-1) == 0"],
        "contract_checks": ["assert callable(inc)"],
    }
    prediction = {
        "id": "increment",
        "checkpoint": "checkpoint-2",
        "response": "Soma um ao valor.\n```python\ndef inc(x):\n    return x + 1\n```",
        "truncated": False,
    }
    result = evaluate_response(task, prediction, _runner)
    assert result["functional_pass"] is True
    assert result["test_fraction"] == 1.0
    assert result["contract_fraction"] == 1.0
    assert result["format"] == 1.0
    assert result["hidden_tests"]["total"] == 2
    assert len(result["hidden_tests_sha256"]) == 64

    truncated = evaluate_response(task, {**prediction, "truncated": True}, _runner)
    assert truncated["test_fraction"] == 1.0
    assert truncated["functional_pass"] is False
    assert truncated["truncated"] is True


def test_failed_and_timeout_tests_are_counted():
    responses = iter(
        [
            {"passed": True, "status": "passed"},
            {"passed": False, "status": "timeout"},
        ]
    )
    task = {
        "id": "one",
        "family": "retry",
        "hidden_tests": ["assert True", "assert True  # second"],
    }
    result = evaluate_response(
        task,
        {
            "id": "one",
            "checkpoint": "candidate",
            "response": "Explicacao.\n```python\nx = 1\n```",
        },
        lambda _code, _test: next(responses),
    )
    assert result["test_fraction"] == 0.5
    assert result["timeout"] is True
    assert result["functional_pass"] is False


def test_sandbox_source_rejection_fails_format():
    task = {"id": "one", "hidden_tests": ["assert True"]}

    def reject(_code, _test):
        raise ValueError("Import is not allowed")

    result = evaluate_response(
        task,
        {
            "id": "one",
            "checkpoint": "candidate",
            "response": "Explicacao.\n```python\nimport os\n```",
        },
        reject,
    )
    assert result["format"] == 0.0
    assert result["format_reason"] == "sandbox_rejected"


def test_repetition_fraction_is_normalized():
    assert repetition_fraction("texto curto") == 0.0
    score = repetition_fraction("um dois tres quatro um dois tres quatro")
    assert 0.0 < score <= 1.0


def test_report_hashes_completeness_and_lexicographic_ranking(tmp_path: Path):
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        json.dumps(
            {
                "id": "one",
                "family": "parser",
                "hidden_tests": ["assert answer() == 1"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps(
            {
                "id": "one",
                "checkpoint": "checkpoint-2",
                "response": "Direto.\n```python\ndef answer(): return 1\n```",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report = build_contract_report(
        tasks,
        [predictions],
        _runner,
        eval_losses={"checkpoint-2": 0.8},
    )
    assert report["models"]["checkpoint-2"]["functional_pass_count"] == 1
    assert report["top_three"] == ["checkpoint-2"]
    assert len(report["tasks_sha256"]) == 64
    assert len(next(iter(report["predictions_sha256"].values()))) == 64

    ranked = rank_contract_models(
        {
            "a": {**_model(20), "contract_fraction": 0.8},
            "b": {**_model(20), "contract_fraction": 0.9},
            "c": _model(21, contracts=0.1),
        },
        {"a": 0.1, "b": 0.9, "c": 2.0},
    )
    assert [row["checkpoint"] for row in ranked] == ["c", "b", "a"]


def test_report_fails_closed_on_missing_predictions(tmp_path: Path):
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        "\n".join(
            json.dumps({"id": item, "hidden_tests": ["assert True"]})
            for item in ("one", "two")
        )
        + "\n",
        encoding="utf-8",
    )
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps(
            {
                "id": "one",
                "checkpoint": "candidate",
                "response": "Ok.\n```python\nx = 1\n```",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing tasks"):
        build_contract_report(tasks, [predictions], _runner)


def test_manual_ratings_follow_per_task_blind_mapping():
    ratings = [
        {
            "id": "one",
            "ratings": {
                "A": {
                    "correctness": 5,
                    "instruction_following": 4,
                    "explanation_quality": 3,
                },
                "B": {
                    "correctness": 2,
                    "instruction_following": 1,
                    "explanation_quality": 0,
                },
            },
        }
    ]
    totals = aggregate_manual_ratings(
        ratings, [{"id": "one", "A": "adapter", "B": "base"}]
    )
    assert totals["adapter"]["total"] == 12
    assert totals["base"]["correctness"] == 2


def test_promotion_gate_passes_exact_boundaries():
    candidate = _model(23)
    champion = _model(20)
    candidate["families"]["retry"]["functional_pass_count"] = 2
    gate = promotion_gate(
        {"candidate": candidate, "champion": champion},
        {"candidate": _regression(), "base": _regression(83, 36, 29, 18)},
        candidate="candidate",
        champion="champion",
        base="base",
    )
    assert gate["promote"] is True
    assert all(gate["checks"].values())


@pytest.mark.parametrize(
    "mutation,failed_check",
    [
        (lambda candidate, champion, regression: candidate.update(task_count=32), "hidden_has_33_tasks"),
        (lambda candidate, champion, regression: candidate.update(functional_pass_count=22), "hidden_plus_three"),
        (lambda candidate, champion, regression: candidate.update(contract_fraction=0.99), "contract_not_worse"),
        (lambda candidate, champion, regression: candidate.update(format=0.99), "format_not_worse"),
        (
            lambda candidate, champion, regression: candidate["families"]["retry"].update(functional_pass_count=1),
            "family_drop_at_most_one",
        ),
        (lambda candidate, champion, regression: regression.update(task_count=12), "regression_has_13_tasks"),
        (lambda candidate, champion, regression: regression.update(total=83), "regression_total_at_least_84"),
        (lambda candidate, champion, regression: regression.update(correctness=36), "regression_correctness_at_least_37"),
        (lambda candidate, champion, regression: regression.update(instruction_following=28), "regression_instruction_at_least_29"),
        (lambda candidate, champion, regression: regression.update(explanation_quality=16), "regression_explanation_at_least_17"),
    ],
)
def test_promotion_gate_fails_each_boundary(mutation, failed_check):
    candidate = _model(23)
    champion = _model(20)
    regression = _regression()
    mutation(candidate, champion, regression)
    gate = promotion_gate(
        {"candidate": candidate, "champion": champion},
        {"candidate": regression, "base": _regression(83, 36, 29, 18)},
        candidate="candidate",
        champion="champion",
        base="base",
    )
    assert gate["promote"] is False
    assert gate["checks"][failed_check] is False


def test_promotion_gate_requires_candidate_to_strictly_beat_base():
    gate = promotion_gate(
        {"candidate": _model(23), "champion": _model(20)},
        {"candidate": _regression(), "base": _regression()},
        candidate="candidate",
        champion="champion",
        base="base",
    )
    assert gate["promote"] is False
    assert gate["checks"]["regression_beats_base"] is False
