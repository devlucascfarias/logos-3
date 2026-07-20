import hashlib
import json
import os

import pytest

from qwen_sft.corrective import (
    EVALUATION_PROMPTS,
    FAMILY_SPECS,
    OPENCODE_REVISION,
    assert_not_contaminated,
    canonicalize_opencode_row,
    generate_contract_candidates,
    generate_evaluation_tasks,
    load_replay_candidates,
    verify_candidate,
)
from qwen_sft.sandbox import run_verified
from qwen_sft.data import split_by_group


def count_messages(messages):
    return sum(len(message["content"].split()) + 1 for message in messages)


def perfect_opencode_row(**overrides):
    judgement = {
        name: {"score": 5, "justification": "The implementation is correct and handles its edge cases."}
        for name in (
            "requirement_conformance",
            "logical_correctness",
            "edge_case_consideration",
        )
    }
    row = {
        "id": "row-1",
        "input": "Implement add(a, b).",
        "output": "```python\ndef add(a, b):\n    return a + b\n```",
        "llm_judgement": json.dumps(judgement),
        "unit_tests": repr(["assert add(2, 3) == 5", "assert add(-1, 1) == 0"]),
        "tests_execution_status": repr(["pass", "pass"]),
        "average_test_score": "1.0",
    }
    row.update(overrides)
    return row


def test_opencode_filter_and_canonicalization():
    candidate = canonicalize_opencode_row(
        perfect_opencode_row(), token_counter=count_messages, execute=False
    )
    assert candidate["verified"] is False
    assert candidate["verification_level"] == "local_public_tests"
    assert candidate["source_revision"] == OPENCODE_REVISION
    assert candidate["source_sha256"] == hashlib.sha256(
        candidate["source"].encode("utf-8")
    ).hexdigest()
    assert candidate["revision_sha256"] == hashlib.sha256(
        OPENCODE_REVISION.encode("utf-8")
    ).hexdigest()
    answer = candidate["messages"][-1]["content"]
    assert answer.count("```python") == 1
    assert "assert add(2, 3)" in answer
    with pytest.raises(ValueError, match="verified=true"):
        verify_candidate(candidate)


def test_evaluation_prompts_use_independent_archetypes():
    assert set(EVALUATION_PROMPTS) == {spec.name for spec in FAMILY_SPECS}
    for spec in FAMILY_SPECS:
        evaluation = EVALUATION_PROMPTS[spec.name]
        assert set(spec.train_names).isdisjoint(spec.eval_names)
        assert len(evaluation) == 4
        assert len(set(evaluation)) == 4
        assert not set(evaluation) & set(spec.pt_prompts + spec.en_prompts)


def test_verification_evidence_is_tamper_evident():
    candidate = canonicalize_opencode_row(
        perfect_opencode_row(), token_counter=count_messages, execute=False
    )
    candidate["verified"] = True
    candidate["verification_evidence"]["status"] = "passed"
    with pytest.raises(ValueError, match="divergente"):
        verify_candidate(candidate)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"average_test_score": "0.9"}, "n\u00e3o \u00e9 1.0"),
        ({"tests_execution_status": repr(["pass", "fail"])}, "Nem todos"),
        ({"unit_tests": "not a list"}, "malformado"),
        ({"unit_tests": "[]"}, "n\u00e3o vazia"),
        ({"tests_execution_status": repr(["pass"])}, "tamanhos diferentes"),
    ],
)
def test_opencode_rejects_incomplete_or_malformed_rows(changes, message):
    with pytest.raises(ValueError, match=message):
        canonicalize_opencode_row(
            perfect_opencode_row(**changes),
            token_counter=count_messages,
            execute=False,
        )


def test_opencode_requires_all_three_judgements_at_five():
    row = perfect_opencode_row()
    judgement = json.loads(row["llm_judgement"])
    judgement["edge_case_consideration"]["score"] = 4
    row["llm_judgement"] = json.dumps(judgement)
    with pytest.raises(ValueError, match="edge_case_consideration"):
        canonicalize_opencode_row(row, token_counter=count_messages, execute=False)


def test_opencode_collection_rejects_rows_over_stage_limit():
    from qwen_sft.corrective import collect_opencode_candidates

    accepted, rejections = collect_opencode_candidates(
        [perfect_opencode_row()],
        token_counter=lambda messages: 2049,
        candidate_token_target=1,
        evaluation_tasks=[],
        execute=False,
        max_seq_length=2048,
    )
    assert accepted == []
    assert rejections["too_long>2048"] == 1


def test_generates_eleven_dev_and_thirty_three_hidden_without_leaking_tests():
    evaluation = generate_evaluation_tasks()
    assert len(FAMILY_SPECS) == 11
    assert len(evaluation["dev_v1"]) == 11
    assert len(evaluation["corrective_hidden_v1"]) == 33
    assert len({task["family"] for task in evaluation["dev_v1"]}) == 11
    assert all(task["hidden_tests"] for task in evaluation["corrective_hidden_v1"])
    assert all(
        isinstance(check, str) and "assert" in check
        for task in evaluation["corrective_hidden_v1"]
        for check in task["contract_checks"]
    )

    candidates = generate_contract_candidates(
        token_counter=count_messages,
        candidate_token_target=5_000,
        execute=False,
    )
    assert sum(item["num_tokens"] for item in candidates) >= 5_000
    assert all("hidden_tests" not in candidate for candidate in candidates)
    assert_not_contaminated(
        candidates,
        [*evaluation["dev_v1"], *evaluation["corrective_hidden_v1"]],
    )


@pytest.mark.skipif(os.name != "posix", reason="sandbox fail-closed requer POSIX")
def test_all_family_solutions_pass_and_known_defects_fail():
    broken = {
        "ordered_unique": "def {fn}(items):\n    return list(items)",
        "strict_parser": "def {fn}(lines):\n    return {}",
        "retry_identity": "def {fn}(operation, attempts, retry_on):\n    return operation()",
        "binary_boundary": "def {fn}(values, target):\n    return -1",
        "single_pass_chunks": "def {fn}(iterable, size):\n    if False:\n        yield []",
        "recursive_copy": "def {fn}(left, right):\n    return left",
        "stable_selection": "def {fn}(items, k, key):\n    return list(items)[:k]",
        "dependency_cycles": "def {fn}(graph):\n    return list(graph)",
        "memo_last": "def {fn}(function):\n    return function",
        "rolling_window": "def {fn}(values, window):\n    return []",
        "transactional_update": "def {fn}(state, updates, validator):\n    return state",
    }
    assert set(broken) == {spec.name for spec in FAMILY_SPECS}
    for spec in FAMILY_SPECS:
        fn = spec.eval_names[0]
        tests = "\n".join(
            (_format.replace("{fn}", fn) for _format in spec.public_tests + spec.hidden_tests)
        )
        correct = run_verified(spec.solution.replace("{fn}", fn), tests)
        defective = run_verified(broken[spec.name].replace("{fn}", fn), tests)
        assert correct.passed, spec.name
        assert not defective.passed, spec.name


def test_decontamination_rejects_ids_templates_hashes_and_near_prompts():
    task = {
        "id": "held-out",
        "template_id": "eval/family/0",
        "prompt": "Implement a parser that requires exactly one separator and rejects empty keys.",
    }
    candidate = {
        "id": "train",
        "template_id": "train/family/0",
        "messages": [
            {
                "role": "user",
                "content": "Implement a parser that requires exactly one separator and rejects empty keys.",
            }
        ],
    }
    with pytest.raises(ValueError, match="normalizado"):
        assert_not_contaminated([candidate], [task])


def test_contract_split_keeps_each_archetype_on_one_side():
    candidates = generate_contract_candidates(
        token_counter=count_messages,
        candidate_token_target=5_000,
        execute=False,
    )
    train, validation = split_by_group(
        candidates,
        validation_fraction=0.2,
        seed=42,
        min_validation_examples=11,
    )
    train_groups = {row["group_id"] for row in train}
    validation_groups = {row["group_id"] for row in validation}
    assert train_groups
    assert validation_groups
    assert train_groups.isdisjoint(validation_groups)


def test_replay_verifies_manifest_hashes_preserves_group_and_strips_think(tmp_path):
    run = tmp_path / "pilot"
    adapter = run / "adapter"
    data = run / "data"
    adapter.mkdir(parents=True)
    data.mkdir()
    train = data / "train.jsonl"
    validation = data / "validation.jsonl"
    report = data / "dataset_report.json"
    train.write_text(
        json.dumps(
            {
                "id": "pilot-row",
                "group_id": "original-group",
                "source": "nvidia/OpenCodeReasoning",
                "messages": [
                    {"role": "user", "content": "task"},
                    {"role": "assistant", "content": "<think>secret</think>\nanswer"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    validation.write_text("{}\n", encoding="utf-8")
    report.write_text("{}\n", encoding="utf-8")

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    (adapter / "run_manifest.json").write_text(
        json.dumps(
            {
                "data": {
                    "train_sha256": digest(train),
                    "validation_sha256": digest(validation),
                    "dataset_report_sha256": digest(report),
                }
            }
        ),
        encoding="utf-8",
    )
    rows = load_replay_candidates(run, token_counter=count_messages)
    assert rows[0]["group_id"] == "original-group"
    assert "<think>" not in rows[0]["messages"][-1]["content"]
    assert rows[0]["verification_level"] == "replay_champion"
    assert rows[0]["license"] == "cc-by-4.0"
    verify_candidate(rows[0])

    train.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="divergente"):
        load_replay_candidates(run, token_counter=count_messages)
