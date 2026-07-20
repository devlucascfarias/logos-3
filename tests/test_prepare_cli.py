import json
import hashlib
import subprocess
import sys
from pathlib import Path

from qwen_sft import sandbox
from qwen_sft.corrective import (
    OPENCODE_NAME,
    OPENCODE_REVISION,
    evidence_sha256,
    manifest_payload_sha256,
)
from qwen_sft.io import file_sha256


ROOT = Path(__file__).resolve().parents[1]


def signed_manifest(value):
    value = dict(value)
    value["manifest_payload_sha256"] = manifest_payload_sha256(value)
    return value


def test_prepare_cli_with_local_candidates(tmp_path):
    categories = (
        "verified_code",
        "reasoning",
        "swe",
        "claude_code",
        "general_tools",
    )
    bands = ("short", "medium", "long", "direct")
    candidate_path = tmp_path / "candidates.jsonl"
    rows = []
    for category in categories:
        for band in bands:
            for index in range(80):
                rows.append(
                    {
                        "id": f"{category}-{band}-{index}",
                        "group_id": f"group-{category}-{band}-{index}",
                        "source": "fixture",
                        "category": category,
                        "reasoning_band": band,
                        "verified": True,
                        "num_tokens": 100,
                        "fingerprint": f"fp-{category}-{band}-{index}",
                        "messages": [
                            {"role": "user", "content": "tarefa"},
                            {"role": "assistant", "content": "resposta"},
                        ],
                    }
                )
    candidate_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    output_dir = tmp_path / "processed"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "prepare_data.py"),
            "--config",
            str(ROOT / "configs" / "recipe.yaml"),
            "--stage",
            "baseline",
            "--token-budget",
            "20_000",
            "--candidates-jsonl",
            str(candidate_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(
        (output_dir / "dataset_report.json").read_text(encoding="utf-8")
    )
    assert (output_dir / "train.jsonl").exists()
    assert (output_dir / "validation.jsonl").exists()
    assert report["mix"]["budget_reached"]
    assert report["mix"]["budget_fraction"] >= 0.95
    assert report["mix"]["max_category_deviation"] <= 0.05
    assert report["mix"]["max_reasoning_deviation"] <= 0.05
    assert report["mix"]["candidate_matrix"]
    assert report["mix"]["selected_matrix"]
    assert report["split"]["validation_examples"] >= 32
    assert report["split"]["group_overlap"] is False


def test_prepare_cli_rejects_empty_training_set(tmp_path):
    candidate_path = tmp_path / "empty.jsonl"
    candidate_path.write_text("", encoding="utf-8")
    output_dir = tmp_path / "processed"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "prepare_data.py"),
            "--config",
            str(ROOT / "configs" / "recipe.yaml"),
            "--stage",
            "baseline",
            "--token-budget",
            "1000",
            "--candidates-jsonl",
            str(candidate_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Nenhum exemplo de treino foi selecionado" in (
        result.stdout + result.stderr
    )
    assert (output_dir / "dataset_report.json").exists()
    assert not (output_dir / "train.jsonl").exists()
    assert not (output_dir / "validation.jsonl").exists()


def test_prepare_corrective_requires_and_verifies_candidate_manifest(tmp_path):
    candidate_path = tmp_path / "corrective.jsonl"
    rows = []
    for category, level in (
        ("corrective_contracts", "local_hidden_tests"),
        ("corrective_opencode", "local_public_tests"),
        ("corrective_replay", "replay_champion"),
    ):
        for index in range(100):
            evidence = {
                "status": "passed",
                "runner_version": "test-runner-v1",
            }
            if level == "replay_champion":
                evidence["data_sha256"] = f"data-{index}"
            else:
                evidence["code_sha256"] = f"code-{index}"
                evidence["tests_sha256"] = f"tests-{index}"
            evidence["sha256"] = evidence_sha256(evidence)
            rows.append(
                {
                    "id": f"{category}-{index}",
                    "group_id": f"{category}-group-{index}",
                    "source": "fixture",
                    "source_sha256": hashlib.sha256(b"fixture").hexdigest(),
                    "source_revision": "local",
                    "revision_sha256": hashlib.sha256(b"local").hexdigest(),
                    "license": "cc-by-4.0",
                    "category": category,
                    "reasoning_band": "direct",
                    "verified": True,
                    "verification_level": level,
                    "verification_evidence": evidence,
                    "num_tokens": 100,
                    "fingerprint": f"fp-{category}-{index}",
                    "messages": [
                        {"role": "user", "content": "tarefa"},
                        {"role": "assistant", "content": "resposta"},
                    ],
                }
            )
    candidate_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    candidate_hash = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    evaluation_artifacts = {}
    for name, count in (
        ("dev_v1", 11),
        ("corrective_hidden_v1", 33),
        ("regression_v1", 13),
    ):
        evaluation_path = tmp_path / f"{name}.jsonl"
        evaluation_path.write_text("{}\n" * count, encoding="utf-8")
        evaluation_artifacts[name] = {
            "path": str(evaluation_path),
            "sha256": file_sha256(evaluation_path),
            "examples": count,
        }

    def corrective_manifest(candidates_sha256, production_eligible=True):
        return signed_manifest(
            {
                "stage": "corrective_v1",
                "seed": 42,
                "production_eligible": production_eligible,
                "source": {
                    "name": OPENCODE_NAME,
                    "name_sha256": hashlib.sha256(
                        OPENCODE_NAME.encode("utf-8")
                    ).hexdigest(),
                    "revision": OPENCODE_REVISION,
                    "revision_sha256": hashlib.sha256(
                        OPENCODE_REVISION.encode("utf-8")
                    ).hexdigest(),
                },
                "executor": {
                    "runner_version": sandbox.RUNNER_VERSION,
                    "sha256": file_sha256(sandbox.__file__),
                },
                "artifacts": {
                    "candidates": {
                        "path": str(candidate_path),
                        "sha256": candidates_sha256,
                        "examples": len(rows),
                        "tokens": 30_000,
                    },
                    **evaluation_artifacts,
                },
            }
        )

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(corrective_manifest(candidate_hash)),
        encoding="utf-8",
    )
    output_dir = tmp_path / "processed-corrective"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "prepare_data.py"),
            "--config",
            str(ROOT / "configs" / "recipe.yaml"),
            "--stage",
            "corrective_v1",
            "--token-budget",
            "6000",
            "--candidates-jsonl",
            str(candidate_path),
            "--candidate-manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(
        (output_dir / "dataset_report.json").read_text(encoding="utf-8")
    )
    assert report["effective_data_config"]["max_seq_length"] == 2048
    assert report["effective_data_config"]["candidate_oversample"] == 1.25
    assert report["effective_data_config"]["reasoning_distribution"] == {
        "direct": 1.0
    }
    assert report["mix"]["budget_fraction"] >= 0.95
    assert report["mix"]["max_category_deviation"] <= 0.05
    expected_mix = {
        "corrective_contracts": 0.50,
        "corrective_opencode": 0.35,
        "corrective_replay": 0.15,
    }
    for category, expected_fraction in expected_mix.items():
        actual = report["mix"]["category_distribution"][category]["fraction"]
        assert abs(actual - expected_fraction) <= 0.05
    assert (
        report["collection"]["candidate_manifest"]["candidates_sha256"]
        == candidate_hash
    )

    manifest_path.write_text(
        json.dumps(corrective_manifest(candidate_hash, False)),
        encoding="utf-8",
    )
    unverified = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "prepare_data.py"),
            "--config",
            str(ROOT / "configs" / "recipe.yaml"),
            "--stage",
            "corrective_v1",
            "--token-budget",
            "6000",
            "--candidates-jsonl",
            str(candidate_path),
            "--candidate-manifest",
            str(manifest_path),
            "--output-dir",
            str(tmp_path / "unverified"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert unverified.returncode != 0
    assert "não é elegível para produção" in (
        unverified.stdout + unverified.stderr
    )

    tampered_manifest = corrective_manifest(candidate_hash)
    tampered_manifest["seed"] = 41
    manifest_path.write_text(json.dumps(tampered_manifest), encoding="utf-8")
    tampered = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "prepare_data.py"),
            "--config",
            str(ROOT / "configs" / "recipe.yaml"),
            "--stage",
            "corrective_v1",
            "--token-budget",
            "6000",
            "--candidates-jsonl",
            str(candidate_path),
            "--candidate-manifest",
            str(manifest_path),
            "--output-dir",
            str(tmp_path / "tampered"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert tampered.returncode != 0
    assert "Hash interno do manifesto" in tampered.stdout + tampered.stderr

    manifest_path.write_text(
        json.dumps(corrective_manifest("0" * 64)),
        encoding="utf-8",
    )
    rejected = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "prepare_data.py"),
            "--config",
            str(ROOT / "configs" / "recipe.yaml"),
            "--stage",
            "corrective_v1",
            "--token-budget",
            "6000",
            "--candidates-jsonl",
            str(candidate_path),
            "--candidate-manifest",
            str(manifest_path),
            "--output-dir",
            str(tmp_path / "rejected"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "não corresponde ao hash" in rejected.stdout + rejected.stderr
