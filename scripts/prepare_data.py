from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from qwen_sft.config import load_config, merged_data_config
from qwen_sft import sandbox
from qwen_sft.corrective import (
    OPENCODE_NAME,
    OPENCODE_REVISION,
    manifest_payload_sha256,
    verify_candidate,
)
from qwen_sft.data import (
    build_candidates,
    load_holdout_ids,
    mix_by_tokens,
    split_by_group,
    training_row,
)
from qwen_sft.io import file_sha256, read_jsonl, write_jsonl


CORRECTIVE_VERIFICATION_LEVELS = {
    "local_hidden_tests",
    "local_public_tests",
    "replay_champion",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepara, filtra e mistura o corpus SFT por tokens."
    )
    parser.add_argument("--config", default="configs/recipe.yaml")
    parser.add_argument(
        "--stage",
        choices=(
            "corrective_v1",
            "pilot",
            "pilot_continuation",
            "baseline",
            "main",
            "agentic",
        ),
        default="pilot",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        help="Sobrescreve o orçamento da etapa (útil para smoke tests).",
    )
    parser.add_argument(
        "--max-source-rows",
        type=int,
        help="Limite de linhas examinadas por fonte.",
    )
    parser.add_argument(
        "--candidates-jsonl",
        help="Usa candidatos canônicos locais e não acessa o Hugging Face Hub.",
    )
    parser.add_argument(
        "--candidate-manifest",
        help=(
            "Manifesto assinado dos candidatos locais. Obrigatório para "
            "corrective_v1."
        ),
    )
    parser.add_argument("--output-dir", help="Padrão: data/processed/<stage>.")
    return parser.parse_args()


def _encoded_token_count(encoded: Any) -> int:
    if isinstance(encoded, Mapping):
        if "input_ids" not in encoded:
            raise ValueError("A tokenização não retornou input_ids.")
        encoded = encoded["input_ids"]
    return len(encoded)


def _tokenizer_functions(model_name: str, revision: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Instale as dependências: pip install -r requirements-colab.txt"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=revision, use_fast=True
    )
    if not getattr(tokenizer, "chat_template", None):
        raise SystemExit("O tokenizer não possui o chat template oficial do Qwen3.")

    def count_messages(messages):
        encoded = tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=True,
        )
        return _encoded_token_count(encoded)

    def count_text(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    return count_messages, count_text


def _buffered_shuffle(
    rows: Iterable[dict[str, Any]], seed: int, buffer_size: int
) -> Iterator[dict[str, Any]]:
    if buffer_size <= 1:
        yield from rows
        return

    rng = random.Random(seed)
    buffer: list[dict[str, Any]] = []
    for row in rows:
        if len(buffer) < buffer_size:
            buffer.append(row)
            continue
        index = rng.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = row

    while buffer:
        yield buffer.pop(rng.randrange(len(buffer)))


def _load_raw_jsonl_stream(
    source: dict[str, Any], seed: int, shuffle_buffer: int
) -> Iterator[dict[str, Any]]:
    try:
        from huggingface_hub import HfFileSystem
    except ImportError as exc:
        raise SystemExit(
            "Instale as dependências: pip install -r requirements-colab.txt"
        ) from exc

    revision = source.get("revision", "main")
    remote_path = (
        f"datasets/{source['name']}@{revision}/{source['raw_jsonl_file']}"
    )
    token = os.environ.get("HF_TOKEN")
    filesystem = HfFileSystem(token=token)

    def rows() -> Iterator[dict[str, Any]]:
        with filesystem.open(remote_path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"JSON inválido em {remote_path}:{line_number}"
                    ) from exc
                if not isinstance(row, dict):
                    raise RuntimeError(
                        f"Linha não é um objeto JSON em "
                        f"{remote_path}:{line_number}"
                    )
                yield row

    source_buffer = int(source.get("shuffle_buffer", shuffle_buffer))
    return _buffered_shuffle(rows(), seed=seed, buffer_size=source_buffer)


def _load_stream(source: dict[str, Any], seed: int, shuffle_buffer: int):
    if source.get("raw_jsonl_file"):
        return _load_raw_jsonl_stream(source, seed, shuffle_buffer)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Instale as dependências: pip install -r requirements-colab.txt"
        ) from exc

    kwargs: dict[str, Any] = {
        "split": source.get("split", "train"),
        "streaming": True,
        "revision": source.get("revision", "main"),
    }
    token = os.environ.get("HF_TOKEN")
    if token:
        kwargs["token"] = token
    config_name = source.get("config_name")
    stream = load_dataset(source["name"], config_name, **kwargs)
    if hasattr(stream, "shuffle"):
        stream = stream.shuffle(seed=seed, buffer_size=shuffle_buffer)
    return stream


def _resolve_source_revision(source: dict[str, Any]) -> dict[str, Any]:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return source
    resolved = dict(source)
    token = os.environ.get("HF_TOKEN")
    info = HfApi(token=token).dataset_info(
        source["name"], revision=source.get("revision", "main")
    )
    resolved["requested_revision"] = source.get("revision", "main")
    resolved["revision"] = info.sha
    return resolved


def _collect_remote_candidates(
    config: dict[str, Any],
    data_config: dict[str, Any],
    stage_name: str,
    token_budget: int,
    max_source_rows: int | None,
    count_messages,
    count_text,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seed = int(config.get("seed", 42))
    stage = config["stages"][stage_name]
    active_categories = set(stage["category_weights"])
    oversample = float(data_config.get("candidate_oversample", 1.35))
    holdout_ids = load_holdout_ids(data_config.get("holdout_ids_file"))
    successful_only = bool(stage.get("successful_trajectories_only", False))

    candidates: list[dict[str, Any]] = []
    seen_fingerprints: set[str] = set()
    seen_near_fingerprints: set[str] = set()
    group_counts: Counter[str] = Counter()
    max_examples_per_group = int(
        data_config.get("max_examples_per_group", 0)
    )
    source_reports: list[dict[str, Any]] = []
    for source_index, configured_source in enumerate(config["sources"]):
        source = dict(configured_source)
        category = source["category"]
        if category not in active_categories:
            continue
        source = _resolve_source_revision(source)
        source_target = round(
            token_budget
            * float(stage["category_weights"][category])
            * float(source.get("source_weight", 1.0))
            * oversample
        )
        accepted_tokens = 0
        accepted_examples = 0
        accepted_groups: set[str] = set()
        scanned_rows = 0
        rejections: Counter[str] = Counter()
        print(
            f"\n[{source['name']}] alvo de candidatos: "
            f"{source_target:,} tokens"
        )
        try:
            stream = _load_stream(
                source,
                seed + source_index,
                int(data_config.get("shuffle_buffer", 10000)),
            )
            for row_index, row in enumerate(stream):
                if max_source_rows is not None and scanned_rows >= max_source_rows:
                    break
                if accepted_tokens >= source_target:
                    break
                scanned_rows += 1
                built, reason = build_candidates(
                    dict(row),
                    row_index=row_index,
                    source=source,
                    system_prompt=config["system_prompt"],
                    data_config=data_config,
                    token_counter=count_messages,
                    text_token_counter=count_text,
                    successful_only=successful_only,
                    holdout_ids=holdout_ids,
                )
                if not built:
                    rejections[reason or "unknown"] += 1
                    continue
                for example in built:
                    if example["fingerprint"] in seen_fingerprints:
                        rejections["duplicate"] += 1
                        continue
                    near_fingerprint = str(
                        example.get(
                            "near_fingerprint", example["fingerprint"]
                        )
                    )
                    if near_fingerprint in seen_near_fingerprints:
                        rejections["near_duplicate"] += 1
                        continue
                    group_id = str(example["group_id"])
                    if (
                        max_examples_per_group > 0
                        and group_counts[group_id] >= max_examples_per_group
                    ):
                        rejections["group_limit"] += 1
                        continue
                    seen_fingerprints.add(example["fingerprint"])
                    seen_near_fingerprints.add(near_fingerprint)
                    group_counts[group_id] += 1
                    accepted_groups.add(group_id)
                    candidates.append(example)
                    accepted_tokens += int(example["num_tokens"])
                    accepted_examples += 1
        except Exception as exc:
            raise RuntimeError(
                f"Falha ao processar a fonte {source['name']}: {exc}"
            ) from exc

        report = {
            "source": source["name"],
            "requested_revision": source.get("requested_revision", "main"),
            "resolved_revision": source.get("revision", "main"),
            "category": category,
            "target_candidate_tokens": source_target,
            "accepted_tokens": accepted_tokens,
            "accepted_examples": accepted_examples,
            "accepted_groups": len(accepted_groups),
            "scanned_rows": scanned_rows,
            "target_reached": accepted_tokens >= source_target,
            "rejections": dict(sorted(rejections.items())),
        }
        source_reports.append(report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return candidates, {"sources": source_reports}


def _candidate_artifact(manifest: dict[str, Any]) -> dict[str, Any]:
    artifact = manifest.get("artifacts", {}).get("candidates", {})
    if isinstance(artifact, dict) and artifact:
        return artifact
    return {
        "path": manifest.get("candidates_path"),
        "sha256": manifest.get("candidates_sha256"),
        "examples": manifest.get("candidate_examples"),
        "tokens": manifest.get("candidate_tokens"),
    }


def _load_candidate_manifest(
    manifest_path: str | Path,
    candidate_path: str | Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    path = Path(manifest_path)
    if not path.exists():
        raise SystemExit(f"Manifesto de candidatos ausente: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise SystemExit(f"Manifesto de candidatos inválido: {path}") from exc
    if not isinstance(manifest, dict):
        raise SystemExit("O manifesto de candidatos deve ser um objeto JSON.")
    if manifest.get("production_eligible") is not True:
        raise SystemExit(
            "O manifesto de candidatos não é elegível para produção; "
            "execute o builder sem --skip-execution."
        )
    expected_payload_hash = str(
        manifest.get("manifest_payload_sha256") or ""
    ).strip()
    if not expected_payload_hash or expected_payload_hash != manifest_payload_sha256(
        manifest
    ):
        raise SystemExit("Hash interno do manifesto de candidatos divergente.")
    if manifest.get("stage") != "corrective_v1" or manifest.get("seed") != 42:
        raise SystemExit("Stage ou seed inválidos no manifesto corrective_v1.")
    source = manifest.get("source")
    if not isinstance(source, dict) or (
        source.get("name") != OPENCODE_NAME
        or source.get("revision") != OPENCODE_REVISION
        or source.get("name_sha256")
        != hashlib.sha256(OPENCODE_NAME.encode("utf-8")).hexdigest()
        or source.get("revision_sha256")
        != hashlib.sha256(OPENCODE_REVISION.encode("utf-8")).hexdigest()
    ):
        raise SystemExit("Fonte ou revisão do OpenCode divergente no manifesto.")
    executor = manifest.get("executor")
    if not isinstance(executor, dict) or (
        executor.get("runner_version") != sandbox.RUNNER_VERSION
        or executor.get("sha256") != file_sha256(sandbox.__file__)
    ):
        raise SystemExit("Executor do manifesto diverge do executor local.")
    verified_artifacts: dict[str, dict[str, Any]] = {}
    for name in ("dev_v1", "corrective_hidden_v1", "regression_v1"):
        evaluation_artifact = manifest.get("artifacts", {}).get(name)
        if not isinstance(evaluation_artifact, dict):
            raise SystemExit(f"Artefato de avaliação ausente no manifesto: {name}")
        evaluation_path = Path(str(evaluation_artifact.get("path", "")))
        expected_evaluation_hash = str(
            evaluation_artifact.get("sha256") or ""
        )
        if (
            not evaluation_path.exists()
            or not expected_evaluation_hash
            or file_sha256(evaluation_path) != expected_evaluation_hash
        ):
            raise SystemExit(f"Hash do artefato de avaliação divergente: {name}")
        expected_count = int(evaluation_artifact.get("examples", -1))
        actual_count = sum(1 for _ in read_jsonl(evaluation_path))
        if expected_count != actual_count:
            raise SystemExit(f"Contagem do artefato de avaliação divergente: {name}")
        verified_artifacts[name] = {
            "path": str(evaluation_path),
            "sha256": expected_evaluation_hash,
            "examples": actual_count,
        }
    artifact = _candidate_artifact(manifest)
    expected_hash = str(artifact.get("sha256") or "").strip()
    if not expected_hash:
        raise SystemExit("Hash candidates_sha256 ausente no manifesto.")
    actual_hash = file_sha256(candidate_path)
    if actual_hash != expected_hash:
        raise SystemExit(
            "O JSONL de candidatos não corresponde ao hash do manifesto: "
            f"{candidate_path}"
        )
    expected_examples = artifact.get("examples")
    if expected_examples is not None and int(expected_examples) != len(rows):
        raise SystemExit(
            "A contagem de candidatos não corresponde ao manifesto "
            f"({len(rows)} != {expected_examples})."
        )
    actual_tokens = sum(int(row.get("num_tokens", 0)) for row in rows)
    expected_tokens = artifact.get("tokens")
    if expected_tokens is not None and int(expected_tokens) != actual_tokens:
        raise SystemExit(
            "A contagem de tokens dos candidatos não corresponde ao manifesto "
            f"({actual_tokens} != {expected_tokens})."
        )
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "candidates_path": str(candidate_path),
        "candidates_sha256": actual_hash,
        "examples": len(rows),
        "tokens": actual_tokens,
        "evaluation_artifacts": verified_artifacts,
    }


def _validate_corrective_candidate(
    row: dict[str, Any], *, max_seq_length: int
) -> None:
    candidate_id = str(row.get("id", "<sem-id>"))
    forbidden_fields = {"hidden_tests", "tests_hidden", "contract_checks"}
    leaked_fields = forbidden_fields & set(row)
    if leaked_fields:
        raise SystemExit(
            f"Testes ocultos presentes no candidato {candidate_id}: "
            f"{sorted(leaked_fields)}"
        )
    try:
        verify_candidate(row)
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(
            f"Evidência corretiva inválida em {candidate_id}: {exc}"
        ) from exc
    if row.get("verified") is not True:
        raise SystemExit(
            f"Candidato corretivo não verificado: {candidate_id}"
        )
    level = str(row.get("verification_level", ""))
    if level not in CORRECTIVE_VERIFICATION_LEVELS:
        raise SystemExit(
            f"verification_level inválido em {candidate_id}: {level or '<vazio>'}"
        )
    evidence = row.get("verification_evidence")
    if not isinstance(evidence, dict):
        raise SystemExit(
            f"verification_evidence ausente em {candidate_id}."
        )
    required = ("sha256", "runner_version")
    if evidence.get("status") != "passed" or any(
        not str(evidence.get(key, "")).strip() for key in required
    ):
        raise SystemExit(
            f"Evidência de execução incompleta em {candidate_id}."
        )
    if level == "replay_champion":
        if not str(evidence.get("data_sha256", "")).strip():
            raise SystemExit(
                f"data_sha256 do replay ausente em {candidate_id}."
            )
    elif not str(evidence.get("code_sha256", "")).strip() or not str(
        evidence.get("tests_sha256", "")
    ).strip():
        raise SystemExit(
            f"Hashes de código/testes ausentes em {candidate_id}."
        )
    num_tokens = int(row.get("num_tokens", 0))
    if num_tokens <= 0 or num_tokens > max_seq_length:
        raise SystemExit(
            f"Candidato {candidate_id} possui {num_tokens} tokens; limite da "
            f"etapa: {max_seq_length}."
        )
    if row.get("reasoning_band") != "direct":
        raise SystemExit(
            f"Candidato {candidate_id} não está no bucket direct."
        )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    stage = config["stages"][args.stage]
    data_config = merged_data_config(config, args.stage)
    token_budget = int(args.token_budget or stage["token_budget"])
    if token_budget <= 0:
        raise SystemExit("--token-budget deve ser positivo.")

    if args.stage == "corrective_v1" and (
        not args.candidates_jsonl or not args.candidate_manifest
    ):
        raise SystemExit(
            "corrective_v1 exige --candidates-jsonl e --candidate-manifest; "
            "coleta remota direta não é permitida."
        )
    if args.candidate_manifest and not args.candidates_jsonl:
        raise SystemExit("--candidate-manifest exige --candidates-jsonl.")

    if args.candidates_jsonl:
        active_categories = set(stage["category_weights"])
        local_rows = list(read_jsonl(args.candidates_jsonl))
        manifest_report = (
            _load_candidate_manifest(
                args.candidate_manifest, args.candidates_jsonl, local_rows
            )
            if args.candidate_manifest
            else None
        )
        candidates = []
        rejected_categories: Counter[str] = Counter()
        for example in local_rows:
            category = str(example.get("category", ""))
            if category not in active_categories:
                if args.stage == "corrective_v1":
                    raise SystemExit(
                        f"Categoria inesperada no corpus corrective_v1: "
                        f"{category or '<vazia>'}"
                    )
                rejected_categories[category or "<vazia>"] += 1
                continue
            if args.stage == "corrective_v1":
                _validate_corrective_candidate(
                    example,
                    max_seq_length=int(data_config["max_seq_length"]),
                )
            candidates.append(example)
        collection_report = {
            "local_candidates": args.candidates_jsonl,
            "candidate_manifest": manifest_report,
            "accepted_candidates": len(candidates),
            "rejected_categories": dict(sorted(rejected_categories.items())),
        }
    else:
        count_messages, count_text = _tokenizer_functions(
            config["model_name"], config.get("model_revision", "main")
        )
        candidates, collection_report = _collect_remote_candidates(
            config,
            data_config,
            args.stage,
            token_budget,
            args.max_source_rows,
            count_messages,
            count_text,
        )

    selected, mix_report = mix_by_tokens(
        candidates,
        token_budget=token_budget,
        category_weights={
            key: float(value) for key, value in stage["category_weights"].items()
        },
        reasoning_weights={
            key: float(value)
            for key, value in data_config["reasoning_distribution"].items()
        },
        seed=int(config.get("seed", 42)),
        max_examples_per_group=int(
            data_config.get("max_examples_per_group", 0)
        ),
    )
    train, validation = split_by_group(
        selected,
        validation_fraction=float(data_config.get("validation_fraction", 0.02)),
        seed=int(config.get("seed", 42)),
        min_validation_examples=int(
            data_config.get("validation_min_examples", 0)
        ),
    )

    output_dir = Path(args.output_dir or f"data/processed/{args.stage}")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    report = {
        "stage": args.stage,
        "config": str(Path(args.config)),
        "effective_data_config": {
            "max_seq_length": int(data_config["max_seq_length"]),
            "candidate_oversample": float(
                data_config.get("candidate_oversample", 1.0)
            ),
            "reasoning_distribution": data_config["reasoning_distribution"],
            "max_examples_per_group": int(
                data_config.get("max_examples_per_group", 0)
            ),
        },
        "collection": collection_report,
        "mix": mix_report,
        "split": {
            "train_examples": len(train),
            "train_tokens": sum(int(item["num_tokens"]) for item in train),
            "validation_examples": len(validation),
            "validation_tokens": sum(
                int(item["num_tokens"]) for item in validation
            ),
            "group_overlap": bool(
                {item["group_id"] for item in train}
                & {item["group_id"] for item in validation}
            ),
        },
    }
    report_path = output_dir / "dataset_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not train:
        for stale_path in (train_path, validation_path):
            if stale_path.exists():
                stale_path.unlink()
        raise SystemExit(
            "Nenhum exemplo de treino foi selecionado. Consulte "
            f"{report_path} para ver as rejeições por fonte."
        )

    max_category_deviation = float(
        data_config.get("max_category_deviation", 1.0)
    )
    if mix_report["max_category_deviation"] > max_category_deviation:
        for stale_path in (train_path, validation_path):
            if stale_path.exists():
                stale_path.unlink()
        raise SystemExit(
            "A distribuição por categoria excedeu o desvio permitido "
            f"({mix_report['max_category_deviation']:.1%} > "
            f"{max_category_deviation:.1%}). Consulte a matriz de candidatos "
            f"em {report_path}."
        )

    max_reasoning_deviation = float(
        data_config.get("max_reasoning_deviation", 1.0)
    )
    if mix_report["max_reasoning_deviation"] > max_reasoning_deviation:
        for stale_path in (train_path, validation_path):
            if stale_path.exists():
                stale_path.unlink()
        raise SystemExit(
            "A distribuição de raciocínio excedeu o desvio permitido "
            f"({mix_report['max_reasoning_deviation']:.1%} > "
            f"{max_reasoning_deviation:.1%}). Consulte "
            f"{report_path} e aumente a diversidade dos candidatos."
        )

    min_budget_fraction = float(
        data_config.get("min_token_budget_fraction", 0.0)
    )
    if mix_report["budget_fraction"] < min_budget_fraction:
        for stale_path in (train_path, validation_path):
            if stale_path.exists():
                stale_path.unlink()
        raise SystemExit(
            "O mixer não atingiu a cobertura mínima do orçamento "
            f"({mix_report['budget_fraction']:.1%} < "
            f"{min_budget_fraction:.1%}). Consulte {report_path} e aumente "
            "a varredura de candidatos."
        )

    write_jsonl(train_path, (training_row(example) for example in train))
    write_jsonl(
        validation_path, (training_row(example) for example in validation)
    )
    print(f"\nTreino: {train_path} ({len(train):,} exemplos)")
    print(f"Validação: {validation_path} ({len(validation):,} exemplos)")
    print(f"Relatório: {output_dir / 'dataset_report.json'}")
    if not mix_report["budget_reached"]:
        print(
            "AVISO: o conjunto de candidatos não atingiu o orçamento solicitado; "
            "consulte o relatório de rejeições e aumente a varredura."
        )


if __name__ == "__main__":
    main()
