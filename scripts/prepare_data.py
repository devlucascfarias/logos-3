from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from qwen_sft.config import load_config
from qwen_sft.data import (
    build_candidates,
    load_holdout_ids,
    mix_by_tokens,
    split_by_group,
    training_row,
)
from qwen_sft.io import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepara, filtra e mistura o corpus SFT por tokens."
    )
    parser.add_argument("--config", default="configs/recipe.yaml")
    parser.add_argument(
        "--stage",
        choices=("pilot", "baseline", "main", "agentic"),
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
    stage_name: str,
    token_budget: int,
    max_source_rows: int | None,
    count_messages,
    count_text,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seed = int(config.get("seed", 42))
    data_config = config["data"]
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


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    stage = config["stages"][args.stage]
    token_budget = int(args.token_budget or stage["token_budget"])
    if token_budget <= 0:
        raise SystemExit("--token-budget deve ser positivo.")

    if args.candidates_jsonl:
        active_categories = set(stage["category_weights"])
        candidates = [
            example
            for example in read_jsonl(args.candidates_jsonl)
            if example.get("category") in active_categories
        ]
        collection_report = {"local_candidates": args.candidates_jsonl}
    else:
        count_messages, count_text = _tokenizer_functions(
            config["model_name"], config.get("model_revision", "main")
        )
        candidates, collection_report = _collect_remote_candidates(
            config,
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
            for key, value in config["data"]["reasoning_distribution"].items()
        },
        seed=int(config.get("seed", 42)),
        max_examples_per_group=int(
            config["data"].get("max_examples_per_group", 0)
        ),
    )
    train, validation = split_by_group(
        selected,
        validation_fraction=float(config["data"].get("validation_fraction", 0.02)),
        seed=int(config.get("seed", 42)),
        min_validation_examples=int(
            config["data"].get("validation_min_examples", 0)
        ),
    )

    output_dir = Path(args.output_dir or f"data/processed/{args.stage}")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    report = {
        "stage": args.stage,
        "config": str(Path(args.config)),
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
        config["data"].get("max_category_deviation", 1.0)
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
        config["data"].get("max_reasoning_deviation", 1.0)
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
        config["data"].get("min_token_budget_fraction", 0.0)
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
