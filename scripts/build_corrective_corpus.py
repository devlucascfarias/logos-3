from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from qwen_sft.config import load_config, merged_data_config
from qwen_sft.corrective import (
    OPENCODE_NAME,
    OPENCODE_REVISION,
    assert_not_contaminated,
    build_manifest,
    collect_opencode_candidates,
    generate_contract_candidates,
    generate_evaluation_tasks,
    load_replay_candidates,
    manifest_payload_sha256,
    verify_candidate,
)
from qwen_sft.io import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materializa e assina os candidatos verificados de corrective_v1."
    )
    parser.add_argument("--config", default="configs/recipe.yaml")
    parser.add_argument(
        "--pilot-run-path",
        required=True,
        help="C\u00f3pia local do diret\u00f3rio pilot_500k_step7 preservado.",
    )
    parser.add_argument(
        "--output-dir", default="data/interim/corrective_v1"
    )
    parser.add_argument(
        "--opencode-jsonl",
        help="Fixture JSONL local; se omitido, usa streaming no Hub.",
    )
    parser.add_argument("--max-opencode-rows", type=int)
    parser.add_argument("--contracts-candidate-tokens", type=int, default=200_000)
    parser.add_argument("--opencode-candidate-tokens", type=int, default=140_000)
    parser.add_argument("--replay-candidate-tokens", type=int, default=60_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-execution",
        action="store_true",
        help="Somente para testes do gerador; manifestos de produ\u00e7\u00e3o n\u00e3o devem usar esta op\u00e7\u00e3o.",
    )
    return parser.parse_args()


def _encoded_token_count(encoded: Any) -> int:
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    return len(encoded)


def _token_counter(model_name: str, revision: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Instale as depend\u00eancias: pip install -r requirements-colab.txt"
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=revision, use_fast=True
    )
    if not getattr(tokenizer, "chat_template", None):
        raise SystemExit("O tokenizer n\u00e3o possui chat template.")

    def count(messages):
        return _encoded_token_count(
            tokenizer.apply_chat_template(
                list(messages),
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        )

    return count


def _opencode_rows(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    if args.opencode_jsonl:
        rows: Iterable[dict[str, Any]] = read_jsonl(args.opencode_jsonl)
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise SystemExit(
                "Instale as depend\u00eancias: pip install -r requirements-colab.txt"
            ) from exc
        stream = load_dataset(
            OPENCODE_NAME,
            "train",
            split="train",
            streaming=True,
            revision=OPENCODE_REVISION,
        )
        rows = stream.shuffle(seed=args.seed, buffer_size=10_000)
    if args.max_opencode_rows is None:
        return rows

    def limited():
        for index, row in enumerate(rows):
            if index >= args.max_opencode_rows:
                break
            yield dict(row)

    return limited()


def _take_token_target(
    rows: list[dict[str, Any]], target: int, seed: int
) -> list[dict[str, Any]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    selected: list[dict[str, Any]] = []
    tokens = 0
    for row in shuffled:
        if tokens >= target:
            break
        selected.append(row)
        tokens += int(row["num_tokens"])
    if tokens < target:
        raise RuntimeError(
            f"Replay insuficiente: {tokens:,} de {target:,} tokens candidatos."
        )
    return selected


def _regression_rows(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or len(value) != 13:
        raise ValueError("regression_v1 deve conter os 13 prompts preservados.")
    return [
        {
            "id": str(item["id"]),
            "template_id": f"regression/{item['id']}",
            "title": str(item["title"]),
            "prompt": str(item["prompt"]),
        }
        for item in value
    ]


def main() -> None:
    args = parse_args()
    if args.seed != 42:
        raise SystemExit("corrective_v1 usa seed fixa 42.")
    if args.skip_execution:
        print("AVISO: --skip-execution destina-se exclusivamente a testes locais.")
    for label in (
        "contracts_candidate_tokens",
        "opencode_candidate_tokens",
        "replay_candidate_tokens",
    ):
        if getattr(args, label) <= 0:
            raise SystemExit(f"--{label.replace('_', '-')} deve ser positivo.")

    config = load_config(args.config)
    data_config = merged_data_config(config, "corrective_v1")
    max_seq_length = int(data_config["max_seq_length"])
    count_tokens = _token_counter(
        config["model_name"], config.get("model_revision", "main")
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation = generate_evaluation_tasks()
    regression = _regression_rows(Path("examples/smoke_eval_prompts.json"))
    all_evaluation = [
        *evaluation["dev_v1"],
        *evaluation["corrective_hidden_v1"],
        *regression,
    ]

    print("[1/5] Gerando e executando microcontratos locais...")
    contracts = generate_contract_candidates(
        token_counter=count_tokens,
        candidate_token_target=args.contracts_candidate_tokens,
        seed=args.seed,
        execute=not args.skip_execution,
        max_seq_length=max_seq_length,
    )

    print("[2/5] Filtrando e reexecutando OpenCodeInstruct...")
    opencode, rejections = collect_opencode_candidates(
        _opencode_rows(args),
        token_counter=count_tokens,
        candidate_token_target=args.opencode_candidate_tokens,
        evaluation_tasks=all_evaluation,
        execute=not args.skip_execution,
        max_seq_length=max_seq_length,
    )
    opencode_tokens = sum(int(item["num_tokens"]) for item in opencode)
    if opencode_tokens < args.opencode_candidate_tokens:
        raise SystemExit(
            "OpenCodeInstruct n\u00e3o atingiu o alvo de candidatos: "
            f"{opencode_tokens:,}/{args.opencode_candidate_tokens:,}. "
            f"Rejei\u00e7\u00f5es: {json.dumps(rejections, ensure_ascii=False)}"
        )

    print("[3/5] Verificando hashes e materializando replay do campe\u00e3o...")
    replay = _take_token_target(
        load_replay_candidates(
            args.pilot_run_path,
            token_counter=count_tokens,
            max_seq_length=max_seq_length,
        ),
        args.replay_candidate_tokens,
        args.seed,
    )
    candidates = [*contracts, *opencode, *replay]
    if not args.skip_execution:
        for candidate in candidates:
            verify_candidate(candidate)
    assert_not_contaminated(candidates, all_evaluation)
    random.Random(args.seed).shuffle(candidates)

    print("[4/5] Gravando candidatos e avalia\u00e7\u00f5es separadas...")
    candidates_path = output_dir / "candidates.jsonl"
    write_jsonl(candidates_path, candidates)
    evaluation_paths: dict[str, Path] = {}
    for name, rows in evaluation.items():
        path = output_dir / f"{name}.jsonl"
        write_jsonl(path, rows)
        evaluation_paths[name] = path
    regression_path = output_dir / "regression_v1.jsonl"
    write_jsonl(
        regression_path,
        regression,
    )
    evaluation_paths["regression_v1"] = regression_path

    print("[5/5] Assinando manifesto...")
    manifest = build_manifest(
        candidates_path=candidates_path,
        candidates=candidates,
        evaluation_paths=evaluation_paths,
        config={
            "contracts_candidate_tokens": args.contracts_candidate_tokens,
            "opencode_candidate_tokens": args.opencode_candidate_tokens,
            "replay_candidate_tokens": args.replay_candidate_tokens,
            "opencode_rejections": rejections,
            "skip_execution": args.skip_execution,
            "max_seq_length": max_seq_length,
        },
        allow_unverified=args.skip_execution,
    )
    if args.skip_execution:
        manifest["production_eligible"] = False
    else:
        manifest["production_eligible"] = True
    manifest["manifest_payload_sha256"] = manifest_payload_sha256(manifest)
    manifest_path = output_dir / "candidate_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
    print(f"Candidatos: {candidates_path}")
    print(f"Manifesto: {manifest_path}")


if __name__ == "__main__":
    main()
