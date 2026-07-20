from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from qwen_sft.contract_eval import (
    aggregate_manual_ratings,
    build_contract_report,
    promotion_gate,
)
from qwen_sft.io import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Executa testes ocultos em previsoes corrective_v1, ranqueia "
            "checkpoints e aplica opcionalmente o gate de promocao."
        )
    )
    parser.add_argument("--tasks", required=True, help="JSONL dev_v1 ou corrective_hidden_v1.")
    parser.add_argument(
        "--predictions",
        nargs="+",
        required=True,
        help="Um ou mais JSONL produzidos por generate_evaluation.py.",
    )
    parser.add_argument("--output", required=True, help="Relatorio JSON de avaliacao.")
    parser.add_argument(
        "--eval-loss",
        action="append",
        default=[],
        metavar="CHECKPOINT=VALUE",
        help="Eval loss usada somente no ultimo desempate; pode ser repetida.",
    )
    parser.add_argument("--promotion-output", help="Saida JSON do gate de promocao.")
    parser.add_argument("--ratings", help="ratings.json da regressao cega de 13 tarefas.")
    parser.add_argument("--mapping", help="mapping.json da regressao cega.")
    parser.add_argument("--candidate", help="Identidade/checkpoint candidato no relatorio.")
    parser.add_argument("--champion", help="Identidade/checkpoint campeao no relatorio.")
    parser.add_argument("--base", default="base", help="Identidade do base no mapping.")
    parser.add_argument(
        "--candidate-rating-identity",
        default="adapter",
        help="Identidade do candidato no mapping da regressao cega.",
    )
    parser.add_argument(
        "--base-rating-identity",
        default="base",
        help="Identidade do modelo-base no mapping da regressao cega.",
    )
    return parser.parse_args()


def _parse_eval_losses(values: list[str]) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for value in values:
        checkpoint, separator, raw_loss = value.rpartition("=")
        if not separator or not checkpoint:
            raise ValueError(f"--eval-loss invalido: {value!r}; use CHECKPOINT=VALUE.")
        loss = float(raw_loss)
        if loss < 0 or not math.isfinite(loss):
            raise ValueError(f"Eval loss invalida para {checkpoint!r}.")
        parsed[checkpoint] = loss
    return parsed


def _sandbox_runner(code: str, test: str) -> Any:
    try:
        from qwen_sft.sandbox import run_verified
    except ImportError as exc:
        raise RuntimeError(
            "Executor restrito indisponivel: qwen_sft.sandbox.run_verified."
        ) from exc
    return run_verified(
        code,
        test,
        timeout_seconds=3.0,
        cpu_seconds=2,
        memory_mib=256,
        output_limit=64 * 1024,
    )


def _sandbox_metadata() -> dict[str, Any]:
    from qwen_sft import sandbox

    return {
        "runner_version": sandbox.RUNNER_VERSION,
        "source_sha256": file_sha256(sandbox.__file__),
        "timeout_seconds": 3.0,
        "cpu_seconds": 2,
        "memory_mib": 256,
        "output_limit_bytes": 64 * 1024,
        "isolated_python": "-I -S",
    }


def _load_json_list(path: str | Path) -> list[dict[str, Any]]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"Lista de objetos JSON esperada em {path}.")
    return value


def _write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    try:
        losses = _parse_eval_losses(args.eval_loss)
        report = build_contract_report(
            args.tasks,
            args.predictions,
            _sandbox_runner,
            eval_losses=losses,
        )
        report["executor"] = _sandbox_metadata()
        _write_json(args.output, report)
        print(f"Relatorio: {args.output}")
        if report["ranking"]:
            print("Top 3:")
            for position, item in enumerate(report["ranking"][:3], start=1):
                print(
                    f"  {position}. {item['checkpoint']}: "
                    f"{item['functional_pass_count']}/{item['task_count']} funcionais, "
                    f"contratos={item['contract_fraction']:.3f}, "
                    f"formato={item['format']:.3f}"
                )

        promotion_requested = any(
            value
            for value in (
                args.promotion_output,
                args.ratings,
                args.mapping,
                args.candidate,
                args.champion,
            )
        )
        if promotion_requested:
            required = {
                "--promotion-output": args.promotion_output,
                "--ratings": args.ratings,
                "--mapping": args.mapping,
                "--candidate": args.candidate,
                "--champion": args.champion,
            }
            missing = [name for name, value in required.items() if not value]
            if missing:
                raise ValueError(f"Gate incompleto; faltam: {', '.join(missing)}")
            ratings = _load_json_list(args.ratings)
            mapping = _load_json_list(args.mapping)
            totals = aggregate_manual_ratings(ratings, mapping)
            gate = promotion_gate(
                report["models"],
                totals,
                candidate=args.candidate,
                champion=args.champion,
                base=args.base,
                candidate_rating_identity=args.candidate_rating_identity,
                base_rating_identity=args.base_rating_identity,
            )
            gate["evaluation_report"] = str(args.output)
            gate["ratings"] = str(args.ratings)
            gate["ratings_sha256"] = file_sha256(args.ratings)
            gate["mapping"] = str(args.mapping)
            gate["mapping_sha256"] = file_sha256(args.mapping)
            _write_json(args.promotion_output, gate)
            decision = "PROMOVER" if gate["promote"] else "MANTER CAMPEAO"
            print(f"Gate: {decision} ({args.promotion_output})")
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
