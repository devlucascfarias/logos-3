from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen_sft.config import load_config
from qwen_sft.scoring import rank_checkpoints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ordena checkpoints pela fórmula definida na receita."
    )
    parser.add_argument(
        "reports",
        nargs="+",
        help="Relatórios JSON com checkpoint e métricas normalizadas em [0,1].",
    )
    parser.add_argument(
        "--config", default="configs/checkpoint_scoring.yaml"
    )
    parser.add_argument(
        "--output", default="outputs/evaluations/checkpoint_ranking.json"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config) if args.config.endswith("recipe.yaml") else None
    if config is not None:
        raise SystemExit(
            "Use configs/checkpoint_scoring.yaml, não a receita completa."
        )

    import yaml

    with Path(args.config).open("r", encoding="utf-8") as handle:
        scoring = yaml.safe_load(handle)
    reports = []
    for path in args.reports:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(value, list):
            reports.extend(value)
        else:
            reports.append(value)
    ranked = rank_checkpoints(
        reports,
        {key: float(value) for key, value in scoring["weights"].items()},
        {key: float(value) for key, value in scoring["penalties"].items()},
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(ranked, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if ranked:
        print(f"Melhor checkpoint: {ranked[0].get('checkpoint', '<sem nome>')}")
        print(f"Score: {ranked[0]['score']:.6f}")
    print(f"Ranking: {output}")


if __name__ == "__main__":
    main()
