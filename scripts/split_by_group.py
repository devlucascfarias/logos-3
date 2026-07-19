from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from fable_distill.io import iter_jsonl, write_jsonl
from fable_distill.splitting import split_by_connected_groups, validate_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leak-safe split by connected identifiers")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.90)
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(iter_jsonl(args.input))
    splits = split_by_connected_groups(
        rows,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    validate_splits(splits)
    output_dir = Path(args.output_dir)
    counts = {}
    for name, values in splits.items():
        counts[name] = write_jsonl(output_dir / f"{name}.jsonl", values)
    manifest = {
        "input": args.input,
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "validation": args.validation_ratio,
            "test": args.test_ratio,
        },
        "counts": counts,
        "leakage_validation": "passed",
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

