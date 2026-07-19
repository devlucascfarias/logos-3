from __future__ import annotations

import argparse
import random
from collections import defaultdict

import _bootstrap  # noqa: F401

from fable_distill.io import iter_jsonl, write_jsonl
from fable_distill.scoring import preference_reward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build chosen/rejected pairs from verified runs")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--validation-output", default=None)
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-margin", type=float, default=0.25)
    return parser.parse_args()


def _reward(row: dict) -> float:
    if "reward" in row:
        return float(row["reward"])
    return preference_reward(
        tests_passed=bool(row.get("tests_passed", False)),
        build_passed=bool(row.get("build_passed", False)),
        invalid_calls=int(row.get("invalid_calls", 0)),
        normalized_patch_size=float(row.get("normalized_patch_size", 0.0)),
        extra_penalty=float(row.get("extra_penalty", 0.0)),
    )


def main() -> None:
    args = parse_args()
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in iter_jsonl(args.input):
        row["computed_reward"] = _reward(row)
        grouped[str(row.get("task_id"))].append(row)

    def pairs():
        for task_id, candidates in grouped.items():
            if len(candidates) < 2:
                continue
            ranked = sorted(candidates, key=lambda row: row["computed_reward"])
            rejected, chosen = ranked[0], ranked[-1]
            margin = chosen["computed_reward"] - rejected["computed_reward"]
            if margin < args.min_margin:
                continue
            yield {
                "task_id": task_id,
                "prompt": chosen["prompt"],
                "chosen": chosen["completion"],
                "rejected": rejected["completion"],
                "chosen_reward": chosen["computed_reward"],
                "rejected_reward": rejected["computed_reward"],
                "margin": margin,
            }

    values = list(pairs())
    random.Random(args.seed).shuffle(values)
    if args.validation_output and values:
        validation_count = max(1, round(len(values) * args.validation_ratio))
        validation = values[:validation_count]
        training = values[validation_count:]
        write_jsonl(args.output, training)
        write_jsonl(args.validation_output, validation)
        print(
            f"wrote {len(training)} train pairs to {args.output} and "
            f"{len(validation)} validation pairs to {args.validation_output}"
        )
    else:
        count = write_jsonl(args.output, values)
        print(f"wrote {count} preference pairs to {args.output}")


if __name__ == "__main__":
    main()
