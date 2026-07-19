from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import _bootstrap  # noqa: F401

from fable_distill.datasets import parse_source_row
from fable_distill.formatting import (
    build_training_examples,
    flatten_for_training,
    normalized_prompt_hash,
)
from fable_distill.io import load_yaml, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build canonical flattened training examples")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Maximum rows per source")
    parser.add_argument("--max-output-examples", type=int, default=None)
    parser.add_argument("--exclude-glint", action="store_true")
    return parser.parse_args()


def _iter_splits(path: Path) -> Iterator[tuple[str, Any]]:
    from datasets import DatasetDict, load_from_disk

    loaded = load_from_disk(str(path))
    if isinstance(loaded, DatasetDict):
        yield from loaded.items()
    else:
        yield "train", loaded


def _weighted_take(
    pools: dict[str, list[dict[str, Any]]],
    weights: dict[str, float],
    total: int,
    seed: int,
) -> list[dict[str, Any]]:
    randomizer = random.Random(seed)
    for pool in pools.values():
        randomizer.shuffle(pool)
    active = [name for name, pool in pools.items() if pool and weights.get(name, 0) > 0]
    if not active or total <= 0:
        return []
    weight_sum = sum(weights[name] for name in active)
    desired = {name: total * weights[name] / weight_sum for name in active}
    quotas = {name: min(len(pools[name]), int(desired[name])) for name in active}
    remaining = total - sum(quotas.values())
    while remaining > 0:
        candidates = [name for name in active if quotas[name] < len(pools[name])]
        if not candidates:
            break
        choice = max(
            candidates,
            key=lambda name: (
                desired[name] - quotas[name],
                weights[name],
                -len(pools[name]),
            ),
        )
        quotas[choice] += 1
        remaining -= 1
    selected: list[dict[str, Any]] = []
    for name in active:
        selected.extend(pools[name][: quotas[name]])
    randomizer.shuffle(selected)
    return selected


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    try:
        import datasets  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Install datasets: pip install -e '.[data]'") from exc

    output = Path(args.output or Path(config["interim_dir"]) / "examples.jsonl")
    min_score = float(config.get("min_quality_score", 0.0))
    max_chars = int(config.get("max_tool_output_chars", 12000))
    pools: dict[str, list[dict[str, Any]]] = {}
    weights: dict[str, float] = {}
    source_rows = 0
    counters: Counter[str] = Counter()
    reasoning_mode_weights = {
        str(name): float(weight)
        for name, weight in config.get("sampling", {}).get("reasoning_modes", {}).items()
    }

    for source in config.get("sources", []):
        alias = str(source.get("alias"))
        source_name = source.get("name")
        if not source.get("enabled", True) or not source_name:
            continue
        if args.exclude_glint and alias == "glint":
            continue
        path = Path(config["raw_dir"]) / alias
        if not path.exists():
            print(f"warning: missing {path}; run download_datasets.py")
            continue
        download_manifest = Path(config["manifest_dir"]) / f"{alias}.download.json"
        manifest_data: dict[str, Any] = {}
        if download_manifest.exists():
            manifest_data = json.loads(download_manifest.read_text(encoding="utf-8"))
        resolved_revision = str(
            manifest_data.get("resolved_revision", source.get("revision", "main"))
        )
        resolved_license = str(source.get("license", "unknown"))
        if resolved_license == "unknown":
            resolved_license = str(manifest_data.get("license_card", "unknown"))
        pool: list[dict[str, Any]] = []
        source_count = 0
        source_max_rows = source.get("max_rows")
        for split_name, split in _iter_splits(path):
            for index, raw in enumerate(split):
                if args.limit is not None and source_count >= args.limit:
                    break
                if source_max_rows is not None and source_count >= int(source_max_rows):
                    break
                source_rows += 1
                source_count += 1
                canonical = parse_source_row(
                    dict(raw),
                    source_dataset=str(source_name),
                    source_revision=resolved_revision,
                    license_name=resolved_license,
                    row_index=index,
                    max_tool_output_chars=max_chars,
                )
                if canonical is None:
                    counters["filtered_invalid"] += 1
                    continue
                canonical.metadata["source_split"] = split_name
                if canonical.quality_score < min_score:
                    counters["filtered_quality"] += 1
                    continue
                for training_example in build_training_examples(
                    canonical,
                    reasoning_mode_weights=reasoning_mode_weights or None,
                ):
                    row = flatten_for_training(training_example)
                    row["normalized_prompt_hash"] = normalized_prompt_hash(
                        training_example.messages
                    )
                    pool.append(row)
                    counters[f"type_{training_example.target_type}"] += 1
            if args.limit is not None and source_count >= args.limit:
                break
            if source_max_rows is not None and source_count >= int(source_max_rows):
                break
        pools[alias] = pool
        weights[alias] = float(source.get("weight", 1.0))
        counters[f"source_{alias}"] = len(pool)

    raw_available = sum(len(pool) for pool in pools.values())
    type_weights = {
        str(name): float(weight)
        for name, weight in config.get("sampling", {}).get("example_types", {}).items()
    }
    type_feasible_by_source: dict[str, int] = {}
    for source_index, (source_alias, source_pool) in enumerate(pools.items()):
        type_pools: dict[str, list[dict[str, Any]]] = {}
        for row in source_pool:
            type_pools.setdefault(str(row.get("target_type", "unknown")), []).append(row)
        active_types = [
            name
            for name, pool in type_pools.items()
            if pool and type_weights.get(name, 0) > 0
        ]
        if not active_types:
            pools[source_alias] = []
            type_feasible_by_source[source_alias] = 0
            continue
        type_weight_sum = sum(type_weights[name] for name in active_types)
        type_feasible = int(
            min(
                len(type_pools[name]) * type_weight_sum / type_weights[name]
                for name in active_types
            )
        )
        pools[source_alias] = _weighted_take(
            type_pools,
            type_weights,
            type_feasible,
            int(config.get("seed", 42)) + source_index + 1,
        )
        type_feasible_by_source[source_alias] = len(pools[source_alias])

    available = sum(len(pool) for pool in pools.values())
    active = [name for name, pool in pools.items() if pool and weights.get(name, 0) > 0]
    if active:
        weight_sum = sum(weights[name] for name in active)
        feasible_total = int(
            min(len(pools[name]) * weight_sum / weights[name] for name in active)
        )
    else:
        feasible_total = 0
    requested_total = args.max_output_examples or feasible_total
    total = min(requested_total, feasible_total, available)
    rows = _weighted_take(pools, weights, total, int(config.get("seed", 42)))
    written = write_jsonl(output, rows)
    selected_sources = Counter(str(row.get("source_dataset", "unknown")) for row in rows)
    selected_types = Counter(str(row.get("target_type", "unknown")) for row in rows)
    selected_reasoning_modes = Counter(
        str(row.get("reasoning_mode", "unknown")) for row in rows
    )
    report = {
        "source_rows_read": source_rows,
        "raw_available_examples": raw_available,
        "type_balanced_examples": available,
        "feasible_weighted_examples": feasible_total,
        "type_balanced_by_source": type_feasible_by_source,
        "written": written,
        "selected_by_source": dict(selected_sources),
        "selected_by_type": dict(selected_types),
        "selected_by_reasoning_mode": dict(selected_reasoning_modes),
        "configured_reasoning_mode_weights": reasoning_mode_weights,
        **dict(counters),
    }
    report_path = output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
