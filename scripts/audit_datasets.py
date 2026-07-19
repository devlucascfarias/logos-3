from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from fable_distill.filtering import API_KEY_RE, EMAIL_RE, KNOWN_TOKEN_RE
from fable_distill.io import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit downloaded datasets without exporting rows")
    parser.add_argument("--config", default="configs/data.yaml")
    return parser.parse_args()


def _scan_value(value: Any, counters: Counter[str]) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _scan_value(item, counters)
    elif isinstance(value, list):
        for item in value:
            _scan_value(item, counters)
    elif isinstance(value, str):
        counters["email_matches"] += len(EMAIL_RE.findall(value))
        counters["secret_assignment_matches"] += len(API_KEY_RE.findall(value))
        counters["known_token_matches"] += len(KNOWN_TOKEN_RE.findall(value))


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    try:
        from datasets import DatasetDict, load_from_disk
    except ImportError as exc:
        raise SystemExit("Install datasets: pip install -e '.[data]'") from exc

    manifest_dir = Path(config["manifest_dir"])
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for source in config.get("sources", []):
        alias = source.get("alias")
        path = Path(config["raw_dir"]) / str(alias)
        if not source.get("enabled", True) or not source.get("name") or not path.exists():
            continue
        loaded = load_from_disk(str(path))
        splits = loaded if isinstance(loaded, DatasetDict) else DatasetDict({"train": loaded})
        report: dict[str, Any] = {
            "source_dataset": source["name"],
            "alias": alias,
            "audited_at": datetime.now(timezone.utc).isoformat(),
            "license": source.get("license", "unknown"),
            "splits": {},
        }
        for split_name, split in splits.items():
            counters: Counter[str] = Counter()
            for row in split.select(range(min(len(split), 1000))):
                _scan_value(row, counters)
            report["splits"][split_name] = {
                "rows": len(split),
                "columns": split.column_names,
                "features": str(split.features),
                "fingerprint": split._fingerprint,
                "pii_scan_first_1000": dict(counters),
            }
        output = manifest_dir / f"{alias}.audit.json"
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"audited {source['name']} -> {output}")


if __name__ == "__main__":
    main()

