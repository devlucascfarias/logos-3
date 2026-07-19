from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from fable_distill.io import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download configured Hugging Face datasets")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--exclude-glint", action="store_true")
    parser.add_argument("--force", action="store_true", help="Replace existing saved datasets")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("Install data dependencies: pip install -e '.[data]'") from exc

    raw_dir = Path(config["raw_dir"])
    manifest_dir = Path(config["manifest_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    for source in config.get("sources", []):
        name = source.get("name")
        alias = source.get("alias")
        if not source.get("enabled", True) or not name:
            continue
        if args.exclude_glint and alias == "glint":
            print(f"skip {name}: --exclude-glint")
            continue
        destination = raw_dir / str(alias)
        if destination.exists() and not args.force:
            print(f"skip {name}: {destination} already exists")
            continue
        if destination.exists():
            shutil.rmtree(destination)

        requested_revision = str(source.get("revision", "main"))
        info: Any = api.dataset_info(name, revision=requested_revision)
        resolved_revision = getattr(info, "sha", requested_revision)
        print(f"download {name}@{resolved_revision}")
        dataset = load_dataset(name, revision=resolved_revision, token=token)
        dataset.save_to_disk(str(destination))
        card_data = getattr(info, "card_data", None)
        if isinstance(card_data, dict):
            card_license = card_data.get("license")
        else:
            card_license = getattr(card_data, "license", None) if card_data else None
        manifest = {
            "source_dataset": name,
            "alias": alias,
            "requested_revision": requested_revision,
            "resolved_revision": resolved_revision,
            "license_configured": source.get("license", "unknown"),
            "license_card": card_license or "unknown",
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "dataset_path": str(destination),
            "splits": list(dataset.keys()) if hasattr(dataset, "keys") else ["train"],
            "fingerprints": (
                {key: value._fingerprint for key, value in dataset.items()}
                if hasattr(dataset, "items")
                else {"train": dataset._fingerprint}
            ),
        }
        output = manifest_dir / f"{alias}.download.json"
        output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"saved {output}")


if __name__ == "__main__":
    main()
