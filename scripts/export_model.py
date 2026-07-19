from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import _bootstrap  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an adapter or optionally merge it on CPU")
    parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--merge", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.adapter)
    output = Path(args.output)
    if not source.exists():
        raise SystemExit(f"Adapter not found: {source}")
    if not args.merge:
        if output.exists():
            raise SystemExit(f"Output already exists: {output}")
        shutil.copytree(source, output)
        (output / "export_manifest.json").write_text(
            json.dumps(
                {"base_model": args.base_model, "adapter": str(source), "merged": False},
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"adapter exported to {output}")
        return
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install training dependencies: pip install -r requirements.txt") from exc
    print("Loading the full base model on CPU; ensure at least ~32 GiB free system RAM.")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    merged = PeftModel.from_pretrained(base, str(source)).merge_and_unload()
    output.mkdir(parents=True, exist_ok=False)
    merged.save_pretrained(output, safe_serialization=True, max_shard_size="4GB")
    AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True).save_pretrained(output)
    (output / "export_manifest.json").write_text(
        json.dumps(
            {"base_model": args.base_model, "adapter": str(source), "merged": True},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"merged model exported to {output}")


if __name__ == "__main__":
    main()
