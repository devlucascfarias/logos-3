from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from fable_distill.callbacks import environment_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record CUDA/package environment and optional 4-bit smoke test")
    parser.add_argument("--output", default="outputs/logs/environment.json")
    parser.add_argument("--model", default="Qwen/Qwen3-8B-Base")
    parser.add_argument("--model-smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot = environment_snapshot()
    try:
        import accelerate
        import datasets
        import peft
        import transformers

        snapshot["packages"] = {
            "accelerate": accelerate.__version__,
            "datasets": datasets.__version__,
            "peft": peft.__version__,
            "transformers": transformers.__version__,
        }
    except ImportError as exc:
        snapshot["package_import_error"] = str(exc)
    if args.model_smoke:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise SystemExit("Install training dependencies: pip install -r requirements.txt") from exc
        if not torch.cuda.is_available():
            raise SystemExit("--model-smoke requires a CUDA GPU")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            ),
            device_map={"": 0},
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        encoded = tokenizer("Return a short Python function.", return_tensors="pt").to(model.device)
        with torch.inference_mode():
            logits = model(**encoded).logits
        snapshot["model_smoke"] = {
            "model": args.model,
            "logits_shape": list(logits.shape),
            "passed": True,
        }
        snapshot["vram_peak_bytes"] = torch.cuda.max_memory_allocated()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(snapshot, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

