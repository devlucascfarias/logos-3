from __future__ import annotations

import argparse
import random
from pathlib import Path

import _bootstrap  # noqa: F401

from fable_distill.io import iter_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate multiple model candidates per task")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B-Base")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise SystemExit("Install training dependencies: pip install -r requirements.txt") from exc
    if not torch.cuda.is_available():
        raise SystemExit("Candidate generation for the 8B model requires a CUDA runtime")
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
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    random.seed(args.seed)

    def candidates():
        for task_index, task in enumerate(iter_jsonl(args.tasks)):
            prompt = str(task.get("prompt") or task.get("instruction") or "")
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            for candidate_index in range(args.num_candidates):
                torch.manual_seed(args.seed + task_index * args.num_candidates + candidate_index)
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=args.temperature > 0,
                        temperature=max(args.temperature, 1e-5),
                        pad_token_id=tokenizer.eos_token_id,
                    )
                completion = tokenizer.decode(
                    generated[0, inputs["input_ids"].shape[1] :],
                    skip_special_tokens=False,
                )
                yield {
                    "task_id": task.get("task_id", task_index),
                    "candidate_id": candidate_index,
                    "prompt": prompt,
                    "completion": completion,
                    "model": args.model,
                    "adapter": args.adapter,
                    "seed": args.seed + task_index * args.num_candidates + candidate_index,
                }

    count = write_jsonl(Path(args.output), candidates())
    print(f"wrote {count} candidates to {args.output}")


if __name__ == "__main__":
    main()

