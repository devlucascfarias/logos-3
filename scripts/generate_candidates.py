from __future__ import annotations

import argparse
import random
from pathlib import Path

import _bootstrap  # noqa: F401

from fable_distill.formatting import (
    generation_messages_from_row,
    reasoning_mode_uses_thinking,
    render_generation_prompt,
)
from fable_distill.io import iter_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate multiple model candidates per task")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--reasoning-mode",
        choices=("thinking", "non_thinking"),
        default="thinking",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
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
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
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
            prompt_row = dict(task)
            prompt_row.setdefault("reasoning_mode", args.reasoning_mode)
            prompt, reasoning_mode = render_generation_prompt(tokenizer, prompt_row)
            thinking = reasoning_mode_uses_thinking(reasoning_mode)
            temperature = (
                args.temperature
                if args.temperature is not None
                else (0.6 if thinking else 0.7)
            )
            top_p = args.top_p if args.top_p is not None else (0.95 if thinking else 0.8)
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(model.device)
            for candidate_index in range(args.num_candidates):
                torch.manual_seed(args.seed + task_index * args.num_candidates + candidate_index)
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=args.top_k,
                        repetition_penalty=args.repetition_penalty,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                completion = tokenizer.decode(
                    generated[0, inputs["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                )
                yield {
                    "task_id": task.get("task_id", task_index),
                    "candidate_id": candidate_index,
                    "prompt": prompt,
                    "prompt_messages": generation_messages_from_row(prompt_row),
                    "completion": completion,
                    "model": args.model,
                    "adapter": args.adapter,
                    "reasoning_mode": reasoning_mode,
                    "enable_thinking": thinking,
                    "generation": {
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": args.top_k,
                        "repetition_penalty": args.repetition_penalty,
                    },
                    "seed": args.seed + task_index * args.num_candidates + candidate_index,
                }

    count = write_jsonl(Path(args.output), candidates())
    print(f"wrote {count} candidates to {args.output}")


if __name__ == "__main__":
    main()
