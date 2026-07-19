from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import _bootstrap  # noqa: F401

from fable_distill.formatting import (
    reasoning_mode_uses_thinking,
    render_generation_prompt,
)
from fable_distill.io import iter_jsonl
from fable_distill.tools import parse_tool_call


TAG_RE = re.compile(r"<(tool_call|think|plan)>|</(tool_call|think|plan)>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute static metrics from generated predictions")
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument(
        "--predictions",
        default=None,
        help="JSONL with completion/prediction; defaults to configured static file",
    )
    parser.add_argument("--output", default="outputs/evaluations/static_metrics.json")
    parser.add_argument(
        "--generate-models",
        action="store_true",
        help="Generate and compare base plus configured adapters on CUDA",
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def _tags_balanced(text: str) -> bool:
    stack: list[str] = []
    for match in TAG_RE.finditer(text):
        opening, closing = match.groups()
        if opening:
            stack.append(opening)
        elif not stack or stack.pop() != closing:
            return False
    return not stack


def compute_report(rows: Iterable[dict]) -> dict:
    metrics: Counter[str] = Counter()
    tool_names: Counter[str] = Counter()
    for row in rows:
        text = str(row.get("prediction", row.get("completion", "")))
        metrics["examples"] += 1
        if _tags_balanced(text):
            metrics["balanced_tags"] += 1
        if "<tool_call>" in text:
            metrics["tool_call_examples"] += 1
            try:
                parsed = parse_tool_call(text)
                metrics["valid_tool_calls"] += 1
                tool_names[parsed["name"]] += 1
            except ValueError:
                metrics["invalid_tool_calls"] += 1
        if row.get("target_tool_name"):
            try:
                parsed = parse_tool_call(text)
                if parsed["name"] == row["target_tool_name"]:
                    metrics["correct_tool_name"] += 1
            except ValueError:
                pass
    total = metrics["examples"] or 1
    tool_total = metrics["tool_call_examples"] or 1
    return {
        **dict(metrics),
        "balanced_tag_rate": metrics["balanced_tags"] / total,
        "valid_tool_call_rate": metrics["valid_tool_calls"] / tool_total,
        "tool_name_counts": dict(tool_names),
    }


def _generation_kwargs(generation: dict, reasoning_mode: str) -> dict:
    thinking = reasoning_mode_uses_thinking(reasoning_mode)
    profile_name = "thinking" if thinking else "non_thinking"
    profile = dict(generation.get("profiles", {}).get(profile_name, {}))
    do_sample = bool(profile.get("do_sample", True))
    kwargs = {
        "max_new_tokens": int(generation.get("max_new_tokens", 512)),
        "do_sample": do_sample,
        "repetition_penalty": float(generation.get("repetition_penalty", 1.0)),
    }
    if do_sample:
        kwargs.update(
            {
                "temperature": float(
                    profile.get("temperature", 0.6 if thinking else 0.7)
                ),
                "top_p": float(profile.get("top_p", 0.95 if thinking else 0.8)),
                "top_k": int(profile.get("top_k", 20)),
            }
        )
    return kwargs


def generate_comparison(config: dict, limit: int | None, output_path: Path) -> dict:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise SystemExit("Install training dependencies: pip install -r requirements.txt") from exc
    if not torch.cuda.is_available():
        raise SystemExit("--generate-models requires a CUDA GPU")
    from fable_distill.callbacks import set_global_seed
    from fable_distill.io import write_jsonl

    rows = list(iter_jsonl(config["data"]["static_file"]))
    if limit:
        rows = rows[:limit]
    generation = config["generation"]
    base_name = config["model"]["base"]
    variants: list[tuple[str, str | None]] = [("base", None)]
    for adapter in config["model"].get("adapters", []):
        if Path(adapter).exists():
            variants.append((Path(adapter).name, adapter))
        else:
            print(f"warning: adapter not found, skipping: {adapter}")

    reports: dict[str, dict] = {}
    predictions_dir = output_path.parent / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    for variant_name, adapter in variants:
        set_global_seed(int(generation.get("seed", 42)))
        tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            base_name,
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
        if adapter:
            model = PeftModel.from_pretrained(model, adapter)
        model.eval()
        generated_rows: list[dict] = []
        for index, row in enumerate(rows):
            prompt, reasoning_mode = render_generation_prompt(tokenizer, row)
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
            ).to(model.device)
            torch.manual_seed(int(generation.get("seed", 42)) + index)
            generation_kwargs = _generation_kwargs(generation, reasoning_mode)
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    **generation_kwargs,
                    pad_token_id=tokenizer.pad_token_id,
                )
            prediction = tokenizer.decode(
                generated[0, encoded["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            generated_rows.append(
                {
                    "example_id": row.get("example_id", index),
                    "prediction": prediction,
                    "target_tool_name": row.get("target_tool_name"),
                    "variant": variant_name,
                    "seed": int(generation.get("seed", 42)) + index,
                    "reasoning_mode": reasoning_mode,
                    "enable_thinking": reasoning_mode_uses_thinking(reasoning_mode),
                    "generation": generation_kwargs,
                }
            )
        prediction_path = predictions_dir / f"{variant_name}.jsonl"
        write_jsonl(prediction_path, generated_rows)
        reports[variant_name] = compute_report(generated_rows)
        reports[variant_name]["predictions"] = str(prediction_path)
        del model
        torch.cuda.empty_cache()

    base_report = reports["base"]
    deltas: dict[str, dict[str, float]] = {}
    for name, report in reports.items():
        if name == "base":
            continue
        deltas[name] = {
            metric: float(report.get(metric, 0)) - float(base_report.get(metric, 0))
            for metric in ("balanced_tag_rate", "valid_tool_call_rate")
        }
    return {
        "model": base_name,
        "seed": generation.get("seed", 42),
        "generation": generation,
        "reports": reports,
        "deltas_vs_base": deltas,
    }


def main() -> None:
    args = parse_args()
    from fable_distill.io import load_yaml

    config = load_yaml(args.config)
    output = Path(args.output)
    if args.generate_models:
        report = generate_comparison(config, args.limit, output)
    else:
        path = args.predictions or config["data"]["static_file"]
        rows = iter_jsonl(path)
        if args.limit:
            rows = iter(list(rows)[: args.limit])
        report = compute_report(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
