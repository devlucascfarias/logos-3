from __future__ import annotations

import argparse
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from qwen_sft.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compara cegamente o Qwen3-8B base com um adapter treinado."
    )
    parser.add_argument("--config", default="configs/recipe.yaml")
    parser.add_argument(
        "--stage",
        choices=("pilot", "baseline", "main", "agentic"),
        default="pilot",
    )
    parser.add_argument(
        "--prompts", default="examples/smoke_eval_prompts.json"
    )
    parser.add_argument("--adapter-path")
    parser.add_argument(
        "--reference-adapter-path",
        help="Adapter campeão opcional para comparação cega A/B/C.",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ativa o modo thinking do chat template.",
    )
    return parser.parse_args()


def _load_prompts(path: str | Path) -> list[dict[str, str]]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value:
        raise ValueError("O arquivo de prompts deve conter uma lista não vazia.")
    prompts: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"Prompt {index} não é um objeto.")
        prompt_id = str(item.get("id", "")).strip()
        title = str(item.get("title", "")).strip()
        prompt = str(item.get("prompt", "")).strip()
        if not prompt_id or not title or not prompt:
            raise ValueError(f"Prompt {index} possui campos obrigatórios vazios.")
        if prompt_id in seen:
            raise ValueError(f"ID de prompt duplicado: {prompt_id}")
        seen.add(prompt_id)
        prompts.append({"id": prompt_id, "title": title, "prompt": prompt})
    return prompts


def _blind_results(
    results: list[dict[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rng = random.Random(seed)
    comparisons: list[dict[str, Any]] = []
    mapping: list[dict[str, str]] = []
    for result in results:
        variants = [
            key
            for key, value in result.items()
            if isinstance(value, dict) and {"text", "seconds"} <= value.keys()
        ]
        shuffled = list(variants)
        rng.shuffle(shuffled)
        labels = {
            chr(ord("A") + index): variant
            for index, variant in enumerate(shuffled)
        }
        comparison = {
            "id": result["id"],
            "title": result["title"],
            "prompt": result["prompt"],
            "generation_seconds": {},
        }
        for label, variant in labels.items():
            comparison[label] = result[variant]["text"]
            comparison["generation_seconds"][label] = result[variant]["seconds"]
        comparisons.append(comparison)
        mapping.append({"id": result["id"], **labels})
    return comparisons, mapping


def _render_markdown(comparisons: list[dict[str, Any]]) -> str:
    sections = [
        "# Comparação cega: Qwen3-8B base vs. adapters",
        "",
        "Avalie antes de abrir `mapping.json`. Para cada resposta, atribua notas "
        "de 0 a 5 em correção, cumprimento das instruções e qualidade da explicação.",
    ]
    for item in comparisons:
        labels = list(item["generation_seconds"])
        sections.extend(
            [
                "",
                f"## {item['title']} (`{item['id']}`)",
                "",
                "### Prompt",
                "",
                item["prompt"],
            ]
        )
        for label in labels:
            sections.extend(
                [
                    "",
                    f"### Resposta {label}",
                    "",
                    item[label],
                ]
            )
        sections.extend(
            [
                "",
                "| Resposta | Correção (0–5) | Instruções (0–5) | "
                "Explicação (0–5) | Observações |",
                "|---|---:|---:|---:|---|",
                *[f"| {label} |  |  |  |  |" for label in labels],
            ]
        )
    return "\n".join(sections) + "\n"


def _rating_template(comparisons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ratings: list[dict[str, Any]] = []
    for item in comparisons:
        labels = list(item["generation_seconds"])
        ratings.append(
            {
                "id": item["id"],
                "ratings": {
                    label: {
                        "correctness": None,
                        "instruction_following": None,
                        "explanation_quality": None,
                    }
                    for label in labels
                },
                "notes": "",
            }
        )
    return ratings


def main() -> None:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens deve ser positivo.")

    config = load_config(args.config)
    prompts = _load_prompts(args.prompts)
    adapter_path = Path(
        args.adapter_path or f"outputs/adapters/{args.stage}"
    )
    if not (adapter_path / "adapter_config.json").exists():
        raise SystemExit(f"Adapter não encontrado: {adapter_path}")
    reference_path = (
        Path(args.reference_adapter_path)
        if args.reference_adapter_path
        else None
    )
    if reference_path and not (reference_path / "adapter_config.json").exists():
        raise SystemExit(f"Adapter de referência não encontrado: {reference_path}")

    output_dir = Path(
        args.output_dir or f"outputs/evaluations/{args.stage}_smoke"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from peft import PeftModel
        from tqdm import tqdm
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
    except ImportError as exc:
        raise SystemExit(
            "Instale as dependências: pip install -r requirements-colab.txt"
        ) from exc

    if not torch.cuda.is_available():
        raise SystemExit("A comparação requer CUDA.")

    print("[1/3] Carregando tokenizer e modelo-base em 4 bits...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        revision=config.get("model_revision", "main"),
        quantization_config=quantization,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(
        base,
        str(adapter_path),
        adapter_name="candidate",
        is_trainable=False,
    )
    if reference_path:
        model.load_adapter(
            str(reference_path),
            adapter_name="reference",
            is_trainable=False,
        )
    model.eval()
    model.config.use_cache = True
    device = next(model.parameters()).device
    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )

    def generate(prompt: str, *, adapter_name: str | None) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": config["system_prompt"]},
            {"role": "user", "content": prompt},
        ]
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=args.thinking,
            return_tensors="pt",
            return_dict=True,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        input_length = encoded["input_ids"].shape[-1]
        if adapter_name is None:
            context = model.disable_adapter()
        else:
            model.set_adapter(adapter_name)
            context = nullcontext()
        torch.manual_seed(args.seed)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with context, torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        text = tokenizer.decode(
            generated[0, input_length:],
            skip_special_tokens=True,
        ).strip()
        return {"text": text, "seconds": round(seconds, 3)}

    print(
        f"[2/3] Gerando "
        f"{len(prompts) * (3 if reference_path else 2)} "
        "respostas determinísticas...",
        flush=True,
    )
    results: list[dict[str, Any]] = []
    for prompt in tqdm(prompts, desc="Prompts", unit="prompt"):
        results.append(
            {
                **prompt,
                "base": generate(prompt["prompt"], adapter_name=None),
                "adapter": generate(
                    prompt["prompt"], adapter_name="candidate"
                ),
                **(
                    {
                        "reference": generate(
                            prompt["prompt"], adapter_name="reference"
                        )
                    }
                    if reference_path
                    else {}
                ),
            }
        )

    comparisons, mapping = _blind_results(results, args.seed)
    comparison_path = output_dir / "comparison.json"
    mapping_path = output_dir / "mapping.json"
    ratings_path = output_dir / "ratings.json"
    markdown_path = output_dir / "comparison.md"
    run_config_path = output_dir / "run_config.json"
    comparison_path.write_text(
        json.dumps(comparisons, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    mapping_path.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    ratings_path.write_text(
        json.dumps(
            _rating_template(comparisons),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        _render_markdown(comparisons),
        encoding="utf-8",
    )
    run_config_path.write_text(
        json.dumps(
            {
                "model": config["model_name"],
                "model_revision": config.get("model_revision", "main"),
                "stage": args.stage,
                "candidate_adapter": str(adapter_path),
                "reference_adapter": (
                    str(reference_path) if reference_path else None
                ),
                "prompts": args.prompts,
                "seed": args.seed,
                "thinking": args.thinking,
                "max_new_tokens": args.max_new_tokens,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("[3/3] Comparação concluída.", flush=True)
    print(f"Comparação cega: {markdown_path}")
    print(f"Notas: {ratings_path}")
    print(f"Mapeamento (abra somente após avaliar): {mapping_path}")


if __name__ == "__main__":
    main()
