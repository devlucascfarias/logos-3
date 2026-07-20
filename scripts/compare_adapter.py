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
        "--stage", choices=("baseline", "main", "agentic"), default="baseline"
    )
    parser.add_argument(
        "--prompts", default="examples/smoke_eval_prompts.json"
    )
    parser.add_argument("--adapter-path")
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
        base_first = rng.choice((True, False))
        labels = (
            {"A": "base", "B": "adapter"}
            if base_first
            else {"A": "adapter", "B": "base"}
        )
        comparisons.append(
            {
                "id": result["id"],
                "title": result["title"],
                "prompt": result["prompt"],
                "A": result[labels["A"]]["text"],
                "B": result[labels["B"]]["text"],
                "generation_seconds": {
                    "A": result[labels["A"]]["seconds"],
                    "B": result[labels["B"]]["seconds"],
                },
            }
        )
        mapping.append({"id": result["id"], **labels})
    return comparisons, mapping


def _render_markdown(comparisons: list[dict[str, Any]]) -> str:
    sections = [
        "# Comparação cega: Qwen3-8B base vs. adapter",
        "",
        "Avalie antes de abrir `mapping.json`. Para cada resposta, atribua notas "
        "de 0 a 5 em correção, cumprimento das instruções e qualidade da explicação.",
    ]
    for item in comparisons:
        sections.extend(
            [
                "",
                f"## {item['title']} (`{item['id']}`)",
                "",
                "### Prompt",
                "",
                item["prompt"],
                "",
                "### Resposta A",
                "",
                item["A"],
                "",
                "### Resposta B",
                "",
                item["B"],
                "",
                "| Resposta | Correção (0–5) | Instruções (0–5) | "
                "Explicação (0–5) | Observações |",
                "|---|---:|---:|---:|---|",
                "| A |  |  |  |  |",
                "| B |  |  |  |  |",
            ]
        )
    return "\n".join(sections) + "\n"


def _rating_template(comparisons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": item["id"],
            "ratings": {
                "A": {
                    "correctness": None,
                    "instruction_following": None,
                    "explanation_quality": None,
                },
                "B": {
                    "correctness": None,
                    "instruction_following": None,
                    "explanation_quality": None,
                },
            },
            "notes": "",
        }
        for item in comparisons
    ]


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
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    model.eval()
    model.config.use_cache = True
    device = next(model.parameters()).device
    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )

    def generate(prompt: str, *, adapter_enabled: bool) -> dict[str, Any]:
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
        context = nullcontext() if adapter_enabled else model.disable_adapter()
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
        f"[2/3] Gerando {len(prompts) * 2} respostas determinísticas...",
        flush=True,
    )
    results: list[dict[str, Any]] = []
    for prompt in tqdm(prompts, desc="Prompts", unit="prompt"):
        results.append(
            {
                **prompt,
                "base": generate(prompt["prompt"], adapter_enabled=False),
                "adapter": generate(prompt["prompt"], adapter_enabled=True),
            }
        )

    comparisons, mapping = _blind_results(results, args.seed)
    comparison_path = output_dir / "comparison.json"
    mapping_path = output_dir / "mapping.json"
    ratings_path = output_dir / "ratings.json"
    markdown_path = output_dir / "comparison.md"
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
    print("[3/3] Comparação concluída.", flush=True)
    print(f"Comparação cega: {markdown_path}")
    print(f"Notas: {ratings_path}")
    print(f"Mapeamento (abra somente após avaliar): {mapping_path}")


if __name__ == "__main__":
    main()
