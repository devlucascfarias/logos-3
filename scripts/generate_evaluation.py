from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from qwen_sft.config import load_config
from qwen_sft.io import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera previsões de avaliação sem executar código do modelo."
    )
    parser.add_argument("--config", default="configs/recipe.yaml")
    parser.add_argument("--tasks", required=True, help="JSONL com id e prompt/messages.")
    parser.add_argument("--adapter", help="Adapter PEFT; omita para avaliar o base.")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode",
        choices=("thinking", "direct"),
        help="Padrao: thinking no benchmark e direct na decodificacao deterministica.",
    )
    parser.add_argument(
        "--decoding",
        choices=("benchmark", "deterministic"),
        default="benchmark",
        help="deterministic usa greedy e, por padrao, no maximo 512 tokens.",
    )
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _task_messages(task: dict[str, Any]) -> list[dict[str, Any]]:
    messages = task.get("messages")
    if isinstance(messages, list) and messages:
        return messages
    prompt = task.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Tarefa {task.get('id', '<sem id>')} sem prompt/messages.")
    system = task.get("system")
    result: list[dict[str, Any]] = []
    if isinstance(system, str) and system.strip():
        result.append({"role": "system", "content": system})
    result.append({"role": "user", "content": prompt})
    return result


def _resolve_generation_args(args: argparse.Namespace) -> tuple[str, int, dict[str, Any]]:
    deterministic = args.decoding == "deterministic"
    mode = args.mode or ("direct" if deterministic else "thinking")
    max_new_tokens = args.max_new_tokens
    if max_new_tokens is None:
        max_new_tokens = 512 if deterministic else 2048
    if max_new_tokens <= 0:
        raise ValueError("--max-new-tokens deve ser positivo.")
    if deterministic and mode != "direct":
        raise ValueError("--decoding deterministic requer --mode direct.")
    if deterministic and max_new_tokens != 512:
        raise ValueError(
            "--decoding deterministic requer --max-new-tokens 512."
        )
    if deterministic:
        generation: dict[str, Any] = {
            "do_sample": False,
            "num_beams": 1,
            "use_cache": True,
            "max_new_tokens": max_new_tokens,
        }
    else:
        thinking = mode == "thinking"
        generation = {
            "do_sample": True,
            "temperature": 0.6 if thinking else 0.7,
            "top_p": 0.95 if thinking else 0.8,
            "top_k": 20,
            "max_new_tokens": max_new_tokens,
        }
    return mode, max_new_tokens, generation


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    try:
        mode, max_new_tokens, generation = _resolve_generation_args(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise SystemExit(
            "Instale as dependências: pip install -r requirements-colab.txt"
        ) from exc
    if not torch.cuda.is_available():
        raise SystemExit("A geração do modelo 8B quantizado requer CUDA.")

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        revision=config.get("model_revision", "main"),
        use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        revision=config.get("model_revision", "main"),
        quantization_config=quantization,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    if args.adapter:
        if not Path(args.adapter).exists():
            raise SystemExit(f"Adapter não encontrado: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    thinking = mode == "thinking"
    generation.update(
        {
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
    )
    tasks = list(read_jsonl(args.tasks))
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = args.adapter or config["model_name"]
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for index, task in enumerate(tasks):
            torch.manual_seed(args.seed + index)
            messages = _task_messages(task)
            template_kwargs: dict[str, Any] = {
                "tokenize": True,
                "add_generation_prompt": True,
                "enable_thinking": thinking,
                "return_tensors": "pt",
                "return_dict": True,
            }
            if isinstance(task.get("tools"), list):
                template_kwargs["tools"] = task["tools"]
            encoded = tokenizer.apply_chat_template(messages, **template_kwargs)
            encoded = {
                key: value.to(model.device) for key, value in encoded.items()
            }
            input_length = encoded["input_ids"].shape[-1]
            with torch.inference_mode():
                output_ids = model.generate(**encoded, **generation)
            response = tokenizer.decode(
                output_ids[0, input_length:], skip_special_tokens=True
            ).strip()
            generated_ids = output_ids[0, input_length:]
            generated_tokens = int(generated_ids.shape[-1])
            ended_with_eos = bool(
                generated_tokens
                and tokenizer.eos_token_id is not None
                and int(generated_ids[-1].item()) == int(tokenizer.eos_token_id)
            )
            result = {
                "id": task.get("id", index),
                "checkpoint": checkpoint,
                "mode": mode,
                "decoding": args.decoding,
                "category": task.get("category"),
                "family": task.get("family"),
                "response": response,
                "generation": generation,
                "generated_tokens": generated_tokens,
                "ended_with_eos": ended_with_eos,
                "truncated": generated_tokens >= max_new_tokens and not ended_with_eos,
                "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"[{index + 1}/{len(tasks)}] {result['id']}")
    print(f"Previsões: {target}")


if __name__ == "__main__":
    main()
