from __future__ import annotations

import argparse
import json
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from qwen_sft.config import load_config, merged_training_config
from qwen_sft.io import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina Qwen3-8B com QLoRA NF4 em uma GPU L4."
    )
    parser.add_argument("--config", default="configs/recipe.yaml")
    parser.add_argument(
        "--stage", choices=("baseline", "main", "agentic"), default="baseline"
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default="auto",
        help="auto, none ou o caminho de um checkpoint do Trainer.",
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in (
        "torch",
        "transformers",
        "trl",
        "peft",
        "bitsandbytes",
        "datasets",
        "accelerate",
    ):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "not-installed"
    return result


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _resolve_resume(value: str, output_dir: str) -> str | None:
    if value.lower() in {"none", "false", "no"}:
        return None
    if value != "auto":
        return value
    if not Path(output_dir).exists():
        return None
    try:
        from transformers.trainer_utils import get_last_checkpoint
    except ImportError:
        return None
    return get_last_checkpoint(output_dir)


def _manifest(
    *,
    config_path: str,
    stage: str,
    train_file: Path,
    validation_file: Path,
    training: dict[str, Any],
    torch: Any,
) -> dict[str, Any]:
    return {
        "config": config_path,
        "stage": stage,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "packages": _package_versions(),
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "bf16_supported": torch.cuda.is_bf16_supported(),
            "total_vram_bytes": torch.cuda.get_device_properties(0).total_memory,
        },
        "data": {
            "train_file": str(train_file),
            "train_sha256": file_sha256(train_file),
            "validation_file": str(validation_file),
            "validation_sha256": (
                file_sha256(validation_file) if validation_file.exists() else None
            ),
        },
        "training": training,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    training = merged_training_config(config, args.stage)
    train_file = Path(f"data/processed/{args.stage}/train.jsonl")
    validation_file = Path(f"data/processed/{args.stage}/validation.jsonl")
    output_dir = str(training["output_dir"])

    if args.dry_run:
        print(
            json.dumps(
                {
                    "stage": args.stage,
                    "train_file": str(train_file),
                    "train_exists": train_file.exists(),
                    "validation_file": str(validation_file),
                    "validation_exists": validation_file.exists(),
                    "output_dir": output_dir,
                    "adapter_path": training.get("adapter_path"),
                    "resume": _resolve_resume(
                        args.resume_from_checkpoint, output_dir
                    ),
                    "training": training,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.stage == "agentic" and not training.get("adapter_path"):
        raise SystemExit(
            "A etapa agentic deve continuar um adapter escolhido. Defina "
            "stages.agentic.training.adapter_path em configs/recipe.yaml."
        )
    if not train_file.exists():
        raise SystemExit(
            f"Dados ausentes: {train_file}. Execute scripts/prepare_data.py primeiro."
        )

    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise SystemExit(
            "Instale as dependências: pip install -r requirements-colab.txt"
        ) from exc

    if not torch.cuda.is_available():
        raise SystemExit("O treino QLoRA requer CUDA; selecione uma GPU L4 no Colab.")
    gpu_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"GPU: {gpu_name} ({total_vram:.2f} GiB)")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("A GPU/runtime atual não oferece BF16.")
    if "L4" not in gpu_name.upper():
        print("AVISO: a receita foi dimensionada e validada para uma NVIDIA L4.")

    torch.backends.cuda.matmul.allow_tf32 = bool(training.get("tf32", True))
    quantization = BitsAndBytesConfig(
        load_in_4bit=bool(training.get("load_in_4bit", True)),
        bnb_4bit_quant_type=training.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=bool(
            training.get("bnb_4bit_use_double_quant", True)
        ),
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        revision=config.get("model_revision", "main"),
        use_fast=True,
    )
    if not getattr(tokenizer, "chat_template", None):
        raise SystemExit("O tokenizer não possui o chat template oficial do Qwen3.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_files = {"train": str(train_file)}
    use_eval = validation_file.exists() and not args.no_eval
    if use_eval:
        data_files["validation"] = str(validation_file)
    dataset = load_dataset("json", data_files=data_files)
    if args.max_train_samples:
        size = min(args.max_train_samples, len(dataset["train"]))
        dataset["train"] = dataset["train"].select(range(size))

    adapter_path = training.get("adapter_path")
    peft_config = None
    model: str | Any = config["model_name"]
    trainer_quantization = quantization
    if adapter_path:
        if not Path(adapter_path).exists():
            raise SystemExit(
                f"Adapter para continuação não encontrado: {adapter_path}"
            )
        base = AutoModelForCausalLM.from_pretrained(
            config["model_name"],
            revision=config.get("model_revision", "main"),
            quantization_config=quantization,
            dtype=torch.bfloat16,
            device_map={"": 0},
            attn_implementation=training.get("attn_implementation", "sdpa"),
        )
        base.config.use_cache = False
        base = prepare_model_for_kbit_training(
            base,
            use_gradient_checkpointing=bool(
                training.get("gradient_checkpointing", True)
            ),
            gradient_checkpointing_kwargs={
                "use_reentrant": bool(training.get("use_reentrant", False))
            },
        )
        model = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
        trainer_quantization = None
    else:
        peft_config = LoraConfig(
            r=int(training["lora_r"]),
            lora_alpha=int(training["lora_alpha"]),
            lora_dropout=float(training["lora_dropout"]),
            bias=str(training.get("lora_bias", "none")),
            task_type="CAUSAL_LM",
            target_modules=list(training["target_modules"]),
        )

    sft_kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "per_device_train_batch_size": int(
            training["per_device_train_batch_size"]
        ),
        "per_device_eval_batch_size": int(
            training["per_device_eval_batch_size"]
        ),
        "gradient_accumulation_steps": int(
            training["gradient_accumulation_steps"]
        ),
        "learning_rate": float(training["learning_rate"]),
        "num_train_epochs": float(training["num_train_epochs"]),
        "lr_scheduler_type": training["lr_scheduler_type"],
        "warmup_ratio": float(training["warmup_ratio"]),
        "weight_decay": float(training["weight_decay"]),
        "max_grad_norm": float(training["max_grad_norm"]),
        "optim": training["optim"],
        "bf16": bool(training.get("bf16", True)),
        "tf32": bool(training.get("tf32", True)),
        "gradient_checkpointing": bool(
            training.get("gradient_checkpointing", True)
        ),
        "gradient_checkpointing_kwargs": {
            "use_reentrant": bool(training.get("use_reentrant", False))
        },
        "use_cache": False,
        "logging_steps": int(training["logging_steps"]),
        "save_strategy": "steps",
        "save_steps": int(training["save_steps"]),
        "save_total_limit": int(training["save_total_limit"]),
        "eval_strategy": "steps" if use_eval else "no",
        "eval_steps": int(training["eval_steps"]) if use_eval else None,
        "report_to": training.get("report_to", "tensorboard"),
        "seed": int(config.get("seed", 42)),
        "data_seed": int(config.get("seed", 42)),
        "max_length": int(training["max_seq_length"]),
        "packing": bool(training.get("packing", True)),
        "packing_strategy": training.get("packing_strategy", "wrapped"),
        "eval_packing": False,
        "assistant_only_loss": bool(
            training.get("assistant_only_loss", True)
        ),
        "dataset_num_proc": 2,
        "remove_unused_columns": True,
        "include_num_input_tokens_seen": True,
    }
    if isinstance(model, str):
        sft_kwargs["model_init_kwargs"] = {
            "revision": config.get("model_revision", "main"),
            "dtype": torch.bfloat16,
            "device_map": {"": 0},
            "attn_implementation": training.get(
                "attn_implementation", "sdpa"
            ),
            "use_cache": False,
        }
    if args.max_steps is not None:
        sft_kwargs["max_steps"] = args.max_steps

    manifest = _manifest(
        config_path=args.config,
        stage=args.stage,
        train_file=train_file,
        validation_file=validation_file,
        training=training,
        torch=torch,
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    manifest_path = Path(output_dir) / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": SFTConfig(**sft_kwargs),
        "train_dataset": dataset["train"],
        "eval_dataset": dataset.get("validation") if use_eval else None,
        "processing_class": tokenizer,
    }
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config
    if trainer_quantization is not None:
        trainer_kwargs["quantization_config"] = trainer_quantization
    trainer = SFTTrainer(**trainer_kwargs)

    resume = _resolve_resume(args.resume_from_checkpoint, output_dir)
    print(f"Retomada: {resume or 'não'}")
    result = trainer.train(resume_from_checkpoint=resume)
    metrics = dict(result.metrics)
    metrics["peak_vram_bytes"] = torch.cuda.max_memory_allocated()
    metrics["peak_vram_gib"] = round(
        torch.cuda.max_memory_allocated() / 2**30, 3
    )
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    adapter_output = Path(training["adapter_output_dir"])
    adapter_output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(adapter_output))
    tokenizer.save_pretrained(str(adapter_output))
    if use_eval:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
        manifest["eval_metrics"] = eval_metrics
    manifest["train_metrics"] = metrics
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"Adapter salvo em: {adapter_output}")
    print(f"Pico de VRAM: {metrics['peak_vram_gib']:.2f} GiB")


if __name__ == "__main__":
    main()
