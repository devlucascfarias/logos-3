from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from fable_distill.callbacks import (
    CheckpointManifestCallback,
    VramMetricsCallback,
    finalize_run_manifest,
    set_global_seed,
    write_run_manifest,
)
from fable_distill.checkpoints import find_latest_checkpoint
from fable_distill.collators import AssistantOnlyDataCollator
from fable_distill.io import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen3 with 4-bit QLoRA and assistant-only loss")
    parser.add_argument("--config", default="configs/sft_stage1.yaml")
    parser.add_argument("--resume-from-checkpoint", default="auto")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _require_cuda(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("QLoRA training requires a CUDA GPU; use a Colab L4 runtime")
    free, total = torch.cuda.mem_get_info()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {free / 2**30:.2f} GiB free / {total / 2**30:.2f} GiB total")
    print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    train_config = config["training"]
    if args.dry_run:
        train_file = Path(train_config["train_file"])
        validation_file = Path(train_config["validation_file"])
        print(f"config: {args.config}")
        print(f"train exists: {train_file.exists()} ({train_file})")
        print(f"validation exists: {validation_file.exists()} ({validation_file})")
        print(f"latest checkpoint: {find_latest_checkpoint(train_config['output_dir'])}")
        return

    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from huggingface_hub import HfApi
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit("Install training dependencies: pip install -r requirements.txt") from exc

    _require_cuda(torch)
    seed = int(train_config.get("seed", 42))
    set_global_seed(seed)
    model_config = config["model"]
    requested_revision = model_config.get("revision", "main")
    try:
        model_revision = HfApi().model_info(
            model_config["name"], revision=requested_revision
        ).sha
    except Exception as exc:
        print(f"warning: could not resolve model revision ({exc}); using {requested_revision}")
        model_revision = requested_revision
    dtype = torch.bfloat16 if model_config.get("compute_dtype") == "bfloat16" else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=bool(model_config.get("load_in_4bit", True)),
        bnb_4bit_quant_type=model_config.get("quant_type", "nf4"),
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=bool(model_config.get("double_quant", True)),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name"],
        revision=model_revision,
        trust_remote_code=bool(model_config.get("trust_remote_code", True)),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_config["name"],
        revision=model_revision,
        quantization_config=quantization,
        torch_dtype=dtype,
        device_map={"": 0},
        attn_implementation=model_config.get("attn_implementation", "sdpa"),
        trust_remote_code=bool(model_config.get("trust_remote_code", True)),
    )
    model.config.use_cache = bool(train_config.get("use_cache", False))
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=bool(train_config.get("gradient_checkpointing", True)),
    )
    lora = config["lora"]
    adapter_path = lora.get("adapter_path")
    if adapter_path and Path(adapter_path).exists():
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=int(lora["r"]),
                lora_alpha=int(lora["alpha"]),
                lora_dropout=float(lora["dropout"]),
                bias=lora.get("bias", "none"),
                task_type="CAUSAL_LM",
                target_modules=list(lora["target_modules"]),
            ),
        )
    model.print_trainable_parameters()

    files = {
        "train": train_config["train_file"],
        "validation": train_config["validation_file"],
    }
    dataset = load_dataset("json", data_files=files)
    if args.max_train_samples:
        dataset["train"] = dataset["train"].select(
            range(min(args.max_train_samples, len(dataset["train"])))
        )
    output_dir = str(train_config["output_dir"])
    kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "per_device_train_batch_size": int(train_config["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(train_config["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": int(train_config["gradient_accumulation_steps"]),
        "learning_rate": float(train_config["learning_rate"]),
        "num_train_epochs": float(train_config["num_train_epochs"]),
        "warmup_ratio": float(train_config["warmup_ratio"]),
        "weight_decay": float(train_config["weight_decay"]),
        "max_grad_norm": float(train_config["max_grad_norm"]),
        "optim": train_config["optim"],
        "lr_scheduler_type": train_config["lr_scheduler_type"],
        "bf16": bool(train_config.get("bf16", True)),
        "fp16": bool(train_config.get("fp16", False)),
        "gradient_checkpointing": bool(train_config.get("gradient_checkpointing", True)),
        "logging_steps": int(train_config["logging_steps"]),
        "eval_steps": int(train_config["eval_steps"]),
        "save_steps": int(train_config["save_steps"]),
        "save_total_limit": int(train_config["save_total_limit"]),
        "seed": seed,
        "data_seed": seed,
        "report_to": train_config.get("report_to", "none"),
        "remove_unused_columns": False,
        "save_strategy": "steps",
        "eval_strategy": "steps",
    }
    if args.max_steps is not None:
        kwargs["max_steps"] = args.max_steps
    signature = inspect.signature(TrainingArguments.__init__)
    if "include_num_input_tokens_seen" in signature.parameters:
        kwargs["include_num_input_tokens_seen"] = True
    if "eval_strategy" not in signature.parameters:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    training_arguments = TrainingArguments(**kwargs)
    collator = AssistantOnlyDataCollator(
        tokenizer,
        max_length=int(train_config["max_seq_length"]),
    )
    write_run_manifest(
        output_dir,
        config,
        dataset_path=train_config["train_file"],
        extra={
            "model_revision": model_revision,
            "model_requested_revision": requested_revision,
            "train_examples": len(dataset["train"]),
            "validation_examples": len(dataset["validation"]),
        },
    )
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collator,
        callbacks=[VramMetricsCallback(), CheckpointManifestCallback(output_dir)],
    )
    resume: str | bool | None
    if args.resume_from_checkpoint == "auto":
        resume = find_latest_checkpoint(output_dir)
    elif args.resume_from_checkpoint.lower() in {"none", "false", "no"}:
        resume = None
    else:
        resume = args.resume_from_checkpoint
    print(f"resume checkpoint: {resume or 'none'}")
    train_result = trainer.train(resume_from_checkpoint=resume)
    train_metrics = dict(train_result.metrics)
    train_metrics["num_input_tokens_seen"] = getattr(
        trainer.state, "num_input_tokens_seen", None
    )
    train_metrics["total_flos"] = getattr(trainer.state, "total_flos", None)
    if torch.cuda.is_available():
        train_metrics["vram_peak_bytes"] = torch.cuda.max_memory_allocated()
        train_metrics["vram_peak_gib"] = round(
            torch.cuda.max_memory_allocated() / 2**30, 3
        )
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    finalize_run_manifest(output_dir, train_metrics)
    adapter_output = Path(
        train_config.get("adapter_output_dir", Path(output_dir).parent.parent / "adapters")
    )
    adapter_output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(adapter_output))
    tokenizer.save_pretrained(str(adapter_output))
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    if torch.cuda.is_available():
        print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
