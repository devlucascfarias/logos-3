from __future__ import annotations

import argparse
import inspect

import _bootstrap  # noqa: F401

from fable_distill.callbacks import set_global_seed, write_run_manifest
from fable_distill.io import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA DPO training after a stable SFT")
    parser.add_argument("--config", default="configs/preference.yaml")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if args.dry_run:
        print(config)
        return
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        raise SystemExit("Install training dependencies: pip install -r requirements.txt") from exc
    if not torch.cuda.is_available():
        raise SystemExit("DPO for the 8B model requires a CUDA runtime")

    model_cfg, data_cfg, train_cfg = config["model"], config["data"], config["training"]
    seed = int(train_cfg.get("seed", 42))
    set_global_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name"],
        trust_remote_code=True,
        use_fast=True,
    )
    if not getattr(tokenizer, "chat_template", None):
        raise SystemExit("The selected tokenizer does not provide an official chat template")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=model_cfg.get("quant_type", "nf4"),
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=bool(model_cfg.get("double_quant", True)),
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        quantization_config=quantization,
        device_map={"": 0},
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    adapter = model_cfg.get("adapter_path")
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
    data = load_dataset(
        "json",
        data_files={
            "train": data_cfg["train_file"],
            "validation": data_cfg["validation_file"],
        },
    )
    dpo_kwargs = {
        "output_dir": train_cfg["output_dir"],
        "per_device_train_batch_size": int(train_cfg["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(train_cfg["gradient_accumulation_steps"]),
        "learning_rate": float(train_cfg["learning_rate"]),
        "num_train_epochs": float(train_cfg["num_train_epochs"]),
        "beta": float(train_cfg["beta"]),
        "optim": train_cfg["optim"],
        "bf16": bool(train_cfg.get("bf16", True)),
        "gradient_checkpointing": bool(train_cfg.get("gradient_checkpointing", True)),
        "save_steps": int(train_cfg["save_steps"]),
        "logging_steps": int(train_cfg["logging_steps"]),
        "seed": seed,
        "remove_unused_columns": False,
        "report_to": "none",
    }
    signature = inspect.signature(DPOConfig.__init__)
    if "max_length" in signature.parameters:
        dpo_kwargs["max_length"] = int(train_cfg["max_length"])
        dpo_kwargs["max_prompt_length"] = int(train_cfg["max_prompt_length"])
    training_args = DPOConfig(**dpo_kwargs)
    trainer_kwargs = {
        "model": model,
        "ref_model": None,
        "args": training_args,
        "train_dataset": data["train"],
        "eval_dataset": data["validation"],
    }
    trainer_signature = inspect.signature(DPOTrainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    if not adapter:
        trainer_kwargs["peft_config"] = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        )
    write_run_manifest(
        train_cfg["output_dir"],
        config,
        data_cfg["train_file"],
        {"training_method": "DPO"},
    )
    trainer = DPOTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(train_cfg["output_dir"])
    tokenizer.save_pretrained(train_cfg["output_dir"])


if __name__ == "__main__":
    main()
