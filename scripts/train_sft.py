from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
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
        "--stage",
        choices=(
            "corrective_v1",
            "pilot",
            "pilot_continuation",
            "baseline",
            "main",
            "agentic",
        ),
        default="pilot",
    )
    parser.add_argument(
        "--data-stage",
        help="Etapa de dados a reutilizar; sobrescreve training.data_stage.",
    )
    parser.add_argument(
        "--adapter-path",
        help="Adapter inicial para continuação; sobrescreve training.adapter_path.",
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


def _verify_source_data(
    adapter_path: Path,
    train_file: Path,
    validation_file: Path,
) -> None:
    manifest_path = adapter_path / "run_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"Manifesto do adapter inicial ausente: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("data", {})
    checks = (
        (train_file, expected.get("train_sha256"), "treino"),
        (
            validation_file,
            expected.get("validation_sha256"),
            "validação",
        ),
        (
            train_file.parent / "dataset_report.json",
            expected.get("dataset_report_sha256"),
            "relatório",
        ),
    )
    for path, expected_hash, label in checks:
        if not expected_hash:
            raise SystemExit(
                f"Hash de {label} ausente no manifesto do adapter inicial."
            )
        actual_hash = file_sha256(path)
        if actual_hash != expected_hash:
            raise SystemExit(
                f"Os dados de {label} não correspondem ao treino original: "
                f"{path}"
            )


def _estimate_optimizer_steps(
    *,
    train_tokens: int,
    max_seq_length: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    num_train_epochs: float,
) -> dict[str, int | float]:
    if train_tokens <= 0:
        raise ValueError("train_tokens deve ser positivo.")
    if max_seq_length <= 0:
        raise ValueError("max_seq_length deve ser positivo.")
    if per_device_train_batch_size <= 0 or gradient_accumulation_steps <= 0:
        raise ValueError("Batch e acumulação devem ser positivos.")
    if num_train_epochs <= 0:
        raise ValueError("num_train_epochs deve ser positivo.")
    packed_sequences = math.ceil(train_tokens / max_seq_length)
    effective_batch = (
        per_device_train_batch_size * gradient_accumulation_steps
    )
    optimizer_steps = math.ceil(
        packed_sequences * num_train_epochs / effective_batch
    )
    return {
        "train_tokens": train_tokens,
        "packed_sequences": packed_sequences,
        "effective_batch_sequences": effective_batch,
        "num_train_epochs": num_train_epochs,
        "optimizer_steps": optimizer_steps,
    }


def _optimizer_step_preflight(
    report_path: Path,
    training: dict[str, Any],
    *,
    train_file: Path | None = None,
) -> dict[str, int | float]:
    if not report_path.exists():
        raise SystemExit(f"Relatório de dados ausente: {report_path}")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        train_tokens = int(report["split"]["train_tokens"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"Relatório de dados inválido: {report_path}") from exc
    if train_file is not None:
        actual_tokens = 0
        line_number = 0
        try:
            with train_file.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    actual_tokens += int(row["num_tokens"])
        except (
            OSError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise SystemExit(
                f"Dados de treino inválidos em {train_file}:{line_number}"
            ) from exc
        if actual_tokens != train_tokens:
            raise SystemExit(
                "A contagem de tokens do treino não corresponde ao relatório "
                f"({actual_tokens} != {train_tokens})."
            )
    estimate = _estimate_optimizer_steps(
        train_tokens=train_tokens,
        max_seq_length=int(training["max_seq_length"]),
        per_device_train_batch_size=int(
            training["per_device_train_batch_size"]
        ),
        gradient_accumulation_steps=int(
            training["gradient_accumulation_steps"]
        ),
        num_train_epochs=float(training["num_train_epochs"]),
    )
    minimum = training.get("min_optimizer_steps")
    maximum = training.get("max_optimizer_steps")
    steps = int(estimate["optimizer_steps"])
    if minimum is not None and steps < int(minimum):
        raise SystemExit(
            f"Preflight recusou o treino: estimativa de {steps} passos abaixo "
            f"do mínimo {minimum}."
        )
    if maximum is not None and steps > int(maximum):
        raise SystemExit(
            f"Preflight recusou o treino: estimativa de {steps} passos acima "
            f"do máximo {maximum}."
        )
    return estimate


def _make_progress_callback(base_class: type, stage: str):
    class TrainingProgressCallback(base_class):
        def __init__(self) -> None:
            self.progress = None
            self.gradient_accumulation_steps = 1

        def on_train_begin(self, args, state, control, **kwargs):
            if not state.is_world_process_zero:
                return
            from tqdm import tqdm

            self.gradient_accumulation_steps = max(
                int(args.gradient_accumulation_steps), 1
            )
            optimizer_steps = max(int(state.max_steps), 1)
            total = optimizer_steps * self.gradient_accumulation_steps
            initial = min(
                int(state.global_step) * self.gradient_accumulation_steps,
                total,
            )
            self.progress = tqdm(
                total=total,
                initial=initial,
                desc=f"Treino {stage}",
                unit="microbatch",
                dynamic_ncols=True,
                mininterval=0.5,
                smoothing=0.1,
                file=sys.stdout,
                bar_format=(
                    "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                    "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
                ),
            )
            print(
                f"Progresso: {total:,} microbatches "
                f"({optimizer_steps:,} passos × "
                f"{self.gradient_accumulation_steps} acumulações). "
                "O ETA aparece após os primeiros microbatches.",
                flush=True,
            )

        def on_substep_end(self, args, state, control, **kwargs):
            if self.progress is not None:
                self.progress.update(1)

        def on_step_end(self, args, state, control, **kwargs):
            if self.progress is None:
                return
            target = int(state.global_step) * self.gradient_accumulation_steps
            delta = target - int(self.progress.n)
            if delta > 0:
                self.progress.update(delta)

        def on_log(self, args, state, control, logs=None, **kwargs):
            if self.progress is None or not logs:
                return
            values = {}
            for key, label in (
                ("loss", "loss"),
                ("eval_loss", "eval"),
                ("learning_rate", "lr"),
                ("grad_norm", "grad"),
            ):
                value = logs.get(key)
                if isinstance(value, (int, float)):
                    values[label] = f"{value:.4g}"
            if values:
                self.progress.set_postfix(values, refresh=True)

        def on_train_end(self, args, state, control, **kwargs):
            if self.progress is None:
                return
            target = int(state.global_step) * self.gradient_accumulation_steps
            delta = target - int(self.progress.n)
            if delta > 0:
                self.progress.update(delta)
            self.progress.close()

    return TrainingProgressCallback()


def _manifest(
    *,
    config_path: str,
    stage: str,
    data_stage: str,
    train_file: Path,
    validation_file: Path,
    adapter_path: Path | None,
    training: dict[str, Any],
    torch: Any,
) -> dict[str, Any]:
    dataset_report_path = train_file.parent / "dataset_report.json"
    candidate_manifest = None
    if dataset_report_path.exists():
        try:
            dataset_report = json.loads(
                dataset_report_path.read_text(encoding="utf-8")
            )
            candidate_manifest = dataset_report.get("collection", {}).get(
                "candidate_manifest"
            )
        except (OSError, json.JSONDecodeError):
            candidate_manifest = None
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
            "stage": data_stage,
            "train_file": str(train_file),
            "train_sha256": file_sha256(train_file),
            "validation_file": str(validation_file),
            "validation_sha256": (
                file_sha256(validation_file) if validation_file.exists() else None
            ),
            "dataset_report_file": str(train_file.parent / "dataset_report.json"),
            "dataset_report_sha256": (
                file_sha256(train_file.parent / "dataset_report.json")
                if (train_file.parent / "dataset_report.json").exists()
                else None
            ),
            "candidate_manifest": candidate_manifest,
        },
        "source_adapter": (
            {
                "path": str(adapter_path),
                "config_sha256": file_sha256(
                    adapter_path / "adapter_config.json"
                ),
                "model_sha256": file_sha256(
                    adapter_path / "adapter_model.safetensors"
                ),
                "run_manifest_sha256": (
                    file_sha256(adapter_path / "run_manifest.json")
                    if (adapter_path / "run_manifest.json").exists()
                    else None
                ),
            }
            if adapter_path is not None
            else None
        ),
        "training": training,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    training = merged_training_config(config, args.stage)
    data_stage = str(
        args.data_stage or training.get("data_stage") or args.stage
    )
    adapter_path_value = args.adapter_path or training.get("adapter_path")
    adapter_path = Path(adapter_path_value) if adapter_path_value else None
    training["data_stage"] = data_stage
    training["adapter_path"] = str(adapter_path) if adapter_path else None
    train_file = Path(f"data/processed/{data_stage}/train.jsonl")
    validation_file = Path(f"data/processed/{data_stage}/validation.jsonl")
    output_dir = str(training["output_dir"])

    if args.stage == "corrective_v1" and args.max_steps is not None:
        raise SystemExit(
            "corrective_v1 não aceita --max-steps; preserve uma época completa."
        )
    if args.stage == "corrective_v1" and args.max_train_samples is not None:
        raise SystemExit(
            "corrective_v1 não aceita --max-train-samples; use o corpus assinado."
        )

    if args.dry_run:
        report_path = train_file.parent / "dataset_report.json"
        estimate = (
            _optimizer_step_preflight(
                report_path,
                training,
                train_file=train_file if train_file.exists() else None,
            )
            if report_path.exists()
            else None
        )
        print(
            json.dumps(
                {
                    "stage": args.stage,
                    "data_stage": data_stage,
                    "train_file": str(train_file),
                    "train_exists": train_file.exists(),
                    "validation_file": str(validation_file),
                    "validation_exists": validation_file.exists(),
                    "output_dir": output_dir,
                    "adapter_path": str(adapter_path) if adapter_path else None,
                    "resume": _resolve_resume(
                        args.resume_from_checkpoint, output_dir
                    ),
                    "optimizer_step_estimate": estimate,
                    "training": training,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    requires_adapter = args.stage == "agentic" or training.get(
        "requires_adapter"
    )
    if requires_adapter and not adapter_path:
        raise SystemExit(
            f"A etapa {args.stage} exige um adapter inicial. Use "
            "--adapter-path ou defina training.adapter_path na configuração."
        )
    if (
        args.stage == "corrective_v1"
        and adapter_path is not None
        and not (adapter_path / "run_manifest.json").exists()
    ):
        raise SystemExit(
            "corrective_v1 exige run_manifest.json no adapter campeão inicial."
        )
    if adapter_path:
        if not (adapter_path / "adapter_config.json").exists() or not (
            adapter_path / "adapter_model.safetensors"
        ).exists():
            raise SystemExit(
                f"Adapter para continuação não encontrado: {adapter_path}"
            )
        if adapter_path.resolve() == Path(
            training["adapter_output_dir"]
        ).resolve():
            raise SystemExit(
                "O adapter inicial e o diretório de saída devem ser diferentes."
            )
    if not train_file.exists() or train_file.stat().st_size == 0:
        raise SystemExit(
            f"Dados ausentes ou vazios: {train_file}. A continuação deve "
            "reutilizar os dados já aprovados; restaure-os antes do treino."
        )
    if adapter_path and training.get("verify_source_data"):
        _verify_source_data(adapter_path, train_file, validation_file)
    optimizer_step_estimate = _optimizer_step_preflight(
        train_file.parent / "dataset_report.json",
        training,
        train_file=train_file,
    )
    training["optimizer_step_estimate"] = optimizer_step_estimate
    print(
        "Preflight: "
        f"{optimizer_step_estimate['train_tokens']:,} tokens, "
        f"{optimizer_step_estimate['packed_sequences']:,} sequências, "
        f"{optimizer_step_estimate['optimizer_steps']:,} passos estimados.",
        flush=True,
    )

    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from transformers.trainer_callback import PrinterCallback, TrainerCallback
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

    print("[1/4] Carregando tokenizer e dados...", flush=True)
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
    use_eval = (
        validation_file.exists()
        and validation_file.stat().st_size > 0
        and not args.no_eval
    )
    if use_eval:
        data_files["validation"] = str(validation_file)
    dataset = load_dataset("json", data_files=data_files)
    if args.max_train_samples:
        size = min(args.max_train_samples, len(dataset["train"]))
        dataset["train"] = dataset["train"].select(range(size))
    print(
        f"[2/4] Dados prontos: {len(dataset['train']):,} exemplos de treino"
        + (
            f" e {len(dataset['validation']):,} de validação."
            if use_eval
            else "."
        ),
        flush=True,
    )

    print("[3/4] Carregando modelo 4-bit e configurando LoRA...", flush=True)
    peft_config = None
    model: str | Any = config["model_name"]
    trainer_quantization = quantization
    if adapter_path:
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
        model = PeftModel.from_pretrained(
            base, str(adapter_path), is_trainable=True
        )
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
        "disable_tqdm": True,
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
        data_stage=data_stage,
        train_file=train_file,
        validation_file=validation_file,
        adapter_path=adapter_path,
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
        "callbacks": [_make_progress_callback(TrainerCallback, args.stage)],
    }
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config
    if trainer_quantization is not None:
        trainer_kwargs["quantization_config"] = trainer_quantization
    trainer = SFTTrainer(**trainer_kwargs)
    trainer.remove_callback(PrinterCallback)

    resume = _resolve_resume(args.resume_from_checkpoint, output_dir)
    print(f"Retomada: {resume or 'não'}")
    print("[4/4] Treinamento iniciado.", flush=True)
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
    (adapter_output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"Adapter salvo em: {adapter_output}")
    print(f"Pico de VRAM: {metrics['peak_vram_gib']:.2f} GiB")


if __name__ == "__main__":
    main()
