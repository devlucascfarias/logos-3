from __future__ import annotations

import json
import platform
from importlib.metadata import PackageNotFoundError, version

import _bootstrap  # noqa: F401

from qwen_sft.config import load_config


def main() -> None:
    config = load_config("configs/recipe.yaml")
    packages = {}
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
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    result = {
        "python": platform.python_version(),
        "packages": packages,
        "model": config["model_name"],
        "cuda": False,
        "gpu": None,
        "bf16": False,
        "vram_gib": 0.0,
    }
    try:
        import torch

        result["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            result["gpu"] = torch.cuda.get_device_name(0)
            result["bf16"] = torch.cuda.is_bf16_supported()
            result["vram_gib"] = round(
                torch.cuda.get_device_properties(0).total_memory / 2**30, 2
            )
    except ImportError:
        pass
    print(json.dumps(result, indent=2))

    errors = []
    if not result["cuda"]:
        errors.append("CUDA indisponível")
    if result["cuda"] and not result["bf16"]:
        errors.append("BF16 indisponível")
    if result["gpu"] and "L4" not in str(result["gpu"]).upper():
        errors.append("GPU diferente da L4 dimensionada na receita")
    if any(value is None for value in packages.values()):
        errors.append("dependências de treino ausentes")
    if errors:
        raise SystemExit("Ambiente não pronto: " + "; ".join(errors))
    print("Ambiente pronto para o treino.")


if __name__ == "__main__":
    main()
