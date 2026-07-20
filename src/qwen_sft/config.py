from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("A configuração deve ser um objeto YAML.")
    validate_config(config)
    return config


def _validate_distribution(name: str, values: dict[str, Any]) -> None:
    if not values:
        raise ValueError(f"{name} não pode ser vazio.")
    numeric = {key: float(value) for key, value in values.items()}
    if any(value < 0 for value in numeric.values()):
        raise ValueError(f"{name} não aceita pesos negativos.")
    total = sum(numeric.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"{name} deve somar 1.0; soma atual: {total:.8f}.")


def validate_config(config: dict[str, Any]) -> None:
    for key in ("model_name", "data", "stages", "training_defaults", "sources"):
        if key not in config:
            raise ValueError(f"Campo obrigatório ausente: {key}")

    data = config["data"]
    if int(data["max_seq_length"]) <= 0:
        raise ValueError("data.max_seq_length deve ser positivo.")
    if int(data.get("validation_min_examples", 0)) < 0:
        raise ValueError("data.validation_min_examples não pode ser negativo.")
    if int(data.get("max_examples_per_group", 0)) < 0:
        raise ValueError("data.max_examples_per_group não pode ser negativo.")
    if float(data.get("candidate_oversample", 1.0)) < 1:
        raise ValueError("data.candidate_oversample deve ser ao menos 1.")
    for key in (
        "repeated_line_min_chars",
        "max_repeated_line_occurrences",
        "repeated_ngram_size",
        "max_repeated_ngram_occurrences",
        "max_assistant_chars",
    ):
        if int(data.get(key, 1)) <= 0:
            raise ValueError(f"data.{key} deve ser positivo.")
    max_category_deviation = float(data.get("max_category_deviation", 1.0))
    if not 0 <= max_category_deviation <= 1:
        raise ValueError(
            "data.max_category_deviation deve estar no intervalo [0, 1]."
        )
    max_reasoning_deviation = float(data.get("max_reasoning_deviation", 1.0))
    if not 0 <= max_reasoning_deviation <= 1:
        raise ValueError(
            "data.max_reasoning_deviation deve estar no intervalo [0, 1]."
        )
    min_token_budget_fraction = float(
        data.get("min_token_budget_fraction", 0.0)
    )
    if not 0 <= min_token_budget_fraction <= 1:
        raise ValueError(
            "data.min_token_budget_fraction deve estar no intervalo [0, 1]."
        )
    for category, fraction in data.get(
        "direct_conversion_fraction_by_category", {}
    ).items():
        if not 0 <= float(fraction) <= 1:
            raise ValueError(
                "data.direct_conversion_fraction_by_category "
                f"possui valor inválido para {category}."
            )
    _validate_distribution(
        "data.reasoning_distribution", data["reasoning_distribution"]
    )

    stages = config["stages"]
    if not stages:
        raise ValueError("Ao menos uma etapa de treino deve ser configurada.")
    for stage_name, stage in stages.items():
        if int(stage["token_budget"]) <= 0:
            raise ValueError(f"stages.{stage_name}.token_budget deve ser positivo.")
        _validate_distribution(
            f"stages.{stage_name}.category_weights", stage["category_weights"]
        )
        learning_rate = float(stage["training"]["learning_rate"])
        if not 0 < learning_rate <= 1e-2:
            raise ValueError(
                f"Learning rate inválido em stages.{stage_name}: {learning_rate}"
            )
        data_stage = stage["training"].get("data_stage")
        if data_stage is not None and data_stage not in stages:
            raise ValueError(
                f"stages.{stage_name}.training.data_stage desconhecido: "
                f"{data_stage}"
            )

    categories = {
        category
        for stage in stages.values()
        for category in stage["category_weights"]
    }
    source_categories = {source["category"] for source in config["sources"]}
    missing = categories - source_categories
    if missing:
        raise ValueError(f"Categorias sem fonte de dados: {sorted(missing)}")

    for category in categories:
        sources = [
            source
            for source in config["sources"]
            if source["category"] == category
        ]
        total = sum(float(source.get("source_weight", 1.0)) for source in sources)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Os source_weight da categoria {category} devem somar 1.0; "
                f"soma atual: {total:.8f}."
            )


def merged_training_config(
    config: dict[str, Any], stage_name: str
) -> dict[str, Any]:
    if stage_name not in config["stages"]:
        choices = ", ".join(sorted(config["stages"]))
        raise ValueError(f"Etapa desconhecida: {stage_name}. Opções: {choices}")
    merged = dict(config["training_defaults"])
    merged.update(config["stages"][stage_name]["training"])
    return merged
