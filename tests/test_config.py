from pathlib import Path

import pytest

from qwen_sft.config import load_config, merged_training_config, validate_config


ROOT = Path(__file__).resolve().parents[1]


def test_recipe_is_valid_and_matches_core_parameters():
    config = load_config(ROOT / "configs" / "recipe.yaml")
    training = merged_training_config(config, "main")

    assert config["model_name"] == "Qwen/Qwen3-8B"
    assert training["max_seq_length"] == 4096
    assert training["lora_r"] == 32
    assert training["gradient_accumulation_steps"] == 16
    assert training["bnb_4bit_quant_type"] == "nf4"
    assert config["stages"]["main"]["category_weights"]["verified_code"] == 0.45


def test_invalid_distribution_is_rejected():
    config = load_config(ROOT / "configs" / "recipe.yaml")
    config["stages"]["baseline"]["category_weights"]["verified_code"] = 0.5
    with pytest.raises(ValueError, match="somar 1.0"):
        validate_config(config)
