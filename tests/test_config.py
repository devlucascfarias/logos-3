from pathlib import Path

import pytest

from qwen_sft.config import (
    load_config,
    merged_data_config,
    merged_training_config,
    validate_config,
)


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
    assert config["data"]["validation_min_examples"] == 32
    assert config["data"]["min_token_budget_fraction"] == 0.95
    assert config["data"]["reasoning_distribution"]["direct"] == 0.45
    pilot = merged_training_config(config, "pilot")
    assert config["stages"]["pilot"]["token_budget"] == 500_000
    assert pilot["learning_rate"] == 5e-5
    assert pilot["save_steps"] == 1
    assert pilot["eval_steps"] == 1
    continuation = merged_training_config(config, "pilot_continuation")
    assert continuation["data_stage"] == "pilot"
    assert continuation["requires_adapter"] is True
    assert continuation["verify_source_data"] is True
    assert continuation["adapter_path"] is None
    assert continuation["learning_rate"] == 2e-5
    assert continuation["num_train_epochs"] == 1
    assert continuation["output_dir"].endswith("pilot_continuation")
    corrective = merged_training_config(config, "corrective_v1")
    corrective_data = merged_data_config(config, "corrective_v1")
    assert config["stages"]["corrective_v1"]["token_budget"] == 320_000
    assert config["stages"]["corrective_v1"]["category_weights"] == {
        "corrective_contracts": 0.50,
        "corrective_opencode": 0.35,
        "corrective_replay": 0.15,
    }
    assert corrective_data["max_seq_length"] == 2048
    assert corrective_data["candidate_oversample"] == 1.25
    assert corrective_data["reasoning_distribution"] == {"direct": 1.0}
    assert corrective["requires_adapter"] is True
    assert corrective["verify_source_data"] is False
    assert corrective["learning_rate"] == 1e-5
    assert corrective["gradient_accumulation_steps"] == 8
    assert corrective["save_steps"] == 2
    assert corrective["eval_steps"] == 2
    assert corrective["min_optimizer_steps"] == 15
    assert corrective["max_optimizer_steps"] == 25
    assert corrective["output_dir"].endswith("corrective_v1")


def test_invalid_distribution_is_rejected():
    config = load_config(ROOT / "configs" / "recipe.yaml")
    config["stages"]["baseline"]["category_weights"]["verified_code"] = 0.5
    with pytest.raises(ValueError, match="somar 1.0"):
        validate_config(config)
