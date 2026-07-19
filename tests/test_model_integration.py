from __future__ import annotations

import os
from pathlib import Path

import pytest


RUN_MODEL_TESTS = os.environ.get("FABLE_RUN_MODEL_TESTS") == "1"
pytestmark = pytest.mark.skipif(
    not RUN_MODEL_TESTS,
    reason="set FABLE_RUN_MODEL_TESTS=1 on a CUDA runtime for model integration tests",
)


def test_quantized_forward_and_short_generation() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    assert torch.cuda.is_available()
    name = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ),
        device_map={"": 0},
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Return one valid shell tool call."}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(model.device)
    with torch.inference_mode():
        generated = model.generate(**encoded, max_new_tokens=8)
    assert generated.shape[1] > encoded["input_ids"].shape[1]


def test_adapter_can_be_reloaded_when_provided() -> None:
    adapter_path = os.environ.get("FABLE_TEST_ADAPTER")
    if not adapter_path:
        pytest.skip("set FABLE_TEST_ADAPTER to test adapter reload/export")
    from peft import PeftConfig

    config = PeftConfig.from_pretrained(Path(adapter_path))
    assert config.base_model_name_or_path
