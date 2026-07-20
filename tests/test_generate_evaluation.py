import argparse
import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_evaluation.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("generate_evaluation_module", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
generate_evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generate_evaluation)


def _args(**overrides):
    values = {
        "decoding": "benchmark",
        "mode": None,
        "max_new_tokens": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_deterministic_generation_is_direct_greedy_and_512_tokens():
    mode, maximum, generation = generate_evaluation._resolve_generation_args(
        _args(decoding="deterministic")
    )
    assert mode == "direct"
    assert maximum == 512
    assert generation == {
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
        "max_new_tokens": 512,
    }


def test_benchmark_defaults_remain_compatible():
    mode, maximum, generation = generate_evaluation._resolve_generation_args(_args())
    assert mode == "thinking"
    assert maximum == 2048
    assert generation["do_sample"] is True
    assert generation["temperature"] == 0.6


def test_deterministic_rejects_thinking_mode():
    with pytest.raises(ValueError, match="requer --mode direct"):
        generate_evaluation._resolve_generation_args(
            _args(decoding="deterministic", mode="thinking")
        )


def test_deterministic_rejects_a_different_token_budget():
    with pytest.raises(ValueError, match="requer --max-new-tokens 512"):
        generate_evaluation._resolve_generation_args(
            _args(decoding="deterministic", max_new_tokens=256)
        )
