from __future__ import annotations

from evaluate_static import _generation_kwargs


def test_thinking_generation_uses_sampling_profile() -> None:
    config = {
        "max_new_tokens": 512,
        "repetition_penalty": 1.05,
        "profiles": {
            "thinking": {
                "do_sample": True,
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
            },
            "non_thinking": {
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
            },
        },
    }
    thinking = _generation_kwargs(config, "compressed")
    direct = _generation_kwargs(config, "hidden")
    assert thinking["do_sample"] is True
    assert thinking["temperature"] == 0.6
    assert thinking["top_p"] == 0.95
    assert direct["temperature"] == 0.7
    assert direct["top_p"] == 0.8
