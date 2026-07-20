import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compare_adapter.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("compare_adapter_module", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
compare_adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare_adapter)


def test_prompt_fixture_is_valid():
    prompts = compare_adapter._load_prompts(
        ROOT / "examples" / "smoke_eval_prompts.json"
    )

    assert len(prompts) == 13
    assert len({item["id"] for item in prompts}) == 13


def test_blind_results_are_deterministic_and_complete():
    results = [
        {
            "id": "one",
            "title": "One",
            "prompt": "Prompt",
            "base": {"text": "base answer", "seconds": 1.0},
            "adapter": {"text": "adapter answer", "seconds": 2.0},
        }
    ]

    first, first_mapping = compare_adapter._blind_results(results, seed=42)
    second, second_mapping = compare_adapter._blind_results(results, seed=42)

    assert first == second
    assert first_mapping == second_mapping
    assert {first[0]["A"], first[0]["B"]} == {
        "base answer",
        "adapter answer",
    }
    assert {first_mapping[0]["A"], first_mapping[0]["B"]} == {
        "base",
        "adapter",
    }


def test_rendered_outputs_include_rubric():
    comparisons = [
        {
            "id": "one",
            "title": "One",
            "prompt": "Prompt",
            "A": "Answer A",
            "B": "Answer B",
            "generation_seconds": {"A": 1.0, "B": 2.0},
        }
    ]

    markdown = compare_adapter._render_markdown(comparisons)
    ratings = compare_adapter._rating_template(comparisons)
    serialized = json.dumps(ratings)

    assert "Resposta A" in markdown
    assert "Correção (0–5)" in markdown
    assert "instruction_following" in serialized


def test_blind_results_support_reference_adapter():
    results = [
        {
            "id": "one",
            "title": "One",
            "prompt": "Prompt",
            "base": {"text": "base answer", "seconds": 1.0},
            "adapter": {"text": "candidate answer", "seconds": 2.0},
            "reference": {"text": "reference answer", "seconds": 3.0},
        }
    ]

    comparisons, mapping = compare_adapter._blind_results(results, seed=7)
    labels = {"A", "B", "C"}

    assert labels <= comparisons[0].keys()
    assert {comparisons[0][label] for label in labels} == {
        "base answer",
        "candidate answer",
        "reference answer",
    }
    assert {mapping[0][label] for label in labels} == {
        "base",
        "adapter",
        "reference",
    }
    assert "Resposta C" in compare_adapter._render_markdown(comparisons)
    assert "C" in compare_adapter._rating_template(comparisons)[0]["ratings"]
