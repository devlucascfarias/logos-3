import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_prepare_cli_with_local_candidates(tmp_path):
    categories = (
        "verified_code",
        "reasoning",
        "swe",
        "claude_code",
        "general_tools",
    )
    bands = ("short", "medium", "long", "direct")
    candidate_path = tmp_path / "candidates.jsonl"
    rows = []
    for category in categories:
        for band in bands:
            for index in range(4):
                rows.append(
                    {
                        "id": f"{category}-{band}-{index}",
                        "group_id": f"group-{category}-{band}-{index}",
                        "source": "fixture",
                        "category": category,
                        "reasoning_band": band,
                        "verified": True,
                        "num_tokens": 100,
                        "fingerprint": f"fp-{category}-{band}-{index}",
                        "messages": [
                            {"role": "user", "content": "tarefa"},
                            {"role": "assistant", "content": "resposta"},
                        ],
                    }
                )
    candidate_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    output_dir = tmp_path / "processed"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "prepare_data.py"),
            "--config",
            str(ROOT / "configs" / "recipe.yaml"),
            "--stage",
            "baseline",
            "--token-budget",
            "1000",
            "--candidates-jsonl",
            str(candidate_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(
        (output_dir / "dataset_report.json").read_text(encoding="utf-8")
    )
    assert (output_dir / "train.jsonl").exists()
    assert (output_dir / "validation.jsonl").exists()
    assert report["mix"]["budget_reached"]
    assert report["split"]["group_overlap"] is False
