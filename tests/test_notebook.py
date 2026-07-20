import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_colab_notebook_is_valid_and_references_pipeline():
    path = ROOT / "notebooks" / "qwen3_8b_l4_sft_colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    first_cell = notebook["cells"][0]
    assert first_cell["cell_type"] == "code"
    assert '"pull", "--ff-only"' in "".join(first_cell["source"])
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "prepare_data.py" in source
    assert "train_sft.py" in source
    assert 'STAGE = "corrective_v1"' in source
    assert 'DATA_STAGE = "corrective_v1"' in source
    assert "RUN_DATA_PREPARATION = True" in source
    assert "build_corrective_corpus.py" in source
    assert 'TOKEN_BUDGET = 320_000' in source
    assert "CANDIDATE_MANIFEST" in source
    assert '"-u", "scripts/train_sft.py"' in source
    assert "barra de progresso e ETA" in source
    assert "subprocess.Popen" in source
    assert "os.read(process.stdout.fileno(), 4096)" in source
    assert "Log salvo em:" in source
    assert "compare_adapter.py" in source
    assert "generate_evaluation.py" in source
    assert "evaluate_contracts.py" in source
    assert "dev_v1.jsonl" in source
    assert "corrective_hidden_v1.jsonl" in source
    assert '"--decoding", "deterministic"' in source
    assert "comparison.md" in source
    assert "mapping.json" in source
    assert "FRESH_RUN" in source
    assert "outputs/archive" in source
    assert "REFERENCE_ADAPTER_PATH" in source
    assert "SOURCE_ADAPTER_PATH" in source
    assert "LOCAL_PILOT_PATH" in source
    assert '"--data-stage", DATA_STAGE' in source
    assert '"--adapter-path", SOURCE_ADAPTER_PATH' in source
    assert "Hash do replay divergente ou ausente" in source
    assert 'drive.mount("/content/drive")' in source
    assert "drive.flush_and_unmount()" in source
    assert '"--resume-from-checkpoint", "none" if fresh_run else "auto"' in source
    assert "15–25 passos" in source
    assert "TOP_THREE" in source
    assert "promotion_gate.json" in source
    assert "/content/drive/MyDrive/logos-3/runs/corrective_v1" in source
