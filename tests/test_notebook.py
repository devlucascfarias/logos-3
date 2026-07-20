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
    assert 'STAGE = "pilot_continuation"' in source
    assert 'DATA_STAGE = "pilot"' in source
    assert "RUN_DATA_PREPARATION = False" in source
    assert '"-u", "scripts/train_sft.py"' in source
    assert "barra de progresso e ETA" in source
    assert "subprocess.Popen" in source
    assert "os.read(process.stdout.fileno(), 4096)" in source
    assert "Log salvo em:" in source
    assert "compare_adapter.py" in source
    assert "comparison.md" in source
    assert "mapping.json" in source
    assert "FRESH_RUN" in source
    assert "outputs\" / \"archive" in source
    assert "REFERENCE_ADAPTER_PATH" in source
    assert "SOURCE_ADAPTER_PATH" in source
    assert "DATA_BACKUP_PATH" in source
    assert '"--data-stage", DATA_STAGE' in source
    assert '"--adapter-path", SOURCE_ADAPTER_PATH' in source
    assert "Dados pilot verificados por hash" in source
    assert 'drive.mount("/content/drive")' in source
    assert "pilot_continuation" in source
