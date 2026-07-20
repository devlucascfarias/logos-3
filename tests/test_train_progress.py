import importlib.util
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_sft.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("train_sft_module", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
train_sft = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train_sft)


def test_progress_callback_tracks_steps_and_metrics(monkeypatch):
    bars = []

    class FakeBar:
        def __init__(self, **kwargs):
            self.n = kwargs["initial"]
            self.total = kwargs["total"]
            self.description = kwargs["desc"]
            self.postfix = {}
            self.closed = False
            bars.append(self)

        def update(self, amount):
            self.n += amount

        def set_postfix(self, values, refresh):
            self.postfix = values

        def close(self):
            self.closed = True

    monkeypatch.setitem(sys.modules, "tqdm", SimpleNamespace(tqdm=FakeBar))

    class BaseCallback:
        pass

    callback = train_sft._make_progress_callback(BaseCallback, "baseline")
    state = SimpleNamespace(
        is_world_process_zero=True,
        max_steps=30,
        global_step=0,
    )
    args = SimpleNamespace(gradient_accumulation_steps=16)

    callback.on_train_begin(args, state, None)
    callback.on_substep_end(args, state, None)
    state.global_step = 1
    callback.on_step_end(args, state, None)
    callback.on_log(
        args,
        state,
        None,
        logs={"loss": 1.25, "learning_rate": 0.0001},
    )
    callback.on_train_end(args, state, None)

    assert len(bars) == 1
    assert bars[0].description == "Treino baseline"
    assert bars[0].total == 480
    assert bars[0].n == 16
    assert bars[0].postfix == {"loss": "1.25", "lr": "0.0001"}
    assert bars[0].closed


def test_source_data_verification_detects_drift(tmp_path):
    data_dir = tmp_path / "data" / "processed" / "pilot"
    adapter_dir = tmp_path / "adapter"
    data_dir.mkdir(parents=True)
    adapter_dir.mkdir()
    train_file = data_dir / "train.jsonl"
    validation_file = data_dir / "validation.jsonl"
    report_file = data_dir / "dataset_report.json"
    train_file.write_text('{"messages": []}\n', encoding="utf-8")
    validation_file.write_text('{"messages": []}\n', encoding="utf-8")
    report_file.write_text("{}\n", encoding="utf-8")

    def sha256(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    (adapter_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "data": {
                    "train_sha256": sha256(train_file),
                    "validation_sha256": sha256(validation_file),
                    "dataset_report_sha256": sha256(report_file),
                }
            }
        ),
        encoding="utf-8",
    )

    train_sft._verify_source_data(
        adapter_dir, train_file, validation_file
    )
    train_file.write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="não correspondem"):
        train_sft._verify_source_data(
            adapter_dir, train_file, validation_file
        )


def test_continuation_arguments_are_supported(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_sft.py",
            "--stage",
            "pilot_continuation",
            "--data-stage",
            "pilot",
            "--adapter-path",
            "/tmp/adapter",
        ],
    )

    args = train_sft.parse_args()

    assert args.stage == "pilot_continuation"
    assert args.data_stage == "pilot"
    assert args.adapter_path == "/tmp/adapter"
