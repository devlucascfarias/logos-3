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


def test_corrective_optimizer_step_preflight_accepts_expected_range(tmp_path):
    report_path = tmp_path / "dataset_report.json"
    report_path.write_text(
        json.dumps({"split": {"train_tokens": 304_000}}),
        encoding="utf-8",
    )
    estimate = train_sft._optimizer_step_preflight(
        report_path,
        {
            "max_seq_length": 2048,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "num_train_epochs": 1,
            "min_optimizer_steps": 15,
            "max_optimizer_steps": 25,
        },
    )
    assert estimate["packed_sequences"] == 149
    assert estimate["optimizer_steps"] == 19


def test_corrective_optimizer_step_preflight_rejects_outside_range(tmp_path):
    report_path = tmp_path / "dataset_report.json"
    report_path.write_text(
        json.dumps({"split": {"train_tokens": 100_000}}),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="abaixo do mínimo"):
        train_sft._optimizer_step_preflight(
            report_path,
            {
                "max_seq_length": 2048,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 8,
                "num_train_epochs": 1,
                "min_optimizer_steps": 15,
                "max_optimizer_steps": 25,
            },
        )


def test_corrective_arguments_are_supported(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_sft.py",
            "--stage",
            "corrective_v1",
            "--adapter-path",
            "/tmp/champion",
            "--resume-from-checkpoint",
            "none",
        ],
    )
    args = train_sft.parse_args()
    assert args.stage == "corrective_v1"
    assert args.adapter_path == "/tmp/champion"
    assert args.resume_from_checkpoint == "none"


def test_manifest_records_initial_adapter_manifest_hash(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    for name, content in (
        ("adapter_config.json", "{}"),
        ("adapter_model.safetensors", "weights"),
        ("run_manifest.json", "{}"),
    ):
        (adapter / name).write_text(content, encoding="utf-8")
    train_file = tmp_path / "train.jsonl"
    train_file.write_text("{}\n", encoding="utf-8")
    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            get_device_name=lambda _: "GPU",
            is_bf16_supported=lambda: True,
            get_device_properties=lambda _: SimpleNamespace(total_memory=1),
        )
    )
    manifest = train_sft._manifest(
        config_path="config.yaml",
        stage="corrective_v1",
        data_stage="corrective_v1",
        train_file=train_file,
        validation_file=tmp_path / "validation.jsonl",
        adapter_path=adapter,
        training={},
        torch=torch,
    )
    assert manifest["source_adapter"]["run_manifest_sha256"] == hashlib.sha256(
        (adapter / "run_manifest.json").read_bytes()
    ).hexdigest()
