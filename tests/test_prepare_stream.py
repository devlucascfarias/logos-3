import importlib.util
import io
import json
import sys
from collections import UserDict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_data.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("prepare_data", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
prepare_data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_data)


def test_encoded_token_count_reads_batch_encoding_input_ids():
    encoded = UserDict(
        {
            "input_ids": list(range(40)),
            "attention_mask": [1] * 40,
        }
    )

    assert prepare_data._encoded_token_count(encoded) == 40
    assert prepare_data._encoded_token_count(list(range(12))) == 12


def test_buffered_shuffle_is_deterministic():
    rows = [{"id": index} for index in range(20)]

    first = list(prepare_data._buffered_shuffle(rows, seed=42, buffer_size=4))
    second = list(prepare_data._buffered_shuffle(rows, seed=42, buffer_size=4))

    assert first == second
    assert sorted(row["id"] for row in first) == list(range(20))
    assert first != rows


def test_raw_jsonl_stream_uses_pinned_revision(monkeypatch):
    rows = [{"id": "one"}, {"id": "two"}]
    payload = "".join(json.dumps(row) + "\n" for row in rows)
    opened = {}

    class FakeHfFileSystem:
        def __init__(self, token=None):
            opened["token"] = token

        def open(self, path, mode, encoding):
            opened.update(path=path, mode=mode, encoding=encoding)
            return io.StringIO(payload)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfFileSystem", FakeHfFileSystem)
    monkeypatch.setenv("HF_TOKEN", "test-token")
    source = {
        "name": "owner/dataset",
        "revision": "abc123",
        "raw_jsonl_file": "data/train.jsonl",
        "shuffle_buffer": 1,
    }

    result = list(prepare_data._load_stream(source, seed=42, shuffle_buffer=10))

    assert result == rows
    assert opened == {
        "token": "test-token",
        "path": "datasets/owner/dataset@abc123/data/train.jsonl",
        "mode": "r",
        "encoding": "utf-8",
    }
