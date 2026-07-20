import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


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

    callback.on_train_begin(None, state, None)
    state.global_step = 5
    callback.on_step_end(None, state, None)
    callback.on_log(
        None,
        state,
        None,
        logs={"loss": 1.25, "learning_rate": 0.0001},
    )
    callback.on_train_end(None, state, None)

    assert len(bars) == 1
    assert bars[0].description == "Treino baseline"
    assert bars[0].total == 30
    assert bars[0].n == 5
    assert bars[0].postfix == {"loss": "1.25", "lr": "0.0001"}
    assert bars[0].closed
