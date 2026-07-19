from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from importlib import metadata as importlib_metadata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import file_sha256


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def environment_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": git_commit(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch

        snapshot.update(
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "bf16_supported": (
                    torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                ),
            }
        )
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            snapshot.update(
                {
                    "gpu": torch.cuda.get_device_name(0),
                    "vram_total_bytes": total,
                    "vram_free_bytes": free,
                }
            )
    except ImportError:
        snapshot["torch"] = "not-installed"
    packages = {}
    for name in (
        "transformers",
        "accelerate",
        "peft",
        "bitsandbytes",
        "datasets",
        "trl",
        "safetensors",
    ):
        try:
            packages[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    snapshot["packages"] = packages
    return snapshot


def write_run_manifest(
    output_dir: str | Path,
    config: dict[str, Any],
    dataset_path: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        **environment_snapshot(),
        "config": config,
        "dataset": str(dataset_path) if dataset_path else None,
        "dataset_sha256": (
            file_sha256(dataset_path) if dataset_path and Path(dataset_path).is_file() else None
        ),
    }
    if extra:
        manifest.update(extra)
    output = destination / "run_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


try:
    from transformers import TrainerCallback
except ImportError:  # pragma: no cover
    class TrainerCallback:  # type: ignore[no-redef]
        pass


class VramMetricsCallback(TrainerCallback):
    def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **kwargs: Any) -> None:
        try:
            import torch

            if torch.cuda.is_available() and logs is not None:
                logs["vram_peak_gib"] = round(
                    torch.cuda.max_memory_allocated() / (1024**3), 3
                )
        except ImportError:
            return


class CheckpointManifestCallback(TrainerCallback):
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        source = self.output_dir / "run_manifest.json"
        checkpoint = self.output_dir / f"checkpoint-{state.global_step}"
        if not source.exists() or not checkpoint.exists():
            return
        try:
            manifest = json.loads(source.read_text(encoding="utf-8"))
            manifest["checkpoint"] = {
                "global_step": state.global_step,
                "epoch": state.epoch,
                "num_input_tokens_seen": getattr(state, "num_input_tokens_seen", None),
                "total_flos": getattr(state, "total_flos", None),
            }
            try:
                import torch

                if torch.cuda.is_available():
                    manifest["checkpoint"]["vram_peak_bytes"] = (
                        torch.cuda.max_memory_allocated()
                    )
            except ImportError:
                pass
            (checkpoint / "run_manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except (OSError, json.JSONDecodeError):
            return


def finalize_run_manifest(output_dir: str | Path, metrics: dict[str, Any]) -> None:
    path = Path(output_dir) / "run_manifest.json"
    if not path.exists():
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["final_metrics"] = metrics
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
