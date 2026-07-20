from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


RUNNER_VERSION = "corrective-sandbox-v1"
PYTHON_BLOCK_RE = re.compile(
    r"```(?:python|py)\s*\n(?P<code>.*?)```", re.IGNORECASE | re.DOTALL
)
ANY_FENCE_RE = re.compile(r"```[^\n]*\n.*?```", re.DOTALL)


class SandboxViolation(ValueError):
    """Raised when untrusted source violates the corrective corpus policy."""


@dataclass(frozen=True)
class ExecutionResult:
    passed: bool
    status: str
    wall_ms: int
    stdout: str = ""
    stderr: str = ""

    def evidence(self, code: str, tests: str) -> dict[str, Any]:
        code_hash = _sha256_text(code)
        tests_hash = _sha256_text(tests)
        payload = {
            "status": "passed" if self.passed else self.status,
            "code_sha256": code_hash,
            "tests_sha256": tests_hash,
            "runner_version": RUNNER_VERSION,
            "wall_ms": self.wall_ms,
        }
        payload["sha256"] = _sha256_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":"))
        )
        return payload

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def extract_single_python_block(text: str) -> str:
    matches = list(PYTHON_BLOCK_RE.finditer(text))
    all_fences = list(ANY_FENCE_RE.finditer(text))
    if len(matches) != 1 or len(all_fences) != 1:
        raise SandboxViolation("A resposta deve conter exatamente um bloco Python.")
    code = matches[0].group("code").strip()
    if not code:
        raise SandboxViolation("O bloco Python est\u00e1 vazio.")
    return code


_BANNED_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.ClassDef,
    ast.AsyncFunctionDef,
    ast.With,
    ast.AsyncWith,
    ast.Global,
)
_BANNED_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "memoryview",
    "open",
    "setattr",
    "vars",
}


def validate_source(source: str, *, label: str = "code") -> None:
    try:
        tree = ast.parse(source, filename=f"<{label}>", mode="exec")
    except SyntaxError as exc:
        raise SandboxViolation(f"Python inv\u00e1lido em {label}: {exc.msg}") from exc

    for node in ast.walk(tree):
        if isinstance(node, _BANNED_NODES):
            raise SandboxViolation(
                f"{type(node).__name__} n\u00e3o \u00e9 permitido em {label}."
            )
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise SandboxViolation("Acesso dunder n\u00e3o \u00e9 permitido.")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise SandboxViolation("Nomes dunder n\u00e3o s\u00e3o permitidos.")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _BANNED_CALLS:
                raise SandboxViolation(
                    f"Chamada a {node.func.id} n\u00e3o \u00e9 permitida."
                )


_HARNESS = r'''
import sys

LIMIT = int(sys.argv[1])
SOURCE_PATH = sys.argv[2]

class LimitedWriter:
    def __init__(self, stream):
        self.stream = stream
        self.count = 0
    def write(self, value):
        value = str(value)
        self.count += len(value.encode("utf-8", "replace"))
        if self.count > LIMIT:
            raise RuntimeError("sandbox output limit exceeded")
        return self.stream.write(value)
    def flush(self):
        return self.stream.flush()

sys.stdout = LimitedWriter(sys.stdout)
sys.stderr = LimitedWriter(sys.stderr)

allowed = {
    "ArithmeticError": ArithmeticError, "AssertionError": AssertionError,
    "BaseException": BaseException, "Exception": Exception,
    "IndexError": IndexError, "KeyError": KeyError,
    "LookupError": LookupError, "RuntimeError": RuntimeError,
    "StopIteration": StopIteration, "TypeError": TypeError,
    "ValueError": ValueError, "ZeroDivisionError": ZeroDivisionError,
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "enumerate": enumerate, "filter": filter, "float": float,
    "int": int, "isinstance": isinstance, "iter": iter, "len": len,
    "list": list, "map": map, "max": max, "min": min, "next": next,
    "object": object, "pow": pow, "range": range, "reversed": reversed,
    "print": print, "round": round, "set": set, "slice": slice, "sorted": sorted,
    "str": str, "sum": sum, "tuple": tuple, "zip": zip,
}
namespace = {"__builtins__": allowed}
with open(SOURCE_PATH, "r", encoding="utf-8") as handle:
    source = handle.read()
exec(compile(source, SOURCE_PATH, "exec"), namespace, namespace)
'''


def _posix_limits(memory_bytes: int, cpu_seconds: int):
    def apply_limits() -> None:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        resource.setrlimit(resource.RLIMIT_NOFILE, (16, 16))

    return apply_limits


def run_verified(
    code: str,
    tests: str,
    *,
    timeout_seconds: float = 3.0,
    cpu_seconds: int = 2,
    memory_mib: int = 256,
    output_limit: int = 64 * 1024,
) -> ExecutionResult:
    """Execute approved Python in an isolated process.

    The training workflow targets Linux/Colab.  Other platforms are rejected
    because they cannot provide the resource limits promised by the manifest.
    """

    validate_source(code, label="solution")
    validate_source(tests, label="tests")
    if os.name != "posix":
        raise SandboxViolation(
            "O executor verificado requer POSIX (use o runtime Linux do Colab)."
        )
    if timeout_seconds <= 0 or cpu_seconds <= 0 or memory_mib <= 0:
        raise ValueError("Os limites do sandbox devem ser positivos.")

    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="corrective-sandbox-") as directory:
        path = Path(directory) / "candidate.py"
        path.write_text(code.rstrip() + "\n\n" + tests.strip() + "\n", encoding="utf-8")
        command = [
            sys.executable,
            "-I",
            "-S",
            "-c",
            _HARNESS,
            str(output_limit),
            str(path),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=directory,
                env={"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8"},
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_seconds,
                preexec_fn=_posix_limits(memory_mib * 1024 * 1024, cpu_seconds),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecutionResult(
                False,
                "timeout",
                int((time.monotonic() - started) * 1000),
                (exc.stdout or "")[-output_limit:] if isinstance(exc.stdout, str) else "",
                (exc.stderr or "")[-output_limit:] if isinstance(exc.stderr, str) else "",
            )

    wall_ms = int((time.monotonic() - started) * 1000)
    stdout = completed.stdout[-output_limit:]
    stderr = completed.stderr[-output_limit:]
    if completed.returncode == 0:
        return ExecutionResult(True, "passed", wall_ms, stdout, stderr)
    status = "output_limit" if "output limit exceeded" in stderr else "failed"
    return ExecutionResult(False, status, wall_ms, stdout, stderr)
