from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .tools import normalize_arguments, normalize_tool_name, parse_tool_call, truncate_head_tail


DENIED_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\brm\s+-[^\n]*r",
        r"\bdel\s+/[sq]",
        r"\bformat(?:\.com)?\b",
        r"\bmkfs\b",
        r"\bdd\s+if=",
        r"\bshutdown\b|\breboot\b",
        r"\bcurl\b|\bwget\b|\binvoke-webrequest\b",
        r"\bssh\b|\bscp\b|\bnc\b|\bnetcat\b",
        r"\bgit\s+(?:push|reset\s+--hard|clean\s+-)",
        r"\b(?:sudo|runas)\b",
        r"(?:^|[\s;&|])(?:/|[a-z]:\\)(?:\s|$)",
    )
]


@dataclass(slots=True)
class CommandResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    blocked: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def validate_command(command: str, allow_network: bool = False) -> None:
    if not command.strip():
        raise ValueError("Empty shell command")
    patterns = DENIED_PATTERNS
    if allow_network:
        patterns = patterns[:6] + patterns[9:]
    for pattern in patterns:
        if pattern.search(command):
            raise ValueError(f"Command blocked by safety policy: {pattern.pattern}")


class IsolatedHarness:
    def __init__(
        self,
        source_dir: str | Path,
        *,
        timeout_seconds: int = 30,
        max_output_bytes: int = 12000,
        allow_network: bool = False,
    ):
        self.source_dir = Path(source_dir).resolve()
        self.timeout_seconds = timeout_seconds
        self.max_output_bytes = max_output_bytes
        self.allow_network = allow_network
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self.workdir: Path | None = None

    def __enter__(self) -> "IsolatedHarness":
        self._temporary = tempfile.TemporaryDirectory(prefix="fable-agent-")
        self.workdir = Path(self._temporary.name) / "repo"
        shutil.copytree(self.source_dir, self.workdir, symlinks=False)
        return self

    def __exit__(self, *args: object) -> None:
        if self._temporary:
            self._temporary.cleanup()

    def run(self, command: str) -> CommandResult:
        if self.workdir is None:
            raise RuntimeError("Use IsolatedHarness as a context manager")
        try:
            validate_command(command, allow_network=self.allow_network)
        except ValueError as exc:
            return CommandResult(command, -1, "", str(exc), blocked=True)
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONPATH": str(self.workdir),
                "HOME": str(self.workdir),
                "NO_COLOR": "1",
            }
        )
        try:
            completed = subprocess.run(
                command,
                cwd=self.workdir,
                env=environment,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            return CommandResult(
                command=command,
                exit_code=completed.returncode,
                stdout=truncate_head_tail(completed.stdout, self.max_output_bytes),
                stderr=truncate_head_tail(completed.stderr, self.max_output_bytes),
            )
        except subprocess.TimeoutExpired as exc:
            return CommandResult(
                command=command,
                exit_code=-1,
                stdout=truncate_head_tail(exc.stdout or "", self.max_output_bytes),
                stderr=truncate_head_tail(exc.stderr or "", self.max_output_bytes),
                timed_out=True,
            )

    def _safe_path(self, raw_path: str) -> Path:
        if self.workdir is None:
            raise RuntimeError("Use IsolatedHarness as a context manager")
        candidate = (self.workdir / raw_path).resolve()
        try:
            candidate.relative_to(self.workdir.resolve())
        except ValueError as exc:
            raise ValueError(f"Path escapes isolated workspace: {raw_path}") from exc
        return candidate

    def execute_tool(self, name: str, arguments: dict[str, object]) -> dict[str, object]:
        """Execute one canonical tool inside the copied workspace."""

        normalized_name = normalize_tool_name(name)
        args = normalize_arguments(normalized_name, arguments)
        if normalized_name == "shell":
            return self.run(str(args.get("command", ""))).to_dict()
        if normalized_name == "list_files":
            root = self._safe_path(str(args.get("path", ".")))
            if not root.exists():
                return {"error": f"path not found: {args.get('path', '.')}"}
            files = [
                str(path.relative_to(self.workdir)).replace("\\", "/")
                for path in root.rglob("*")
                if path.is_file()
            ][:1000]
            return {"files": files, "truncated": len(files) == 1000}
        if normalized_name == "read_file":
            path = self._safe_path(str(args.get("path", "")))
            if not path.is_file():
                return {"error": f"file not found: {args.get('path', '')}"}
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return {"error": "binary or non-UTF-8 file rejected"}
            return {"content": truncate_head_tail(content, self.max_output_bytes)}
        if normalized_name == "write_file":
            path = self._safe_path(str(args.get("path", "")))
            path.parent.mkdir(parents=True, exist_ok=True)
            content = str(args.get("content", ""))
            path.write_text(content, encoding="utf-8", newline="\n")
            return {"written": str(path.relative_to(self.workdir)).replace("\\", "/")}
        if normalized_name == "search":
            root = self._safe_path(str(args.get("path", ".")))
            pattern = re.compile(str(args.get("pattern", "")))
            matches: list[dict[str, object]] = []
            candidates = [root] if root.is_file() else root.rglob("*")
            for path in candidates:
                if not path.is_file():
                    continue
                try:
                    lines = path.read_text(encoding="utf-8").splitlines()
                except (UnicodeDecodeError, OSError):
                    continue
                for line_number, line in enumerate(lines, 1):
                    if pattern.search(line):
                        matches.append(
                            {
                                "path": str(path.relative_to(self.workdir)).replace("\\", "/"),
                                "line": line_number,
                                "text": line[:500],
                            }
                        )
                        if len(matches) == 200:
                            return {"matches": matches, "truncated": True}
            return {"matches": matches, "truncated": False}
        if normalized_name == "apply_patch":
            if "patch" not in args:
                path = self._safe_path(str(args.get("path", "")))
                if not path.is_file():
                    return {"error": f"file not found: {args.get('path', '')}"}
                content = path.read_text(encoding="utf-8")
                old_text = str(args.get("old_text", ""))
                if not old_text or old_text not in content:
                    return {"error": "old_text was not found; no changes applied"}
                if content.count(old_text) != 1:
                    return {"error": "old_text is ambiguous; no changes applied"}
                path.write_text(
                    content.replace(old_text, str(args.get("new_text", "")), 1),
                    encoding="utf-8",
                    newline="\n",
                )
                return {
                    "written": str(path.relative_to(self.workdir)).replace("\\", "/"),
                    "replacements": 1,
                }
            patch = str(args.get("patch", ""))
            if re.search(r"(?m)^(?:---|\+\+\+)\s+(?:/|[A-Za-z]:|\.\./)", patch):
                return {"error": "patch path escapes isolated workspace", "blocked": True}
            if self.workdir is None:
                raise RuntimeError("Use IsolatedHarness as a context manager")
            patch_file = self.workdir / ".fable-agent.patch"
            patch_file.write_text(patch, encoding="utf-8", newline="\n")
            try:
                checked = subprocess.run(
                    ["git", "apply", "--check", str(patch_file)],
                    cwd=self.workdir,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )
                if checked.returncode != 0:
                    return {
                        "exit_code": checked.returncode,
                        "stderr": truncate_head_tail(checked.stderr, self.max_output_bytes),
                    }
                applied = subprocess.run(
                    ["git", "apply", "--whitespace=nowarn", str(patch_file)],
                    cwd=self.workdir,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )
                return {
                    "exit_code": applied.returncode,
                    "stdout": truncate_head_tail(applied.stdout, self.max_output_bytes),
                    "stderr": truncate_head_tail(applied.stderr, self.max_output_bytes),
                }
            finally:
                patch_file.unlink(missing_ok=True)
        raise ValueError(f"Unsupported tool: {normalized_name}")

    def execute_tagged_tool_call(self, content: str) -> dict[str, object]:
        payload = parse_tool_call(content)
        return self.execute_tool(payload["name"], payload["arguments"])

    def run_many(self, commands: Iterable[str], max_steps: int = 20) -> list[CommandResult]:
        results: list[CommandResult] = []
        for command in list(commands)[:max_steps]:
            result = self.run(command)
            results.append(result)
            if result.blocked or result.timed_out:
                break
        return results
