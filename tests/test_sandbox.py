import os

import pytest

from qwen_sft.sandbox import (
    SandboxViolation,
    extract_single_python_block,
    run_verified,
    validate_source,
)


def test_requires_exactly_one_python_block():
    assert extract_single_python_block("Breve.\n```python\nx = 1\n```") == "x = 1"
    with pytest.raises(SandboxViolation, match="exatamente um"):
        extract_single_python_block("sem bloco")
    with pytest.raises(SandboxViolation, match="exatamente um"):
        extract_single_python_block("```python\nx=1\n```\n```python\ny=2\n```")
    with pytest.raises(SandboxViolation, match="exatamente um"):
        extract_single_python_block("```python\nx=1\n```\n```text\ny\n```")


@pytest.mark.parametrize(
    "source",
    [
        "import os",
        "import socket\nsocket.create_connection(('example.com', 80))",
        "import subprocess\nsubprocess.run(['echo', 'unsafe'])",
        "from pathlib import Path",
        "class Unsafe: pass",
        "open('secret')",
        "eval('1 + 1')",
        "x.__class__",
        "getattr(x, 'value')",
    ],
)
def test_ast_policy_rejects_unsafe_capabilities(source):
    with pytest.raises(SandboxViolation):
        validate_source(source)


def test_ast_policy_rejects_syntax_error():
    with pytest.raises(SandboxViolation, match="Python inválido"):
        validate_source("def broken(")


def test_ast_policy_allows_contract_code_with_closure():
    validate_source(
        "def memo(fn):\n"
        "    cached = None\n"
        "    def wrapped(value):\n"
        "        nonlocal cached\n"
        "        cached = fn(value)\n"
        "        return cached\n"
        "    return wrapped\n"
    )


@pytest.mark.skipif(os.name != "posix", reason="resource limits target Colab/Linux")
def test_executes_in_fresh_limited_process():
    result = run_verified("def add(a, b):\n    return a + b", "assert add(2, 3) == 5")
    assert result.passed
    evidence = result.evidence("def add(a, b):\n    return a + b", "assert add(2, 3) == 5")
    assert evidence["status"] == "passed"
    assert evidence["code_sha256"]
    assert evidence["tests_sha256"]
    assert evidence["sha256"]


@pytest.mark.skipif(os.name != "posix", reason="resource limits target Colab/Linux")
def test_times_out_and_caps_output():
    timed = run_verified("while True:\n    pass", "assert True", timeout_seconds=0.2)
    assert not timed.passed
    assert timed.status in {"timeout", "failed"}

    noisy = run_verified("print('x' * 1000)", "assert True", output_limit=100)
    assert not noisy.passed
    assert noisy.status == "output_limit"
