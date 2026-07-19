from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from fable_distill.harness import IsolatedHarness
from fable_distill.io import iter_jsonl, write_jsonl
from fable_distill.tools import normalize_arguments, normalize_tool_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pre-generated agent commands in isolation")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output", default="outputs/evaluations/agent_runs.jsonl")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--max-output-bytes", type=int, default=12000)
    parser.add_argument("--allow-network", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    def runs():
        for index, task in enumerate(iter_jsonl(args.tasks)):
            repository = task.get("repository_path")
            commands = task.get("commands", [])
            if not repository or not Path(repository).is_dir():
                yield {
                    "task_id": task.get("task_id", index),
                    "error": f"repository_path not found: {repository}",
                }
                continue
            with IsolatedHarness(
                repository,
                timeout_seconds=args.timeout,
                max_output_bytes=args.max_output_bytes,
                allow_network=args.allow_network,
            ) as harness:
                if task.get("tool_calls"):
                    tool_results = []
                    for call in list(task["tool_calls"])[: args.max_steps]:
                        if isinstance(call, str):
                            tool_results.append(harness.execute_tagged_tool_call(call))
                        elif isinstance(call, dict):
                            name = normalize_tool_name(str(call.get("name", "")))
                            arguments = normalize_arguments(name, call.get("arguments", {}))
                            tool_results.append(harness.execute_tool(name, arguments))
                    results = []
                else:
                    results = harness.run_many(commands, max_steps=args.max_steps)
                    tool_results = []
            yield {
                "task_id": task.get("task_id", index),
                "seed": task.get("seed", 42),
                "results": [result.to_dict() for result in results],
                "tool_results": tool_results,
                "tests_passed": bool(
                    (results and results[-1].exit_code == 0)
                    or (
                        tool_results
                        and isinstance(tool_results[-1], dict)
                        and tool_results[-1].get("exit_code") == 0
                    )
                ),
                "invalid_calls": sum(result.blocked for result in results)
                + sum(bool(result.get("blocked")) for result in tool_results),
            }

    count = write_jsonl(args.output, runs())
    summary = {"tasks": count, "output": args.output}
    Path(args.output).with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
