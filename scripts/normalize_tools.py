from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from fable_distill.filtering import sanitize_example
from fable_distill.io import iter_jsonl, write_jsonl
from fable_distill.schemas import CanonicalExample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize and sanitize canonical JSONL")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tool-output-chars", type=int, default=12000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    def rows():
        for raw in iter_jsonl(args.input):
            cleaned = sanitize_example(
                CanonicalExample.from_dict(raw),
                max_tool_output_chars=args.max_tool_output_chars,
            )
            if cleaned:
                yield cleaned.to_dict()

    count = write_jsonl(args.output, rows())
    print(f"wrote {count} normalized examples to {args.output}")


if __name__ == "__main__":
    main()

