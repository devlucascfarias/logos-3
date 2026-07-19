from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Any

from .formatting import normalized_prompt_hash
from .io import sha256_json


def _normalized_text(row: dict[str, Any]) -> str:
    messages = row.get("messages", [])
    joined = "\n".join(
        f"{item.get('role', '')}:{item.get('content', '')}" for item in messages
    )
    return re.sub(r"\s+", " ", joined).strip().lower()


def _shingles(text: str, size: int = 5) -> set[str]:
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    if len(tokens) < size:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def deduplicate_rows(
    rows: list[dict[str, Any]],
    approximate_threshold: float = 0.90,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    exact_seen: set[str] = set()
    pair_seen: set[str] = set()
    session_seen: set[tuple[str, str]] = set()
    exact_kept: list[dict[str, Any]] = []
    stats = Counter(raw=len(rows))

    for row in rows:
        exact_hash = sha256_json(
            {
                "messages": row.get("messages", []),
                "source_dataset": row.get("source_dataset"),
                "target_type": row.get("target_type"),
            }
        )
        if exact_hash in exact_seen:
            stats["removed_exact"] += 1
            continue
        exact_seen.add(exact_hash)
        pair_hash = sha256_json({"prompt": row.get("prompt"), "completion": row.get("completion")})
        if row.get("prompt") is None:
            pair_hash = sha256_json(row.get("messages", []))
        if pair_hash in pair_seen:
            stats["removed_prompt_completion"] += 1
            continue
        pair_seen.add(pair_hash)
        source_pair = (
            str(row.get("source_dataset", "unknown")),
            str(row.get("source_example_id", row.get("example_id", ""))),
            str(row.get("target_type", "unknown")),
            sha256_json(row.get("messages", [])),
        )
        if source_pair in session_seen:
            stats["removed_source_overlap"] += 1
            continue
        session_seen.add(source_pair)
        exact_kept.append(row)

    kept: list[dict[str, Any]] = []
    shingle_cache: list[set[str]] = []
    buckets: dict[str, list[int]] = {}
    try:
        from datasketch import MinHash, MinHashLSH

        lsh = MinHashLSH(threshold=approximate_threshold, num_perm=128)
        minhash_available = True
    except ImportError:
        MinHash = None  # type: ignore[assignment,misc]
        lsh = None
        minhash_available = False
    for row in exact_kept:
        text = _normalized_text(row)
        tokens = _shingles(text)
        bucket_key = normalized_prompt_hash(row.get("messages", []))[:8]
        candidates = set(buckets.get(bucket_key, []))
        minhash = None
        if minhash_available and MinHash is not None and tokens:
            minhash = MinHash(num_perm=128)
            for token in sorted(tokens):
                minhash.update(token.encode("utf-8"))
            candidates.update(int(label) for label in lsh.query(minhash))
        duplicate = False
        for candidate in candidates:
            candidate_text = _normalized_text(kept[candidate])
            similarity = max(
                _jaccard(tokens, shingle_cache[candidate]),
                SequenceMatcher(None, text, candidate_text, autojunk=False).ratio(),
            )
            if similarity >= approximate_threshold:
                duplicate = True
                break
        if duplicate:
            stats["removed_approximate"] += 1
            continue
        index = len(kept)
        kept.append(row)
        shingle_cache.append(tokens)
        buckets.setdefault(bucket_key, []).append(index)
        if minhash is not None:
            lsh.insert(str(index), minhash)

    stats["after_exact"] = len(exact_kept)
    stats["after_approximate"] = len(kept)
    source_counts = Counter(str(row.get("source_dataset", "unknown")) for row in kept)
    report = dict(stats)
    report["by_source"] = dict(source_counts)
    report["approximate_method"] = (
        "MinHashLSH+Jaccard+SequenceMatcher"
        if minhash_available
        else "bucketed-Jaccard+SequenceMatcher"
    )
    return kept, report
