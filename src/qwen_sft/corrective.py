from __future__ import annotations

import ast
import hashlib
import json
import random
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

from qwen_sft.io import file_sha256, read_jsonl
from qwen_sft.sandbox import RUNNER_VERSION, extract_single_python_block, run_verified


OPENCODE_NAME = "nvidia/OpenCodeInstruct"
OPENCODE_REVISION = "8f3ba5bafe4d6e8db46082cf7ae6741bc370604d"
SYSTEM_PROMPT = (
    "Voc\u00ea \u00e9 um assistente especializado em programa\u00e7\u00e3o. Produza uma "
    "solu\u00e7\u00e3o direta, correta e verific\u00e1vel."
)
VERIFICATION_LEVELS = {
    "local_hidden_tests",
    "local_public_tests",
    "replay_champion",
}
ALLOWED_LICENSES = {
    "apache-2.0",
    "bsd-2-clause",
    "bsd-3-clause",
    "cc-by-4.0",
    "isc",
    "mit",
    "project-generated",
}
REPLAY_SOURCE_LICENSES = {
    "nvidia/OpenCodeReasoning": "cc-by-4.0",
    "nvidia/Nemotron-Competitive-Programming-v1": "cc-by-4.0",
    "open-thoughts/OpenThoughts-114k": "apache-2.0",
    "nvidia/Nemotron-SFT-SWE-v2": "cc-by-4.0",
    "nvidia/Nemotron-Post-Training-Dataset-v1": "cc-by-4.0",
}
THINK_RE = re.compile(
    r"<think>.*?(?:</think>|$)", re.IGNORECASE | re.DOTALL
)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def evidence_sha256(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "sha256"}
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def manifest_payload_sha256(value: dict[str, Any]) -> str:
    payload = {
        key: item for key, item in value.items() if key != "manifest_payload_sha256"
    }
    return sha256_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def normalized_prompt(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = re.sub(r"```.*?```", " ", value, flags=re.DOTALL)
    value = re.sub(r"\b[a-zA-Z_]\w*\b(?=\s*\()", " function ", value)
    return " ".join(re.findall(r"\w+", value.casefold(), flags=re.UNICODE))


def prompt_ngrams(value: str, size: int = 5) -> set[tuple[str, ...]]:
    tokens = normalized_prompt(value).split()
    if len(tokens) < size:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}


def jaccard_prompt(left: str, right: str, size: int = 5) -> float:
    a = prompt_ngrams(left, size)
    b = prompt_ngrams(right, size)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if a | b else 0.0


def assert_not_contaminated(
    candidates: Sequence[dict[str, Any]],
    evaluation_tasks: Sequence[dict[str, Any]],
    *,
    threshold: float = 0.70,
) -> None:
    eval_ids = {str(item["id"]) for item in evaluation_tasks}
    eval_templates = {str(item["template_id"]) for item in evaluation_tasks}
    eval_hashes = {sha256_text(normalized_prompt(str(item["prompt"]))) for item in evaluation_tasks}
    eval_prompts = [(str(item["id"]), str(item["prompt"])) for item in evaluation_tasks]
    for candidate in candidates:
        if str(candidate["id"]) in eval_ids:
            raise ValueError(f"ID contaminado: {candidate['id']}")
        if str(candidate.get("template_id", "")) in eval_templates:
            raise ValueError(f"template_id contaminado: {candidate.get('template_id')}")
        prompt = next(
            message["content"]
            for message in candidate["messages"]
            if message["role"] == "user"
        )
        if sha256_text(normalized_prompt(prompt)) in eval_hashes:
            raise ValueError(f"Prompt normalizado contaminado: {candidate['id']}")
        for eval_id, eval_prompt in eval_prompts:
            score = jaccard_prompt(prompt, eval_prompt)
            if score >= threshold:
                raise ValueError(
                    f"Prompt {candidate['id']} contamina {eval_id}: Jaccard={score:.3f}"
                )


@dataclass(frozen=True)
class FamilySpec:
    name: str
    train_names: tuple[str, str, str]
    eval_names: tuple[str, str, str, str]
    solution: str
    public_tests: tuple[str, ...]
    hidden_tests: tuple[str, ...]
    pt_prompts: tuple[str, str, str]
    en_prompts: tuple[str, str, str]
    contract_checks: tuple[str, ...]


def _specs() -> tuple[FamilySpec, ...]:
    return (
        FamilySpec(
            "ordered_unique",
            ("retain_unique", "ordered_distinct", "deduplicate_seen"),
            ("unique_stream", "distinct_first", "compact_unique", "seen_once"),
            """def {fn}(items):
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result""",
            ("assert {fn}([]) == []", "assert {fn}([3, 1, 3, 2, 1]) == [3, 1, 2]"),
            ("assert {fn}('cabca') == ['c', 'a', 'b']", "assert {fn}([(1,), (1,), (2,)]) == [(1,), (2,)]"),
            (
                "Implemente {fn}(items): retenha somente a primeira apari\u00e7\u00e3o de cada valor hashable, na ordem de entrada e em O(n) m\u00e9dio.",
                "Crie {fn}(items) sem imports. A sa\u00edda deve eliminar repeti\u00e7\u00f5es globais sem reordenar os elementos.",
                "Escreva {fn}(items) para compactar valores hashable repetidos, preservando a posi\u00e7\u00e3o da primeira ocorr\u00eancia.",
            ),
            (
                "Implement {fn}(items) in average O(n), keeping the first occurrence of each hashable value in input order.",
                "Write {fn}(items) without imports. Remove duplicate hashable elements globally while retaining their original order.",
                "Create {fn}(items) so repeated hashable values are compacted to their earliest occurrence.",
            ),
            ("first_occurrence_order", "average_linear", "hashable_only"),
        ),
        FamilySpec(
            "strict_parser",
            ("decode_records", "parse_entries", "read_pairs"),
            ("scan_records", "decode_fields", "parse_segments", "read_assignments"),
            """def {fn}(lines):
    result = {}
    for number, line in enumerate(lines, 1):
        if line == '':
            continue
        if line.count(':') != 1:
            raise ValueError(f'invalid line {number}')
        key, value = line.split(':')
        if not key:
            raise ValueError(f'invalid line {number}')
        result.setdefault(key, []).append(value)
    return result""",
            ("assert {fn}(['a:1', '', 'a:2']) == {'a': ['1', '2']}", "assert {fn}(['x:']) == {'x': ['']}"),
            ("\ntry:\n    {fn}(['ok:1', 'bad'])\nexcept ValueError as error:\n    assert '2' in str(error)\nelse:\n    assert False", "\ntry:\n    {fn}(['a:b:c'])\nexcept ValueError:\n    pass\nelse:\n    assert False"),
            (
                "Implemente {fn}(lines). Cada registro n\u00e3o vazio cont\u00e9m exatamente um ':' e uma chave n\u00e3o vazia; agrupe valores por chave. Erros devem informar a linha baseada em 1.",
                "Crie o parser {fn}(lines): ignore apenas strings vazias, valide um \u00fanico separador ':' e acumule valores repetidos em ordem.",
                "Escreva {fn}(lines) para interpretar chave:valor estritamente. Lance ValueError com o n\u00famero da linha para formato inv\u00e1lido.",
            ),
            (
                "Implement {fn}(lines): non-empty records contain exactly one ':' and a non-empty key; group repeated values in order and number errors from one.",
                "Create strict parser {fn}(lines). Ignore only empty strings, require one ':' separator, and preserve repeated value order.",
                "Write {fn}(lines) for strict key:value records, raising ValueError with the one-based line number when malformed.",
            ),
            ("exact_separator", "one_based_error", "empty_value_allowed"),
        ),
        FamilySpec(
            "retry_identity",
            ("attempt_call", "repeat_operation", "try_again"),
            ("invoke_retry", "retry_action", "call_until", "repeat_until"),
            """def {fn}(operation, attempts, retry_on):
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts <= 0:
        raise ValueError('attempts must be a positive integer')
    last_error = None
    for _ in range(attempts):
        try:
            return operation()
        except retry_on as error:
            last_error = error
    raise last_error""",
            ("\nstate = {'n': 0}\ndef flaky():\n    state['n'] += 1\n    if state['n'] < 3:\n        raise LookupError('later')\n    return 7\nassert {fn}(flaky, 3, LookupError) == 7",),
            ("\nerror = ValueError('same')\ndef fail():\n    raise error\ntry:\n    {fn}(fail, 2, ValueError)\nexcept ValueError as caught:\n    assert caught is error\nelse:\n    assert False", "\ntry:\n    {fn}(lambda: 1, True, Exception)\nexcept ValueError:\n    pass\nelse:\n    assert False"),
            (
                "Implemente {fn}(operation, attempts, retry_on). Valide inteiro positivo sem aceitar bool, retorne no primeiro sucesso e propague a pr\u00f3pria \u00faltima exce\u00e7\u00e3o compat\u00edvel.",
                "Crie {fn} para repetir uma fun\u00e7\u00e3o sem argumentos no m\u00e1ximo attempts vezes. Capture somente retry_on e preserve a identidade da falha final.",
                "Escreva {fn} sem espera ou logging; attempts inv\u00e1lido gera ValueError e exce\u00e7\u00f5es fora de retry_on escapam imediatamente.",
            ),
            (
                "Implement {fn}(operation, attempts, retry_on). Reject bool, return on first success, and re-raise the identical final matching exception.",
                "Create {fn} to call a zero-argument operation at most attempts times, catching only retry_on and preserving final exception identity.",
                "Write {fn} without waiting or logging; invalid attempts raises ValueError and non-matching exceptions escape immediately.",
            ),
            ("positive_non_bool", "matching_only", "same_final_exception"),
        ),
        FamilySpec(
            "binary_boundary",
            ("lower_position", "first_match", "leftmost_index"),
            ("boundary_index", "earliest_match", "locate_left", "first_sorted"),
            """def {fn}(values, target):
    low = 0
    high = len(values)
    while low < high:
        middle = (low + high) // 2
        if values[middle] < target:
            low = middle + 1
        else:
            high = middle
    if low < len(values) and values[low] == target:
        return low
    return -1""",
            ("assert {fn}([], 4) == -1", "assert {fn}([1, 2, 2, 2, 5], 2) == 1"),
            ("assert {fn}([1, 3, 5], 2) == -1", "assert {fn}([0, 0, 0], 0) == 0"),
            (
                "Implemente {fn}(values, target) para achar o primeiro \u00edndice em uma sequ\u00eancia ordenada com duplicatas, ou -1, em O(log n).",
                "Crie {fn} por busca bin\u00e1ria, sem slicing nem modifica\u00e7\u00e3o da sequ\u00eancia, retornando a ocorr\u00eancia mais \u00e0 esquerda.",
                "Escreva {fn}(values, target): localize o limite esquerdo de target em tempo logar\u00edtmico e indique aus\u00eancia com -1.",
            ),
            (
                "Implement {fn}(values, target) to find the first index in a sorted sequence with duplicates, or -1, in O(log n).",
                "Create binary-search function {fn} without slicing or mutation, returning the leftmost matching position.",
                "Write {fn}(values, target): locate target's left boundary in logarithmic time and use -1 when absent.",
            ),
            ("logarithmic", "leftmost", "absent_minus_one"),
        ),
        FamilySpec(
            "single_pass_chunks",
            ("batch_stream", "take_chunks", "partition_iter"),
            ("stream_batches", "chunk_once", "yield_groups", "consume_batches"),
            """def {fn}(iterable, size):
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError('size must be a positive integer')
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch""",
            ("assert list({fn}((x for x in range(5)), 2)) == [[0, 1], [2, 3], [4]]",),
            ("assert list({fn}(iter(()), 3)) == []", "\ntry:\n    list({fn}([1], True))\nexcept ValueError:\n    pass\nelse:\n    assert False"),
            (
                "Implemente o generator {fn}(iterable, size), consumindo a entrada uma vez e emitindo listas, inclusive o lote final incompleto. Rejeite bool e tamanhos n\u00e3o positivos.",
                "Crie {fn} para agrupar um iterador de uso \u00fanico sem materializ\u00e1-lo. Cada lista tem no m\u00e1ximo size itens.",
                "Escreva {fn}(iterable, size) como generator de passagem \u00fanica; valide size como int positivo que n\u00e3o seja bool.",
            ),
            (
                "Implement generator {fn}(iterable, size), consuming once and yielding lists including an incomplete final batch. Reject bool and non-positive sizes.",
                "Create {fn} to group a single-use iterator without materializing it; each list has at most size elements.",
                "Write {fn}(iterable, size) as a one-pass generator and validate size as a positive non-bool integer.",
            ),
            ("single_pass", "incomplete_final", "positive_non_bool"),
        ),
        FamilySpec(
            "recursive_copy",
            ("combine_maps", "overlay_nested", "merge_tree"),
            ("join_nested", "merge_without_alias", "overlay_maps", "combine_nested"),
            """def {fn}(left, right):
    def clone(value):
        if isinstance(value, dict):
            return {key: clone(item) for key, item in value.items()}
        return value
    result = {key: clone(value) for key, value in left.items()}
    for key, value in right.items():
        if key in left and isinstance(left[key], dict) and isinstance(value, dict):
            result[key] = {fn}(left[key], value)
        else:
            result[key] = clone(value)
    return result""",
            ("assert {fn}({'a': {'x': 1}}, {'a': {'y': 2}}) == {'a': {'x': 1, 'y': 2}}",),
            ("\nleft = {'a': {'x': 1}}\nmerged = {fn}(left, {})\nmerged['a']['x'] = 9\nassert left == {'a': {'x': 1}}", "\nright = {'z': {'n': 1}}\nmerged = {fn}({}, right)\nmerged['z']['n'] = 2\nassert right['z']['n'] == 1"),
            (
                "Implemente {fn}(left, right): combine dicion\u00e1rios recursivamente, com preced\u00eancia de right, sem modificar nem compartilhar subdicion\u00e1rios com as entradas.",
                "Crie {fn} para sobrepor mapas aninhados. S\u00f3 fa\u00e7a recurs\u00e3o quando ambos os valores forem dict e clone todos os subdicts.",
                "Escreva {fn}(left, right) produzindo um novo grafo de dicion\u00e1rios; conflitos n\u00e3o-dict usam right e nenhum dict de entrada pode ser alias do resultado.",
            ),
            (
                "Implement {fn}(left, right): recursively merge dictionaries with right precedence, without mutating or sharing nested dictionaries with inputs.",
                "Create {fn} to overlay nested mappings, recursing only when both values are dicts and cloning every sub-dict.",
                "Write {fn}(left, right) as a fresh dictionary graph; non-dict conflicts use right and no input dict may alias the result.",
            ),
            ("right_precedence", "recursive_dicts", "no_dict_alias"),
        ),
        FamilySpec(
            "stable_selection",
            ("rank_best", "select_highest", "stable_best"),
            ("choose_top", "rank_stably", "highest_items", "take_best"),
            """def {fn}(items, k, key):
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError('k must be a non-negative integer')
    scored = [(key(item), item) for item in items]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored[:k]]""",
            ("assert {fn}(['a', 'bb', 'c'], 2, len) == ['bb', 'a']", "assert {fn}([1, 2], 0, lambda x: x) == []"),
            ("\ncalls = []\nassert {fn}([3, 1, 2], 9, lambda x: calls.append(x) or x) == [3, 2, 1]\nassert calls == [3, 1, 2]", "\ntry:\n    {fn}([], False, lambda x: x)\nexcept ValueError:\n    pass\nelse:\n    assert False"),
            (
                "Implemente {fn}(items, k, key): retorne at\u00e9 k itens por pontua\u00e7\u00e3o decrescente, preservando empates. Rejeite bool e chame key uma vez por item.",
                "Crie {fn} como top-k est\u00e1vel sem modificar items. k \u00e9 int n\u00e3o negativo; se exceder o tamanho, ordene e retorne todos.",
                "Escreva {fn}(items, k, key), calculando cada score exatamente uma vez e mantendo a ordem original em scores iguais.",
            ),
            (
                "Implement {fn}(items, k, key): return up to k items by descending score with stable ties. Reject bool and call key once per item.",
                "Create stable top-k {fn} without mutating items. k is a non-negative int; when too large, sort and return every item.",
                "Write {fn}(items, k, key), computing each score exactly once and retaining original order for equal scores.",
            ),
            ("descending", "stable_ties", "key_once", "non_bool_k"),
        ),
        FamilySpec(
            "dependency_cycles",
            ("task_sequence", "order_jobs", "resolve_prereqs"),
            ("schedule_tasks", "dependency_list", "topological_tasks", "plan_jobs"),
            """def {fn}(graph):
    state = {}
    result = []
    def visit(node):
        if state.get(node) == 1:
            raise ValueError(f'cycle at {node}')
        if state.get(node) == 2:
            return
        state[node] = 1
        for dependency in graph.get(node, []):
            visit(dependency)
        state[node] = 2
        result.append(node)
    for node in graph:
        visit(node)
    return result""",
            ("assert {fn}({'build': ['compile'], 'compile': []}) == ['compile', 'build']", "assert {fn}({'ship': ['pack']}) == ['pack', 'ship']"),
            ("\ntry:\n    {fn}({'a': ['b'], 'b': ['a']})\nexcept ValueError as error:\n    assert str(error).endswith(('a', 'b'))\nelse:\n    assert False", "assert {fn}({'z': [], 'a': []}) == ['z', 'a']"),
            (
                "Implemente {fn}(graph): ordene tarefas depois das depend\u00eancias, inclua n\u00f3s externos, preserve a ordem recebida e detecte ciclos com ValueError.",
                "Crie {fn} por DFS determin\u00edstica sem modificar graph. Uma aresta em ciclo deve informar uma tarefa participante.",
                "Escreva {fn}(graph) para produzir ordem topol\u00f3gica seguindo a inser\u00e7\u00e3o de chaves e depend\u00eancias; rejeite ciclos.",
            ),
            (
                "Implement {fn}(graph): place tasks after dependencies, include external nodes, preserve received order, and reject cycles with ValueError.",
                "Create deterministic DFS {fn} without mutating graph; a back edge must report a participating task.",
                "Write {fn}(graph) to produce topological order following key and dependency insertion order, rejecting cycles.",
            ),
            ("dependency_first", "received_order", "external_nodes", "cycle_error"),
        ),
        FamilySpec(
            "memo_last",
            ("cache_recent", "remember_call", "memo_previous"),
            ("memo_single", "cache_latest", "remember_latest", "last_result_cache"),
            """def {fn}(function):
    missing = object()
    last_args = missing
    last_kwargs = missing
    last_result = None
    def wrapper(*args, **kwargs):
        nonlocal last_args, last_kwargs, last_result
        if last_args is not missing and args == last_args and kwargs == last_kwargs:
            return last_result
        result = function(*args, **kwargs)
        last_args = args
        last_kwargs = dict(kwargs)
        last_result = result
        return result
    return wrapper""",
            ("\ncalls = []\nwrapped = {fn}(lambda value: calls.append(value) or len(value))\nassert wrapped([1, 2]) == 2\nassert wrapped([1, 2]) == 2\nassert calls == [[1, 2]]",),
            ("\ncalls = []\ndef work(value=0):\n    calls.append(value)\n    if value < 0:\n        raise ValueError('bad')\n    return value\nwrapped = {fn}(work)\nfor _ in range(2):\n    try:\n        wrapped(value=-1)\n    except ValueError:\n        pass\nassert calls == [-1, -1]", "\nwrapped = {fn}(lambda **kw: list(kw.values()))\nassert wrapped(value=[1]) == [[1]]\nassert wrapped(value=[1]) == [[1]]"),
            (
                "Implemente {fn}(function), cacheando somente a chamada bem-sucedida mais recente. Compare args e kwargs por igualdade, inclusive valores n\u00e3o hashable, e n\u00e3o armazene falhas.",
                "Crie o decorador {fn}: aceite argumentos posicionais e nomeados, reutilize apenas o \u00faltimo resultado equivalente e preserve exce\u00e7\u00f5es.",
                "Escreva {fn}(function) com cache de uma entrada sem hashing. Uma chamada que levanta exce\u00e7\u00e3o deve ser executada novamente depois.",
            ),
            (
                "Implement {fn}(function), caching only the most recent successful call. Compare args and kwargs by equality, including unhashable values, and never cache failures.",
                "Create decorator {fn}: accept positional and named arguments, reuse only the equivalent latest result, and preserve exceptions.",
                "Write {fn}(function) with a one-entry cache and no hashing. A failing call must execute again next time.",
            ),
            ("single_entry", "equality_nonhashable", "failures_not_cached"),
        ),
        FamilySpec(
            "rolling_window",
            ("window_means", "moving_mean", "average_windows"),
            ("rolling_means", "sliding_average", "mean_windows", "window_average"),
            """def {fn}(values, window):
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise ValueError('window must be a positive integer')
    buffer = []
    cursor = 0
    total = 0
    result = []
    for value in values:
        if len(buffer) < window:
            buffer.append(value)
            total += value
        else:
            total -= buffer[cursor]
            buffer[cursor] = value
            total += value
            cursor = (cursor + 1) % window
        if len(buffer) == window:
            result.append(total / window)
    return result""",
            ("assert {fn}((x for x in [1, 3, 5]), 2) == [2.0, 4.0]", "assert {fn}([7, 8], 1) == [7.0, 8.0]"),
            ("assert {fn}([1, 2], 3) == []", "\ntry:\n    {fn}([], True)\nexcept ValueError:\n    pass\nelse:\n    assert False"),
            (
                "Implemente {fn}(values, window), calculando m\u00e9dias de janelas completas em O(n), numa passagem sobre qualquer iterable. Rejeite bool e inteiros n\u00e3o positivos.",
                "Crie {fn} para m\u00e9dia m\u00f3vel incremental, inclusive sobre generator; retorne [] quando faltar uma janela completa.",
                "Escreva {fn}(values, window) sem recalcular cada soma. window deve ser int positivo e n\u00e3o bool.",
            ),
            (
                "Implement {fn}(values, window), computing full-window averages in O(n) with one pass over any iterable. Reject bool and non-positive integers.",
                "Create incremental moving average {fn}, including generator inputs; return [] when no full window exists.",
                "Write {fn}(values, window) without recomputing each sum. window must be a positive non-bool int.",
            ),
            ("single_pass", "incremental_sum", "averages", "positive_non_bool"),
        ),
        FamilySpec(
            "transactional_update",
            ("commit_changes", "validated_update", "apply_transaction"),
            ("update_atomically", "commit_validated", "stage_updates", "validate_changes"),
            """def {fn}(state, updates, validator):
    candidate = dict(state)
    for key, value in updates:
        candidate[key] = value
        if not validator(candidate):
            raise ValueError(f'invalid update for {key}')
    return candidate""",
            ("\nstate = {'x': 1}\nassert {fn}(state, [('x', 2)], lambda item: item['x'] > 0) == {'x': 2}\nassert state == {'x': 1}",),
            ("\nstate = {'x': 1}\ntry:\n    {fn}(state, [('x', 0)], lambda item: item['x'] > 0)\nexcept ValueError:\n    pass\nelse:\n    assert False\nassert state == {'x': 1}", "\nmarker = RuntimeError('validator')\ndef reject(_):\n    raise marker\ntry:\n    {fn}({}, [('x', 1)], reject)\nexcept RuntimeError as caught:\n    assert caught is marker\nelse:\n    assert False"),
            (
                "Implemente {fn}(state, updates, validator): aplique pares a uma c\u00f3pia rasa e valide ap\u00f3s cada passo. False gera ValueError; exce\u00e7\u00f5es s\u00e3o propagadas; entradas ficam intactas.",
                "Crie {fn} como atualiza\u00e7\u00e3o transacional em mem\u00f3ria. Retorne o novo dict apenas se todas as valida\u00e7\u00f5es forem verdadeiras.",
                "Escreva {fn}(state, updates, validator) preservando atomicidade: trabalhe numa c\u00f3pia, respeite a ordem e converta retorno falso em ValueError.",
            ),
            (
                "Implement {fn}(state, updates, validator): apply pairs to a shallow copy and validate each step. False raises ValueError, exceptions propagate, and inputs remain intact.",
                "Create {fn} as an in-memory transaction, returning a new dict only when every validation is true.",
                "Write {fn}(state, updates, validator) atomically: work on a copy, preserve update order, and convert false validation to ValueError.",
            ),
            ("shallow_copy", "validate_each", "false_value_error", "exception_identity"),
        ),
    )


FAMILY_SPECS = _specs()


# Authored independently from FamilySpec.pt_prompts/en_prompts: training and
# evaluation share a contract family, never a prompt template or function name.
EVALUATION_PROMPTS: dict[str, tuple[str, str, str, str]] = {
    "ordered_unique": (
        "Construa {fn}(items) para devolver cada item hashable uma única vez; a posição observada primeiro decide a ordem e o custo médio deve ser linear.",
        "Em {fn}(items), descarte ocorrências posteriores mesmo quando não forem adjacentes, sem reordenar as primeiras ocorrências.",
        "Faça {fn}(items) aceitar um iterable de valores hashable e gerar uma lista estável de distintos em O(n) esperado.",
        "Define {fn}(items) so globally repeated hashable entries disappear while the earliest encounter order is retained in expected linear time.",
    ),
    "strict_parser": (
        "Processe em {fn}(lines) registros chave:valor: pule a string vazia, exija um único dois-pontos e chave presente, preserve valores repetidos e numere erros a partir de 1.",
        "{fn}(lines) deve formar listas por chave para linhas com exatamente um ':'. Qualquer registro não vazio fora desse contrato levanta ValueError com sua posição humana.",
        "Implemente {fn}(lines) como leitor rigoroso de campos separados por ':', permitindo valor vazio, mas nunca chave vazia ou separador duplicado.",
        "Build {fn}(lines) for colon-delimited records: skip only empty strings, keep repeated values in encounter order, and include the one-based record number in malformed-input errors.",
    ),
    "retry_identity": (
        "Defina {fn}(operation, attempts, retry_on): attempts é int positivo não-bool; tente até o sucesso e, ao esgotar, relance o mesmo objeto de exceção capturado por último.",
        "{fn} recebe uma callable sem argumentos. Somente retry_on autoriza nova tentativa; falhas de outro tipo devem sair sem interceptação.",
        "Implemente {fn} sem espera nem logs, encerrando no primeiro retorno e preservando a identidade da última falha compatível.",
        "Write {fn}(operation, attempts, retry_on) with a positive non-bool retry count, immediate success return, matching-only catches, and identity-preserving final re-raise.",
    ),
    "binary_boundary": (
        "Em {fn}(values, target), encontre por busca binária a posição mais à esquerda de target numa sequência ordenada, devolvendo -1 se faltar.",
        "Crie {fn} com tempo logarítmico para localizar o início de uma faixa de valores iguais, sem slicing ou mutação.",
        "{fn}(values, target) deve tratar vazio, ausente e duplicatas, sempre escolhendo o menor índice correspondente.",
        "Implement logarithmic {fn}(values, target) for the earliest matching slot in sorted data, using -1 for a missing target and no slicing.",
    ),
    "single_pass_chunks": (
        "Produza em {fn}(iterable, size) um generator de listas consecutivas; consuma a fonte uma vez e emita também o resto menor que size.",
        "{fn} deve funcionar com iterator descartável, sem carregar toda a entrada, e recusar size bool, não inteiro ou não positivo.",
        "Implemente {fn}(iterable, size) para entregar lotes com no máximo size elementos mantendo somente o lote corrente em memória.",
        "Create one-pass generator {fn}(iterable, size) for bounded lists and a possible short tail; reject booleans and invalid positive sizes.",
    ),
    "recursive_copy": (
        "Combine left e right em {fn}: quando ambos os lados de uma chave forem dict, desça recursivamente; caso contrário, right vence. Nenhum subdict do resultado pode ser alias das entradas.",
        "{fn}(left, right) deve criar uma árvore nova de dicionários, preservar chaves exclusivas e aplicar precedência direita nos demais conflitos.",
        "Implemente merge aninhado {fn} sem alterar argumentos e clone também ramos dict que aparecem em apenas um lado.",
        "Write {fn}(left, right) as a recursive right-biased dictionary merge whose resulting dictionaries share no mutable dictionary nodes with either input.",
    ),
    "stable_selection": (
        "Defina {fn}(items, k, key) para ordenar por score decrescente e selecionar até k, mantendo a ordem original dos empates e avaliando key uma vez por item.",
        "{fn} deve rejeitar k bool ou negativo, não modificar items e devolver todos ordenados quando k ultrapassar a quantidade disponível.",
        "Implemente top-k estável {fn}; cacheie cada pontuação antes de ordenar para que key tenha exatamente uma chamada por elemento.",
        "Implement {fn}(items, k, key) as a stable descending top-k with one key evaluation per item, non-negative non-bool k validation, and no input mutation.",
    ),
    "dependency_cycles": (
        "Crie {fn}(graph) para listar cada tarefa depois de suas dependências, incluindo dependências sem chave e respeitando a ordem de inserção em toda travessia.",
        "{fn} deve produzir uma ordenação topológica determinística sem tocar em graph; um ciclo levanta ValueError mencionando um nó do ciclo.",
        "Implemente {fn}(graph) por visita em profundidade com estados distintos para em andamento e concluído, preservando a ordem fornecida.",
        "Build deterministic {fn}(graph), visiting keys and dependency lists in received order, adding external dependencies, and reporting a participating node for cycles.",
    ),
    "memo_last": (
        "Retorne de {fn}(function) um wrapper com somente uma entrada de cache: args e kwargs iguais reutilizam o último sucesso, inclusive quando contêm valores não hashable.",
        "{fn} deve aceitar parâmetros posicionais e nomeados, substituir o cache após sucesso diferente e nunca armazenar uma chamada que levantou exceção.",
        "Implemente {fn}(function) comparando diretamente a última dupla args/kwargs, sem dicionário de cache baseado em hash.",
        "Create {fn}(function) as a one-result memoizer using equality for possibly unhashable positional and named arguments; failed calls must remain uncached.",
    ),
    "rolling_window": (
        "Calcule em {fn}(values, window) a média de cada janela completa numa única passagem, atualizando a soma em O(1) por elemento.",
        "{fn} deve aceitar generator numérico, retornar [] quando a entrada for curta e validar window como inteiro positivo que não seja bool.",
        "Implemente {fn}(values, window) com buffer limitado à janela, incluindo corretamente window=1 e sem indexar a fonte original.",
        "Write one-pass O(n) {fn}(values, window) for every full-window average over arbitrary numeric iterables, rejecting booleans and invalid sizes.",
    ),
    "transactional_update": (
        "Em {fn}(state, updates, validator), aplique cada par em ordem sobre uma cópia rasa e valide após cada mudança; false vira ValueError e exceções mantêm sua identidade.",
        "{fn} deve devolver um novo dict somente se toda validação passar, deixando state e updates intactos em sucesso ou falha.",
        "Implemente {fn} como transação em memória: nenhum estado parcial pode escapar quando validator rejeitar ou levantar erro.",
        "Implement atomic {fn}(state, updates, validator) on a shallow copy, validating each ordered mutation, converting false to ValueError, and propagating validator exceptions unchanged.",
    ),
}


def _format_many(lines: Sequence[str], fn: str) -> str:
    return "\n".join(line.replace("{fn}", fn).strip("\n") for line in lines)


def _answer(explanation: str, code: str, public_tests: str) -> str:
    return f"{explanation.strip()}\n\n```python\n{code.rstrip()}\n\n{public_tests.rstrip()}\n```"


def _candidate(
    *,
    source_id: str,
    group_id: str,
    source: str,
    category: str,
    prompt: str,
    response: str,
    verification_level: str,
    evidence: dict[str, Any],
    token_counter: Callable[[Sequence[dict[str, str]]], int],
    template_id: str | None = None,
    revision: str = "local",
    license_name: str = "cc-by-4.0",
    verified: bool = True,
) -> dict[str, Any]:
    if verification_level not in VERIFICATION_LEVELS:
        raise ValueError(f"verification_level inv\u00e1lido: {verification_level}")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt.strip()},
        {"role": "assistant", "content": response.strip()},
    ]
    fingerprint = sha256_text(json.dumps(messages, ensure_ascii=False, sort_keys=True))
    result = {
        "id": sha256_text(f"{source}:{source_id}:{fingerprint}"),
        "group_id": group_id,
        "source": source,
        "source_sha256": sha256_text(source),
        "source_id": source_id,
        "source_revision": revision,
        "revision_sha256": sha256_text(revision),
        "license": license_name,
        "category": category,
        "reasoning_band": "direct",
        "verified": verified,
        "verification_level": verification_level,
        "verification_evidence": evidence,
        "num_tokens": int(token_counter(messages)),
        "fingerprint": fingerprint,
        "near_fingerprint": sha256_text(normalized_prompt(prompt)),
        "messages": messages,
    }
    if template_id is not None:
        result["template_id"] = template_id
    return result


def generate_evaluation_tasks() -> dict[str, list[dict[str, Any]]]:
    dev: list[dict[str, Any]] = []
    hidden: list[dict[str, Any]] = []
    for spec in FAMILY_SPECS:
        prompt_templates = EVALUATION_PROMPTS[spec.name]
        for index, fn in enumerate(spec.eval_names):
            task = {
                "id": f"{'dev_v1' if index == 0 else 'corrective_hidden_v1'}:{spec.name}:{index}",
                "family": spec.name,
                "template_id": f"eval/{spec.name}/{index}",
                "prompt": prompt_templates[index].replace("{fn}", fn)
                + " Inclua explica\u00e7\u00e3o breve, exatamente um bloco Python e asserts.",
                "public_tests": _format_many(spec.public_tests, fn),
                "hidden_tests": _format_many(spec.hidden_tests, fn),
                "contract_checks": [
                    test.replace("{fn}", fn).strip("\n")
                    for test in spec.hidden_tests
                ],
            }
            (dev if index == 0 else hidden).append(task)
    return {"dev_v1": dev, "corrective_hidden_v1": hidden}


def generate_contract_candidates(
    *,
    token_counter: Callable[[Sequence[dict[str, str]]], int],
    candidate_token_target: int = 200_000,
    seed: int = 42,
    execute: bool = True,
    max_seq_length: int = 2048,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    language_cycle = [True] * 7 + [False] * 3
    rng.shuffle(language_cycle)
    candidates: list[dict[str, Any]] = []
    tokens = 0
    round_index = 0
    while tokens < candidate_token_target:
        spec = FAMILY_SPECS[round_index % len(FAMILY_SPECS)]
        archetype = (round_index // len(FAMILY_SPECS)) % 3
        base_fn = spec.train_names[archetype]
        fn = f"{base_fn}_{round_index:04d}"
        portuguese = language_cycle[round_index % len(language_cycle)]
        prompt_template = (spec.pt_prompts if portuguese else spec.en_prompts)[archetype]
        prompt = prompt_template.replace("{fn}", fn)
        prompt += (
            " Forne\u00e7a uma explica\u00e7\u00e3o curta, exatamente um bloco Python e asserts execut\u00e1veis."
            if portuguese
            else " Provide a short explanation, exactly one Python block, and executable asserts."
        )
        prompt += (
            f" Cen\u00e1rio determin\u00edstico {round_index}: cubra tamb\u00e9m entradas vazias e limites relevantes."
            if portuguese
            else f" Deterministic scenario {round_index}: also cover empty inputs and relevant boundaries."
        )
        code = spec.solution.replace("{fn}", fn)
        public_tests = _format_many(spec.public_tests, fn)
        hidden_tests = _format_many(spec.hidden_tests, fn)
        all_tests = public_tests + "\n" + hidden_tests
        if execute:
            result = run_verified(code, all_tests)
            if not result.passed:
                raise RuntimeError(
                    f"Fixture interno falhou ({spec.name}/{archetype}): {result.stderr}"
                )
            evidence = result.evidence(code, all_tests)
        else:
            evidence = {
                "status": "not_run",
                "code_sha256": sha256_text(code),
                "tests_sha256": sha256_text(all_tests),
                "runner_version": RUNNER_VERSION,
                "wall_ms": 0,
            }
            evidence["sha256"] = evidence_sha256(evidence)
        features = ", ".join(
            feature.replace("_", " ") for feature in spec.contract_checks
        )
        explanation = (
            f"A solu\u00e7\u00e3o aplica {features}, validando o contrato antes de produzir o resultado."
            if portuguese
            else f"The solution applies {features}, validating the contract before producing the result."
        )
        response = _answer(explanation, code, public_tests)
        template_id = f"train/{spec.name}/{archetype}"
        candidate = _candidate(
            source_id=f"{spec.name}:{round_index}",
            group_id=template_id,
            source="logos/corrective-contracts-v1",
            category="corrective_contracts",
            prompt=prompt,
            response=response,
            verification_level="local_hidden_tests",
            evidence=evidence,
            token_counter=token_counter,
            template_id=template_id,
            license_name="project-generated",
            verified=execute,
        )
        if candidate["num_tokens"] > max_seq_length:
            raise RuntimeError(
                f"Microcontrato interno excede {max_seq_length} tokens: "
                f"{candidate['id']}"
            )
        candidates.append(candidate)
        tokens += candidate["num_tokens"]
        round_index += 1
        if round_index > 100_000:
            raise RuntimeError("N\u00e3o foi poss\u00edvel atingir o alvo de microcontratos.")

    evaluation = generate_evaluation_tasks()
    assert_not_contaminated(
        candidates,
        [*evaluation["dev_v1"], *evaluation["corrective_hidden_v1"]],
    )
    return candidates


def _literal_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, str):
        raise ValueError(f"{label} deve ser uma representa\u00e7\u00e3o textual de lista.")
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"{label} malformado.") from exc
    if not isinstance(parsed, list) or not parsed:
        raise ValueError(f"{label} deve ser uma lista n\u00e3o vazia.")
    return parsed


def canonicalize_opencode_row(
    row: dict[str, Any],
    *,
    token_counter: Callable[[Sequence[dict[str, str]]], int],
    execute: bool = True,
) -> dict[str, Any]:
    try:
        score = float(row["average_test_score"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("average_test_score ausente ou inv\u00e1lido.") from exc
    if score != 1.0:
        raise ValueError("average_test_score n\u00e3o \u00e9 1.0.")
    statuses = _literal_list(row.get("tests_execution_status"), "tests_execution_status")
    if not all(isinstance(item, str) and item.lower() == "pass" for item in statuses):
        raise ValueError("Nem todos os testes publicados passaram.")
    tests = _literal_list(row.get("unit_tests"), "unit_tests")
    if not all(isinstance(item, str) and item.strip() for item in tests):
        raise ValueError("unit_tests cont\u00e9m item inv\u00e1lido.")
    if len(statuses) != len(tests):
        raise ValueError("tests_execution_status e unit_tests t\u00eam tamanhos diferentes.")
    try:
        judgement = json.loads(row["llm_judgement"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("llm_judgement ausente ou inv\u00e1lido.") from exc
    for criterion in (
        "requirement_conformance",
        "logical_correctness",
        "edge_case_consideration",
    ):
        if not isinstance(judgement.get(criterion), dict) or judgement[criterion].get("score") != 5:
            raise ValueError(f"{criterion} n\u00e3o recebeu 5/5.")

    code = extract_single_python_block(str(row.get("output", "")))
    tests_text = "\n".join(item.strip() for item in tests)
    if execute:
        execution = run_verified(code, tests_text)
        if not execution.passed:
            raise ValueError(f"A reexecu\u00e7\u00e3o local falhou: {execution.status}")
        evidence = execution.evidence(code, tests_text)
    else:
        evidence = {
            "status": "not_run",
            "code_sha256": sha256_text(code),
            "tests_sha256": sha256_text(tests_text),
            "runner_version": RUNNER_VERSION,
            "wall_ms": 0,
        }
        evidence["sha256"] = evidence_sha256(evidence)

    explanation = str(judgement["logical_correctness"].get("justification", "")).strip()
    explanation = re.sub(r"```.*?```", "", explanation, flags=re.DOTALL).strip()
    if not explanation:
        raise ValueError("logical_correctness.justification est\u00e1 vazio.")
    explanation = " ".join(explanation.split())[:600].rstrip()
    prompt = str(row.get("input", "")).strip()
    if not prompt:
        raise ValueError("input vazio.")
    prompt += (
        "\n\nProvide a brief explanation, exactly one Python code block, and executable asserts in that block."
    )
    response = _answer(explanation, code, tests_text)
    source_id = str(row.get("id", sha256_text(prompt)))
    return _candidate(
        source_id=source_id,
        group_id=f"{OPENCODE_NAME}:{source_id}",
        source=OPENCODE_NAME,
        category="corrective_opencode",
        prompt=prompt,
        response=response,
        verification_level="local_public_tests",
        evidence=evidence,
        token_counter=token_counter,
        revision=OPENCODE_REVISION,
        verified=execute,
    )


def collect_opencode_candidates(
    rows: Iterable[dict[str, Any]],
    *,
    token_counter: Callable[[Sequence[dict[str, str]]], int],
    candidate_token_target: int,
    evaluation_tasks: Sequence[dict[str, Any]],
    execute: bool = True,
    max_seq_length: int = 2048,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    accepted: list[dict[str, Any]] = []
    rejections: dict[str, int] = {}
    tokens = 0
    scanned = 0
    for row in rows:
        if tokens >= candidate_token_target:
            break
        scanned += 1
        try:
            candidate = canonicalize_opencode_row(
                row, token_counter=token_counter, execute=execute
            )
            if int(candidate["num_tokens"]) > max_seq_length:
                raise ValueError(f"too_long>{max_seq_length}")
            assert_not_contaminated([candidate], evaluation_tasks)
        except (ValueError, RuntimeError) as exc:
            reason = str(exc).split(":", 1)[0]
            rejections[reason] = rejections.get(reason, 0) + 1
            if scanned % 1000 == 0:
                print(
                    "OpenCodeInstruct: "
                    f"scanned={scanned:,}, accepted={len(accepted):,}, "
                    f"tokens={tokens:,}, rejected={sum(rejections.values()):,}",
                    flush=True,
                )
            continue
        accepted.append(candidate)
        tokens += int(candidate["num_tokens"])
        if scanned % 1000 == 0:
            print(
                "OpenCodeInstruct: "
                f"scanned={scanned:,}, accepted={len(accepted):,}, "
                f"tokens={tokens:,}, rejected={sum(rejections.values()):,}",
                flush=True,
            )
    if scanned and scanned % 1000:
        print(
            "OpenCodeInstruct: "
            f"scanned={scanned:,}, accepted={len(accepted):,}, "
            f"tokens={tokens:,}, rejected={sum(rejections.values()):,}",
            flush=True,
        )
    return accepted, rejections


def load_replay_candidates(
    pilot_run_path: str | Path,
    *,
    token_counter: Callable[[Sequence[dict[str, str]]], int],
    max_seq_length: int = 2048,
) -> list[dict[str, Any]]:
    run_path = Path(pilot_run_path)
    manifest_path = run_path / "adapter" / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifesto do campe\u00e3o ausente: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("data", {})
    checks = (
        (run_path / "data" / "train.jsonl", "train_sha256"),
        (run_path / "data" / "validation.jsonl", "validation_sha256"),
        (run_path / "data" / "dataset_report.json", "dataset_report_sha256"),
    )
    actual: dict[str, str] = {}
    for path, key in checks:
        expected_hash = expected.get(key)
        if not path.exists() or not expected_hash:
            raise ValueError(f"Replay incompleto: {path} ou {key} ausente.")
        actual_hash = file_sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"Hash do replay divergente: {path}")
        actual[key] = actual_hash

    rows: list[dict[str, Any]] = []
    train_path = checks[0][0]
    for index, row in enumerate(read_jsonl(train_path)):
        source_name = str(row.get("source", "")).strip()
        license_name = str(
            row.get("license") or REPLAY_SOURCE_LICENSES.get(source_name, "")
        ).strip().lower()
        if not source_name or license_name not in ALLOWED_LICENSES:
            continue
        messages = []
        for message in row.get("messages", []):
            content = str(message.get("content", ""))
            if message.get("role") == "assistant":
                content = THINK_RE.sub("", content).strip()
            if content:
                messages.append({"role": str(message["role"]), "content": content})
        if not messages or not any(item["role"] == "assistant" for item in messages):
            continue
        source_id = str(row.get("id", index))
        data_hash = sha256_text(json.dumps(messages, ensure_ascii=False, sort_keys=True))
        evidence = {
            "status": "passed",
            "data_sha256": data_hash,
            "runner_version": "pilot-manifest-replay-v1",
            "wall_ms": 0,
        }
        evidence["sha256"] = evidence_sha256(evidence)
        fingerprint = sha256_text(json.dumps(messages, ensure_ascii=False, sort_keys=True))
        candidate = {
            "id": sha256_text(f"replay:{source_id}:{fingerprint}"),
            "group_id": str(row.get("group_id", f"pilot:{source_id}")),
            "source": source_name,
            "source_sha256": sha256_text(source_name),
            "source_id": source_id,
            "source_revision": "pilot_500k_step7",
            "revision_sha256": sha256_text("pilot_500k_step7"),
            "license": license_name,
            "category": "corrective_replay",
            "reasoning_band": "direct",
            "verified": True,
            "verification_level": "replay_champion",
            "verification_evidence": evidence,
            "num_tokens": int(token_counter(messages)),
            "fingerprint": fingerprint,
            "near_fingerprint": sha256_text(
                normalized_prompt(
                    next(
                        item["content"]
                        for item in messages
                        if item["role"] == "user"
                    )
                )
            ),
            "messages": messages,
        }
        if candidate["num_tokens"] <= max_seq_length:
            rows.append(candidate)
    if not rows:
        raise ValueError("O treino preservado do piloto n\u00e3o cont\u00e9m exemplos reutiliz\u00e1veis.")
    return rows


def verify_candidate(candidate: dict[str, Any]) -> None:
    if candidate.get("verified") is not True:
        raise ValueError("Candidato corretivo sem verified=true.")
    if candidate.get("verification_level") not in VERIFICATION_LEVELS:
        raise ValueError("verification_level ausente ou inv\u00e1lido.")
    source = str(candidate.get("source", "")).strip()
    revision = str(candidate.get("source_revision", "")).strip()
    if not source or candidate.get("source_sha256") != sha256_text(source):
        raise ValueError("Hash da fonte ausente ou divergente.")
    if not revision or candidate.get("revision_sha256") != sha256_text(revision):
        raise ValueError("Hash da revis\u00e3o ausente ou divergente.")
    if str(candidate.get("license", "")).strip().lower() not in ALLOWED_LICENSES:
        raise ValueError("Licen\u00e7a ausente ou n\u00e3o permitida.")
    evidence = candidate.get("verification_evidence")
    if not isinstance(evidence, dict) or evidence.get("status") != "passed":
        raise ValueError("Evid\u00eancia de verifica\u00e7\u00e3o ausente ou reprovada.")
    if not evidence.get("sha256"):
        raise ValueError("Hash da evid\u00eancia ausente.")
    if evidence["sha256"] != evidence_sha256(evidence):
        raise ValueError("Hash da evid\u00eancia divergente.")
    if candidate["verification_level"] == "replay_champion":
        if not evidence.get("data_sha256"):
            raise ValueError("Replay sem data_sha256.")
    elif not evidence.get("code_sha256") or not evidence.get("tests_sha256"):
        raise ValueError("Execu\u00e7\u00e3o local sem hashes de c\u00f3digo e testes.")


def build_manifest(
    *,
    candidates_path: str | Path,
    candidates: Sequence[dict[str, Any]],
    evaluation_paths: dict[str, str | Path],
    config: dict[str, Any],
    allow_unverified: bool = False,
) -> dict[str, Any]:
    if not allow_unverified:
        for candidate in candidates:
            verify_candidate(candidate)
    path = Path(candidates_path)
    candidate_hash = file_sha256(path)
    artifacts: dict[str, Any] = {
        "candidates": {
            "path": str(path),
            "sha256": candidate_hash,
            "examples": len(candidates),
            "tokens": sum(int(item["num_tokens"]) for item in candidates),
        }
    }
    for name, eval_path_value in evaluation_paths.items():
        eval_path = Path(eval_path_value)
        artifacts[name] = {
            "path": str(eval_path),
            "sha256": file_sha256(eval_path),
            "examples": sum(1 for _ in read_jsonl(eval_path)),
        }
    counts: dict[str, dict[str, int]] = {}
    for category in ("corrective_contracts", "corrective_opencode", "corrective_replay"):
        selected = [item for item in candidates if item["category"] == category]
        counts[category] = {
            "examples": len(selected),
            "tokens": sum(int(item["num_tokens"]) for item in selected),
        }
    sandbox_path = Path(__file__).with_name("sandbox.py")
    manifest = {
        "schema_version": 1,
        "stage": "corrective_v1",
        "seed": 42,
        "source": {
            "name": OPENCODE_NAME,
            "name_sha256": sha256_text(OPENCODE_NAME),
            "revision": OPENCODE_REVISION,
            "revision_sha256": sha256_text(OPENCODE_REVISION),
        },
        "sources": [
            {
                "name": source,
                "name_sha256": sha256_text(source),
                "revision": revision,
                "revision_sha256": sha256_text(revision),
            }
            for source, revision in sorted(
                {
                    (str(item["source"]), str(item["source_revision"]))
                    for item in candidates
                }
            )
        ],
        "executor": {
            "runner_version": RUNNER_VERSION,
            "sha256": file_sha256(sandbox_path),
        },
        "config": config,
        "counts": counts,
        "candidates_sha256": candidate_hash,
        "artifacts": artifacts,
    }
    manifest["manifest_payload_sha256"] = manifest_payload_sha256(manifest)
    return manifest
