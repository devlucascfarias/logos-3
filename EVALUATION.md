# Protocolo de avaliação

Compare `Qwen/Qwen3-8B` e cada checkpoint com o mesmo prompt, modo de thinking,
seed, limites de geração e conjunto descontaminado. Execute código gerado em um
container/VM sem credenciais nem acesso ao host.

O gerador aceita tarefas JSONL com `id`, `prompt` (ou `messages`), `category` e,
opcionalmente, `tools`:

```bash
python scripts/generate_evaluation.py \
  --tasks data/evaluation/tasks.jsonl \
  --output outputs/evaluations/base_predictions.jsonl

python scripts/generate_evaluation.py \
  --tasks data/evaluation/tasks.jsonl \
  --adapter outputs/adapters/main \
  --output outputs/evaluations/main_predictions.jsonl
```

Ele apenas gera respostas; não executa o código produzido.

## Métricas mínimas

1. **Código verificado (`code_verified`)**: HumanEval+, MBPP+, subconjunto
   descontaminado do LiveCodeBench e testes próprios em Python/C++.
2. **Debugging (`debugging`)**: fração de correções próprias que passam todos os
   testes, com explicação causal válida.
3. **SWE (`swe_tasks`)**: patches aprovados em pequenos repositórios e em um
   subconjunto SWE-Bench integralmente reservado.
4. **Ferramentas (`tool_use`)**: chamadas válidas, conclusão sem loop e número
   médio de ações.
5. **Qualidade geral (`general_quality`)**: formato, instrução, utilidade e
   regressão em perguntas gerais.
6. **Repetição (`repetition`)**: taxa normalizada de respostas repetitivas.
7. **Overthinking (`overthinking`)**: taxa normalizada de raciocínio
   desnecessariamente longo.

Normalize todas as métricas no intervalo `[0, 1]`, onde as cinco primeiras são
melhores quando maiores e as duas últimas são penalidades.

## Decodificação

Para thinking mode, use `temperature=0.6`, `top_p=0.95`, `top_k=20`; não use
greedy. Para respostas diretas, use `temperature=0.7`, `top_p=0.8`,
`top_k=20`. Registre a configuração junto de cada relatório.

## Contrato do relatório

```json
{
  "checkpoint": "outputs/checkpoints/main/checkpoint-1000",
  "metrics": {
    "code_verified": 0.72,
    "debugging": 0.68,
    "swe_tasks": 0.54,
    "tool_use": 0.71,
    "general_quality": 0.66,
    "repetition": 0.02,
    "overthinking": 0.04
  }
}
```

O seletor aplica:

```text
0.40 * code_verified
+ 0.25 * debugging
+ 0.15 * swe_tasks
+ 0.10 * tool_use
+ 0.10 * general_quality
- repetition
- overthinking
```

Não escolha automaticamente o último checkpoint.

## Gate do piloto

O piloto de 500k deve ser comparado cegamente com o modelo-base e com o adapter
campeão de 250k:

```bash
python scripts/compare_adapter.py \
  --stage pilot \
  --reference-adapter-path /caminho/para/o/adapter-campeao \
  --output-dir outputs/evaluations/pilot_500k \
  --seed 20260722
```

Preencha `comparison.md`/`ratings.json` antes de abrir `mapping.json`. O piloto
só substitui o campeão se ganhar em correção e cumprimento das instruções sem
regredir repetição, truncamento ou overthinking. Se nenhum checkpoint passar
esse gate, mantenha o campeão e revise os dados; não promova pelo `eval_loss`.
