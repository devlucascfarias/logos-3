# Protocolo de avaliação

Compare `Qwen/Qwen3-8B` e cada checkpoint com o mesmo prompt, modo de thinking,
seed, limites de geração e conjunto descontaminado. Benchmarks gerais continuam
em container/VM sem credenciais. O `corrective_v1` usa um executor Colab
específico, fail-closed, com processo isolado e limites de AST, CPU, memória,
tempo e saída descritos abaixo.

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

## Gate da continuação

Compare a continuação com o piloto preservado e o modelo-base:

```bash
python scripts/compare_adapter.py \
  --stage pilot_continuation \
  --reference-adapter-path /caminho/para/pilot_500k_step7/adapter \
  --output-dir outputs/evaluations/pilot_continuation \
  --seed 20260722
```

A continuação só substitui o piloto preservado depois da avaliação cega. Os
rótulos A/B/C são embaralhados separadamente por prompt; agregue as notas por
identidade somente após abrir `mapping.json`.

## Gate `corrective_v1`

A geração corretiva é fixa: modo direto, greedy, `max_new_tokens=512`, mesma
revisão do modelo e mesmo chat template. O modo é ativado com
`--decoding deterministic`; alterar thinking ou o limite de tokens nesse modo é
erro.

Primeiro gere previsões para todos os checkpoints no `dev_v1`:

```bash
python scripts/generate_evaluation.py \
  --tasks data/interim/corrective_v1/dev_v1.jsonl \
  --adapter outputs/checkpoints/corrective_v1/checkpoint-2 \
  --output outputs/evaluations/corrective_v1/checkpoint-2-dev.jsonl \
  --decoding deterministic \
  --seed 20260722

python scripts/evaluate_contracts.py \
  --tasks data/interim/corrective_v1/dev_v1.jsonl \
  --predictions outputs/evaluations/corrective_v1/*-dev.jsonl \
  --output outputs/evaluations/corrective_v1/dev_report.json
```

O ranking desempata por tarefas funcionais aprovadas, fração de contratos,
formato, menos timeout/truncamento e, por último, menor `eval_loss`. Somente os
três primeiros seguem para as 33 tarefas de `corrective_hidden_v1`, junto do
modelo-base e do campeão preservado.

O executor exige exatamente um bloco Python e uma explicação não vazia. A AST
proíbe imports, classes, dunders, arquivos, rede, processos, reflexão, `eval`,
`exec`, `compile` e `open`. Cada resposta roda em processo novo com
`python -I -S`, diretório temporário, 256 MiB, CPU de 2 segundos, timeout total
de 3 segundos e saída máxima de 64 KiB. Em ambiente não POSIX ou se qualquer
limite de isolamento falhar, a amostra é reprovada; não existe fallback
permissivo.

Depois do hidden, faça a comparação cega manual apenas para o melhor finalista:

```bash
python scripts/compare_adapter.py \
  --stage corrective_v1 \
  --adapter-path outputs/checkpoints/corrective_v1/checkpoint-N \
  --reference-adapter-path /content/pilot_500k_step7/adapter \
  --output-dir outputs/evaluations/corrective_v1/regression \
  --max-new-tokens 512 \
  --seed 20260722 \
  --no-thinking
```

Preencha `ratings.json` antes de abrir `mapping.json`. O gate final exige:

- corpus com pelo menos 304 mil tokens e desvio máximo de 5% por bucket;
- 100% das linhas novas verificadas, sem overlap ou contaminação;
- candidato pelo menos três tarefas acima do campeão no hidden;
- nenhum recuo em formato ou contratos e perda máxima de uma tarefa por família;
- regressão cega com total mínimo 84/195, correção 37, instruções 29 e
  explicação 17, além de superar o modelo-base.

O script combina os dois relatórios e registra os hashes usados:

```bash
python scripts/evaluate_contracts.py \
  --tasks data/interim/corrective_v1/corrective_hidden_v1.jsonl \
  --predictions outputs/evaluations/corrective_v1/*-hidden.jsonl \
  --output outputs/evaluations/corrective_v1/hidden_report.json \
  --promotion-output outputs/evaluations/corrective_v1/promotion_gate.json \
  --ratings outputs/evaluations/corrective_v1/regression/ratings.json \
  --mapping outputs/evaluations/corrective_v1/regression/mapping.json \
  --candidate outputs/checkpoints/corrective_v1/checkpoint-N \
  --champion /content/pilot_500k_step7/adapter \
  --base Qwen/Qwen3-8B
```

Se qualquer condição falhar, o resultado é `MANTER CAMPEAO` e
`pilot_500k_step7` permanece inalterado.
