# SFT Qwen3-8B em NVIDIA L4

Implementação reproduzível de
[`receita_sft_qwen3_8b_l4.md`](receita_sft_qwen3_8b_l4.md) para Google Colab
Pro. O pipeline usa QLoRA NF4 com TRL + PEFT, mistura o corpus por **tokens**,
treina somente sobre mensagens do assistente e registra dados suficientes para
comparar o modelo-base com cada checkpoint.

O fluxo entregue cobre:

- download em streaming das seis fontes da receita;
- normalização para `messages` no chat template oficial do Qwen3;
- remoção de segredos, holdouts, loops, duplicatas e exemplos inválidos;
- segmentação por turnos de traces que excedem 4096 tokens;
- mistura `60/20/10/5/5` no baseline e `45/20/20/10/5` no SFT principal;
- distribuição de raciocínio `55/25/10/10`;
- QLoRA rank 32 sobre atenção e MLP, batch efetivo 16;
- retomada automática e continuação agentic a partir de um adapter escolhido;
- seleção do checkpoint pela fórmula da receita.

## Caminho mais curto: Colab

Abra e execute
[`notebooks/qwen3_8b_l4_sft_colab.ipynb`](notebooks/qwen3_8b_l4_sft_colab.ipynb).
O notebook começa em modo smoke test: prepara até 250 mil tokens e executa 30
steps. Depois de validar o ambiente, troque `SMOKE_TEST` para `False` para usar o
orçamento configurado.

Crie um secret `HF_TOKEN` no Colab. Se o repositório não for público, crie
também `GH_TOKEN` com permissão somente de leitura.

## Execução pela CLI

No Colab/Linux com GPU:

```bash
python -m pip install -r requirements-colab.txt
python -m pip install -e . --no-deps
python scripts/environment_check.py
```

Valide o pipeline com um corpus pequeno:

```bash
python scripts/prepare_data.py \
  --stage baseline \
  --token-budget 250000 \
  --max-source-rows 1000
pytest -q
python scripts/train_sft.py \
  --stage baseline \
  --max-steps 30 \
  --max-train-samples 250
```

Quando o smoke test estiver estável, execute as duas etapas úteis:

```bash
python scripts/prepare_data.py --stage baseline
python scripts/train_sft.py --stage baseline --resume-from-checkpoint auto

python scripts/prepare_data.py --stage main
python scripts/train_sft.py --stage main --resume-from-checkpoint auto
```

Os padrões são 15 milhões de tokens no baseline e 50 milhões no treino
principal, ambos dentro das faixas da receita. Altere apenas `token_budget` em
[`configs/recipe.yaml`](configs/recipe.yaml) para outro ponto das faixas
10–20M e 30–80M.

## Dados

Cada linha final tem `messages` e metadados de auditoria:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "<think>...</think>\n..."}
  ],
  "source": "nvidia/OpenCodeReasoning",
  "category": "verified_code",
  "reasoning_band": "short",
  "verified": true,
  "num_tokens": 1734
}
```

O relatório `data/processed/<stage>/dataset_report.json` mostra tokens por
categoria, comprimento de raciocínio, rejeições por fonte e isolamento de
grupos entre treino e validação. `budget_reached=false` significa que a
varredura não encontrou candidatos suficientes; aumente `--max-source-rows` ou
remova esse limite.

Adicione IDs reservados para avaliação em
[`data/eval_holdout_ids.txt`](data/eval_holdout_ids.txt) **antes** do
preprocessing. Consulte [`DATA_SOURCES.md`](DATA_SOURCES.md) para licenças e
proveniência.

## Treino

Os parâmetros centrais seguem a receita:

- `Qwen/Qwen3-8B`, NF4 4-bit, double quant e BF16;
- sequência de 4096, packing e SDPA;
- LoRA `r=32`, `alpha=64`, dropout `0.05`;
- batch físico 1, acumulação 16;
- cosine, warmup 3%, learning rate `1e-4`;
- checkpoint e avaliação a cada 250 steps;
- loss somente nos tokens do assistente.

O packing usa a estratégia `wrapped` porque o `bfd` atual ativa
`padding_free`, que depende de FlashAttention. Isso mantém a instalação da L4
mais previsível sem desativar packing.

Se houver OOM, aplique nesta ordem em `configs/recipe.yaml`: comprimento 3072,
confirme batch 1 e checkpointing, depois LoRA rank 16. Só então desative a
avaliação com `--no-eval`.

## Avaliação e checkpoint

Avalie o base e todos os checkpoints no mesmo conjunto descontaminado. O
pipeline deixa a execução de código gerado para um container/VM separado:
instale [`requirements-eval.txt`](requirements-eval.txt) nesse ambiente e use
EvalPlus/lm-evaluation-harness, além dos testes próprios de debugging e SWE.
Veja o contrato completo em [`EVALUATION.md`](EVALUATION.md).

Gere respostas do base e de um adapter, uma cópia do modelo por vez:

```bash
python scripts/generate_evaluation.py \
  --tasks data/evaluation/tasks.jsonl \
  --output outputs/evaluations/base_predictions.jsonl
python scripts/generate_evaluation.py \
  --tasks data/evaluation/tasks.jsonl \
  --adapter outputs/adapters/main \
  --output outputs/evaluations/main_predictions.jsonl
```

Depois, crie um JSON por checkpoint seguindo
[`examples/checkpoint_metrics.example.json`](examples/checkpoint_metrics.example.json)
e rode:

```bash
python scripts/select_checkpoint.py outputs/evaluations/*.json
```

Para a etapa agentic, defina `stages.agentic.training.adapter_path` como o
adapter vencedor, prepare apenas as trajetórias OpenHands curadas e
bem-sucedidas e continue:

```bash
python scripts/prepare_data.py --stage agentic
python scripts/train_sft.py --stage agentic --resume-from-checkpoint auto
```

O learning rate agentic padrão é `3e-5`, dentro da faixa `2e-5`–`5e-5`.
