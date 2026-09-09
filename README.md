# SFT Qwen3-8B em NVIDIA L4

O projeto implementa um pipeline completo de fine-tuning do **Qwen3-8B** usando **QLoRA em 4 bits**, pensado para rodar em uma GPU NVIDIA L4. O foco é treinar o modelo com dados de programação, raciocínio e comportamento agentic sem simplesmente misturar grandes volumes de texto: o pipeline controla a proporção dos dados por quantidade de tokens, remove duplicatas e exemplos problemáticos, separa dados de avaliação e treina apenas sobre as respostas do assistente.

Além do treinamento, o projeto inclui um processo de seleção e correção dos modelos gerados. Cada checkpoint é avaliado em tarefas que o modelo não viu durante o treino, incluindo testes de código, debugging e regressão, e só é promovido quando realmente supera o modelo anterior sem piorar comportamentos já adquiridos. O pipeline também permite continuar o treinamento a partir do melhor adapter e aplicar rodadas corretivas com novos dados, mantendo rastreabilidade e verificações para garantir que cada etapa seja reproduzível.

mais detalhes em https://github.com/devlucascfarias/logos-3

## Caminho mais curto: Colab

Abra e execute
[`notebooks/qwen3_8b_l4_sft_colab.ipynb`](notebooks/qwen3_8b_l4_sft_colab.ipynb).
O notebook está configurado para `corrective_v1`. Ele copia o campeão de 500k
do Drive, verifica seus hashes, desmonta o Drive, constrói um corpus corretivo
assinado de 320 mil tokens e treina uma época com learning rate `1e-5`. O
campeão original nunca é sobrescrito. Ao final, o notebook ranqueia todos os
checkpoints em `dev_v1`, avalia os três finalistas no hidden de 33 tarefas e
gera a comparação cega dos 13 prompts de regressão.

Crie um secret `HF_TOKEN` no Colab. Se o repositório não for público, crie
também `GH_TOKEN` com permissão somente de leitura.

## Execução pela CLI

No Colab/Linux com GPU:

```bash
python -m pip install -r requirements-colab.txt
python -m pip install -e . --no-deps
python scripts/environment_check.py
```

Execute o piloto candidato:

```bash
python scripts/prepare_data.py \
  --stage pilot
pytest -q
python scripts/train_sft.py \
  --stage pilot \
  --resume-from-checkpoint none
```

Para a rodada corretiva, primeiro copie
`pilot_500k_step7/{adapter,data}` do Drive para um diretório local e então rode:

```bash
python scripts/build_corrective_corpus.py \
  --pilot-run-path /content/pilot_500k_step7 \
  --output-dir data/interim/corrective_v1
python scripts/prepare_data.py \
  --stage corrective_v1 \
  --candidates-jsonl data/interim/corrective_v1/candidates.jsonl \
  --candidate-manifest data/interim/corrective_v1/candidate_manifest.json
python scripts/train_sft.py \
  --stage corrective_v1 \
  --adapter-path /content/pilot_500k_step7/adapter \
  --resume-from-checkpoint none
```

Não use `--skip-execution` para materializar dados de treino. Essa opção existe
somente para testes do builder e produz um manifesto recusado por
`prepare_data.py`.

Somente depois de o piloto superar o modelo-base e o adapter campeão, execute
as etapas maiores:

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
  "verification_level": "local_hidden_tests",
  "verification_evidence": {
    "status": "passed",
    "code_sha256": "...",
    "tests_sha256": "...",
    "sha256": "..."
  },
  "num_tokens": 1734
}
```

O relatório `data/processed/<stage>/dataset_report.json` mostra tokens por
categoria, comprimento de raciocínio, rejeições por fonte e isolamento de
grupos entre treino e validação. As matrizes `candidate_matrix` e
`selected_matrix` permitem localizar exatamente qual combinação de categoria e
faixa de raciocínio está escassa. `budget_reached=false` significa que a
varredura não encontrou candidatos suficientes. O preparo interrompe antes do
treino quando a cobertura fica abaixo de 95% ou qualquer uma das duas
distribuições excede o desvio máximo configurado. A validação reserva pelo
menos 32 exemplos sem sobreposição de grupos.

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

O estágio `pilot` é deliberadamente mais conservador: 500 mil tokens, learning
rate `5e-5`, uma época e checkpoints com avaliação a cada passo. Seus
artefatos ficam em `outputs/checkpoints/pilot` e `outputs/adapters/pilot`.

Para continuar o piloto sem reiniciar o scheduler antigo nem refazer o mix:

```bash
python scripts/train_sft.py \
  --stage pilot_continuation \
  --data-stage pilot \
  --adapter-path /caminho/para/pilot_500k_step7/adapter \
  --resume-from-checkpoint none
```

A continuação exige o `run_manifest.json` do adapter inicial e recusa o treino
se os hashes de `train.jsonl`, `validation.jsonl` ou `dataset_report.json`
divergirem. As saídas ficam em `outputs/checkpoints/pilot_continuation` e
`outputs/adapters/pilot_continuation`.

O `corrective_v1` parte do adapter campeão, mas usa dados novos. Seu mix é
`50/35/15` por tokens entre microcontratos próprios, OpenCodeInstruct verificado
e replay do campeão. O stage usa sequências de 2048, batch físico 1, acumulação
8 e bloqueia o treino se a estimativa estiver fora de 15–25 atualizações. A
primeira execução deve usar `--resume-from-checkpoint none`; use `auto` apenas
para recuperar uma interrupção.

O packing usa a estratégia `wrapped` porque o `bfd` atual ativa
`padding_free`, que depende de FlashAttention. Isso mantém a instalação da L4
mais previsível sem desativar packing.

Se houver OOM, aplique nesta ordem em `configs/recipe.yaml`: comprimento 3072,
confirme batch 1 e checkpointing, depois LoRA rank 16. Só então desative a
avaliação com `--no-eval`.

## Avaliação e checkpoint

Para a rodada corretiva, gere respostas determinísticas de cada checkpoint no
`dev_v1` e execute os testes ocultos:

```bash
python scripts/generate_evaluation.py \
  --tasks data/interim/corrective_v1/dev_v1.jsonl \
  --adapter outputs/checkpoints/corrective_v1/checkpoint-2 \
  --output outputs/evaluations/corrective_v1/checkpoint-2-dev.jsonl \
  --decoding deterministic
python scripts/evaluate_contracts.py \
  --tasks data/interim/corrective_v1/dev_v1.jsonl \
  --predictions outputs/evaluations/corrective_v1/*-dev.jsonl \
  --output outputs/evaluations/corrective_v1/dev_report.json
```

Depois, compare cegamente base, melhor finalista e campeão nos mesmos 13
prompts de regressão:

```bash
python scripts/compare_adapter.py \
  --stage corrective_v1 \
  --adapter-path outputs/checkpoints/corrective_v1/checkpoint-N \
  --reference-adapter-path /content/pilot_500k_step7/adapter \
  --output-dir outputs/evaluations/corrective_v1/regression \
  --seed 20260722
```

Avalie `outputs/evaluations/corrective_v1/regression/comparison.md` antes de consultar
`mapping.json`. Promova o candidato somente se ele superar o campeão em
ao menos três tarefas no hidden, não regredir contratos/formato e passar os
limites da regressão cega; `eval_loss` é apenas o último desempate.

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
