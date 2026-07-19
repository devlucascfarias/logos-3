# Fable → Qwen3-8B Distillation

Pipeline reprodutível para pós-treinamento do `Qwen/Qwen3-8B-Base` como agente de
engenharia de software. O projeto foi desenhado para Google Colab Pro com uma
NVIDIA L4 de 24 GB: QLoRA NF4, BF16, SDPA, batch por dispositivo igual a 1,
gradient checkpointing e retomada automática.

O repositório separa claramente:

- download/auditoria dos dados;
- normalização, remoção de informações sensíveis e proveniência;
- deduplicação e split seguro por repositório/sessão;
- construção de exemplos `next_action`, `trajectory` e `final_answer`;
- treino SFT com loss apenas nos tokens do assistente;
- preferência verificável opcional, somente depois do SFT;
- avaliação estática e um harness local restrito para tarefas executáveis.

## Início rápido local

Requer Python 3.10 ou 3.11. A instalação completa inclui dependências CUDA; para
rodar somente os testes de preprocessing, instale o projeto com as dependências
de desenvolvimento.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest -q
```

No Windows PowerShell, ative o ambiente com:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instalação de treino:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

`flash-attn` é deliberadamente opcional. O padrão é SDPA, mais previsível no
Colab. Instale Flash Attention separadamente apenas se a imagem CUDA suportar.

## Execução no Colab

Abra somente:

```text
notebooks/fable_qwen3_8b_colab_l4.ipynb
```

Execute as células de cima para baixo. O notebook único contém ambiente,
clone/pull do repositório, autenticação no Hugging Face, download, auditoria,
preprocessing, testes, Stage 1, avaliação, Stage 2 opcional e exportação.

Na primeira célula de controles, mantenha `SMOKE_TEST=True` para validar a
pipeline com 750 exemplos e 100 steps. Depois, altere para `False` para executar
o treinamento principal. Stage 2, comparação completa e exportação permanecem
desativados até serem explicitamente habilitados.

O notebook clona `https://github.com/devlucascfarias/logos-3.git` em
`/content/logos-3`. Se já houver um clone limpo, executa
`git pull --ff-only origin main`.

Como o repositório é privado, adicione também um secret `GH_TOKEN` no Colab com
permissão de leitura do repositório. A célula usa o token por um header de
ambiente do Git, sem inseri-lo na URL, nos logs ou no traceback.

Por padrão, artefatos persistentes ficam em
`/content/drive/MyDrive/fable-qwen-distillation`. O notebook é uma interface
fina para os mesmos scripts CLI usados localmente.

Antes da seção de download, crie um secret chamado `HF_TOKEN` no painel
**Secrets** do Colab e habilite seu acesso ao notebook. A célula autentica com
`huggingface_hub.login`, repassa o token ao subprocesso de download e nunca o
imprime ou inclui nos manifests.

## Pipeline CLI

```bash
python scripts/download_datasets.py --config configs/data.yaml
python scripts/audit_datasets.py --config configs/data.yaml
python scripts/build_examples.py --config configs/data.yaml
python scripts/deduplicate.py \
  --input data/interim/examples.jsonl \
  --output data/interim/examples.dedup.jsonl
python scripts/split_by_group.py \
  --input data/interim/examples.dedup.jsonl \
  --output-dir data/processed
pytest -q
python scripts/train_sft.py --config configs/sft_stage1.yaml \
  --resume-from-checkpoint auto
python scripts/evaluate_static.py --config configs/eval.yaml
```

Para gerar uma comparação reproduzível entre o base e todos os adapters
configurados (em um runtime CUDA):

```bash
python scripts/evaluate_static.py --config configs/eval.yaml \
  --generate-models --output outputs/evaluations/model_comparison.json
```

Para um smoke test da Fase 0, limite a construção e o treino:

```bash
python scripts/build_examples.py --config configs/data.yaml \
  --limit 750 --max-output-examples 750
python scripts/train_sft.py --config configs/sft_stage1.yaml \
  --max-steps 100 --max-train-samples 750
```

Use `--exclude-glint` no download e na construção para remover completamente a
fonte Glint. Isso permite adequar o corpus ao licenciamento pretendido.

## Dados e formato

Cada linha canônica contém IDs de exemplo, sessão e repositório, fonte/revisão,
mensagens, tipo de target, modo de reasoning, indicador de sucesso e score de
qualidade. Os arquivos achatados de treino mantêm `prompt`, `completion`,
`messages` e metadados. IDs são SHA-256 do conteúdo normalizado.

O split é feito por grupos e validado contra interseções de:

- `repository_id`;
- `session_id`;
- `normalized_prompt_hash`.

A normalização reduz as ferramentas a `list_files`, `read_file`, `write_file`,
`apply_patch`, `search` e `shell`. Outputs longos usam truncamento head-tail.

## Treino e retomada

`scripts/train_sft.py` carrega o modelo em NF4, prepara LoRA nos módulos de
atenção e MLP, aplica um collator com labels `-100` fora do conteúdo do
assistente, registra um `run_manifest.json` e procura o checkpoint válido mais
recente quando recebe `--resume-from-checkpoint auto`.

O Stage 1 usa contexto 4096 e LoRA rank 32. O Stage 2 retoma o adapter com
contexto 8192 e learning rate menor. Não execute os dois estágios
simultaneamente nem carregue uma segunda cópia do modelo-base na L4.

## Preferência e harness

A etapa de preferência é opcional. `generate_candidates.py` cria candidatos,
`build_preferences.py` transforma resultados verificáveis em pares e
`train_preference.py` executa DPO com QLoRA.

```bash
python scripts/build_preferences.py \
  --input outputs/evaluations/verified_candidates.jsonl \
  --output data/processed/preferences_train.jsonl \
  --validation-output data/processed/preferences_validation.jsonl
python scripts/train_preference.py --config configs/preference.yaml
```

O harness em `evaluate_agent.py` usa um diretório temporário isolado, timeout,
limite de output, número máximo de passos e bloqueio conservador de comandos
perigosos. Ainda assim, trate código gerado por modelos como não confiável; use
um container/VM para avaliações reais.

## Reprodutibilidade

Cada execução registra configuração, seed, revisão do modelo, hash do dataset,
commit Git, ambiente Python/CUDA/GPU, tokens observados e pico de VRAM. Os
manifests de fonte ficam em `data/manifests/`; manifests de treino acompanham o
diretório de saída.

Consulte [DATA_SOURCES.md](DATA_SOURCES.md) antes de redistribuir dados ou
adapters. Licenças devem ser confirmadas na revisão efetivamente baixada.
