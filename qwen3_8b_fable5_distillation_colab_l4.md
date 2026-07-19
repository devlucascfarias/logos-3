# Plano de destilação do Fable 5 para Qwen3-8B Base no Google Colab Pro

## 1. Objetivo

Construir um modelo de até 8 bilhões de parâmetros, baseado em **Qwen3-8B Base**, especializado em comportamento de agente de programação inspirado no Fable 5.

O objetivo não é reproduzir integralmente o Fable 5. Com os traces públicos disponíveis, o resultado esperado é um modelo capaz de aprender principalmente:

- planejamento de tarefas de programação;
- inspeção de repositórios;
- leitura, criação e edição de arquivos;
- uso de terminal e ferramentas;
- ciclos `implementar → compilar/testar → diagnosticar → corrigir`;
- recuperação após erros;
- geração de respostas finais concisas e tecnicamente corretas;
- raciocínio operacional comprimido, sem copiar indiscriminadamente CoTs longos.

O critério principal de sucesso será a capacidade de resolver tarefas executáveis e passar testes, não a similaridade textual com os traces do professor.

---

## 2. Restrições de hardware

Ambiente principal:

- Google Colab Pro;
- GPU NVIDIA L4;
- 24 GB de VRAM;
- armazenamento persistente preferencialmente via Google Drive;
- sessões sujeitas a interrupções;
- RAM de sistema e espaço local variáveis.

Consequências práticas:

1. Não fazer full fine-tuning do Qwen3-8B.
2. Usar **QLoRA em 4 bits**.
3. Usar `bf16` quando suportado pela L4.
4. Ativar gradient checkpointing.
5. Usar Flash Attention 2 ou SDPA.
6. Começar com contexto de 4k ou 8k tokens.
7. Só avançar para 16k em uma fase posterior e com batch unitário.
8. Salvar checkpoints frequentemente no Google Drive.
9. Preparar scripts idempotentes, capazes de retomar do último checkpoint.
10. Evitar preprocessing pesado em tempo de treinamento.

Não assumir que uma única sessão do Colab será suficiente para todo o projeto.

---

## 3. Backbone

Modelo-base:

```text
Qwen/Qwen3-8B-Base
```

Não usar inicialmente a variante Instruct. A versão Base permite controlar melhor o comportamento introduzido pelo pós-treinamento.

Tokenizer:

```text
AutoTokenizer.from_pretrained("Qwen/Qwen3-8B-Base", trust_remote_code=True)
```

Configuração de quantização sugerida:

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
```

---

## 4. Fontes de dados

### 4.1 Dataset principal

```text
greghavens/fable-5-coding-and-debugging-traces
```

Função:

- constituir o núcleo comportamental;
- fornecer trajetórias de engenharia de software;
- priorizar exemplos com verificação externa;
- ensinar ciclos de implementação, build, teste, debugging e correção.

Peso inicial de amostragem:

```text
65%
```

### 4.2 Dataset complementar de reasoning/action

```text
Glint-Research/Fable-5-traces
```

Função:

- fornecer exemplos com campo de raciocínio separado;
- ensinar previsão da próxima ação;
- ensinar seleção de ferramenta;
- fornecer pares `contexto → reasoning → tool call/resposta`.

Peso inicial de amostragem:

```text
20%
```

### 4.3 Dados gerais de código e instruções

Adicionar aproximadamente 15% de dados gerais de programação de alta qualidade e licença compatível.

Função:

- reduzir catastrophic forgetting;
- preservar capacidade geral de código;
- evitar que o modelo memorize um conjunto pequeno de sessões;
- manter diversidade de linguagens e tarefas.

Peso inicial:

```text
15%
```

Selecionar uma fonte pequena e controlada. Não adicionar dezenas de datasets no primeiro experimento.

### 4.4 Dataset que não deve ser combinado diretamente

```text
armand0e/claude-fable-5-claude-code
```

Usar apenas para:

- auditoria;
- reconstrução de sessões;
- inspeção dos logs brutos;
- recuperação de contexto perdido, quando necessário.

Não misturar diretamente com o Glint sem deduplicação por sessão, pois há sobreposição de origem.

---

## 5. Licenciamento e proveniência

Antes de treinar:

1. Registrar a licença de cada fonte.
2. Registrar o commit ou revisão do dataset.
3. Manter um arquivo `DATA_SOURCES.md`.
4. Preservar a coluna de origem em cada exemplo.
5. Não assumir que a licença de um agregador substitui a licença da fonte original.
6. Marcar exemplos cuja procedência seja incerta.
7. Criar uma opção de pipeline que exclua completamente o Glint, caso a licença AGPL seja incompatível com o uso pretendido.

Campos mínimos de proveniência:

```json
{
  "source_dataset": "...",
  "source_revision": "...",
  "source_session_id": "...",
  "source_example_id": "...",
  "license": "..."
}
```

---

## 6. Estrutura do repositório

Criar um repositório com a seguinte estrutura:

```text
fable-qwen-distillation/
├── README.md
├── requirements.txt
├── pyproject.toml
├── configs/
│   ├── data.yaml
│   ├── sft_stage1.yaml
│   ├── sft_stage2.yaml
│   ├── preference.yaml
│   └── eval.yaml
├── notebooks/
│   ├── 00_environment_check.ipynb
│   ├── 01_download_and_audit.ipynb
│   ├── 02_preprocess.ipynb
│   ├── 03_train_qlora.ipynb
│   ├── 04_merge_and_export.ipynb
│   └── 05_evaluate.ipynb
├── scripts/
│   ├── download_datasets.py
│   ├── audit_datasets.py
│   ├── normalize_tools.py
│   ├── build_examples.py
│   ├── deduplicate.py
│   ├── split_by_group.py
│   ├── train_sft.py
│   ├── generate_candidates.py
│   ├── build_preferences.py
│   ├── train_preference.py
│   ├── evaluate_static.py
│   ├── evaluate_agent.py
│   └── export_model.py
├── src/
│   └── fable_distill/
│       ├── __init__.py
│       ├── schemas.py
│       ├── formatting.py
│       ├── filtering.py
│       ├── scoring.py
│       ├── collators.py
│       ├── tools.py
│       └── callbacks.py
├── tests/
│   ├── test_formatting.py
│   ├── test_deduplication.py
│   ├── test_splits.py
│   └── test_tool_normalization.py
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── manifests/
├── outputs/
│   ├── checkpoints/
│   ├── adapters/
│   ├── merged/
│   ├── logs/
│   └── evaluations/
└── DATA_SOURCES.md
```

Scripts devem funcionar tanto em Colab quanto em Linux local.

---

## 7. Dependências

Usar versões fixadas e compatíveis entre si.

Dependências principais:

```text
torch
transformers
accelerate
peft
bitsandbytes
datasets
trl
sentencepiece
safetensors
huggingface_hub
flash-attn, quando a instalação funcionar
wandb, opcional
numpy
pandas
pyyaml
orjson
datasketch
rapidfuzz
pytest
```

Preferir SDPA nativo caso Flash Attention 2 cause problemas de instalação no Colab.

O notebook `00_environment_check.ipynb` deve:

- mostrar GPU;
- mostrar VRAM total e livre;
- verificar suporte a BF16;
- imprimir versões dos pacotes;
- testar carregamento quantizado do modelo;
- testar um forward pass curto;
- montar o Google Drive;
- criar diretórios persistentes;
- registrar o ambiente em JSON.

---

## 8. Esquema canônico de dados

Converter todas as fontes para um formato único.

Exemplo:

```json
{
  "example_id": "sha256...",
  "session_id": "...",
  "repository_id": "...",
  "task_family": "debugging",
  "language": "python",
  "source_dataset": "greghavens/...",
  "messages": [
    {
      "role": "system",
      "content": "You are a software engineering agent operating in a repository. Use tools when necessary and verify changes with tests."
    },
    {
      "role": "user",
      "content": "Implement feature X..."
    },
    {
      "role": "assistant",
      "content": "<plan>Inspect the repository and locate the relevant implementation.</plan>\n<tool_call>{\"name\":\"list_files\",\"arguments\":{\"path\":\".\"}}</tool_call>"
    },
    {
      "role": "tool",
      "name": "list_files",
      "content": "..."
    }
  ],
  "target_type": "tool_call",
  "verified_success": true,
  "quality_score": 0.91
}
```

Manter também um formato achatado para treino:

```json
{
  "prompt": "...",
  "completion": "...",
  "assistant_loss_mask": "...",
  "metadata": "..."
}
```

---

## 9. Normalização das ferramentas

Mapear ferramentas dos datasets para um conjunto pequeno e estável:

```text
list_files
read_file
write_file
apply_patch
search
shell
```

Exemplos de mapeamento:

```text
Bash → shell
Read → read_file
Write → write_file
Edit → apply_patch
Grep → search
Glob → list_files ou search
```

Formato canônico:

```xml
<tool_call>
{"name":"shell","arguments":{"command":"pytest -q"}}
</tool_call>
```

Resultado:

```xml
<tool_result name="shell">
...
</tool_result>
```

Requisitos:

- JSON válido;
- nomes de ferramenta controlados por enum;
- argumentos serializados de forma determinística;
- remoção de campos irrelevantes do runtime;
- limite de tamanho para outputs de ferramentas;
- preservação do início e do fim de logs truncados.

Para logs longos, usar estratégia head-tail:

```text
primeiros N caracteres
<truncated>
últimos N caracteres
```

---

## 10. Limpeza e filtragem

Remover ou corrigir:

- credenciais;
- tokens;
- chaves de API;
- e-mails e dados pessoais;
- caminhos de usuário;
- banners do runtime;
- mensagens de rate limit;
- artefatos ANSI;
- arquivos binários;
- código minificado excessivo;
- dumps gigantes de terminal;
- tentativas duplicadas;
- traces vazios;
- exemplos sem target útil;
- chamadas de ferramenta inválidas;
- raciocínio incompatível com a ação;
- referências indispensáveis a arquivos ausentes;
- passos posteriores à conclusão da tarefa;
- comandos destrutivos não necessários;
- sessões cuja tarefa não pode ser reconstruída.

Não remover automaticamente toda trajetória com falha. Manter falhas que incluam recuperação útil e terminem em solução verificada.

---

## 11. Deduplicação

Executar deduplicação em quatro níveis:

1. Hash exato do exemplo normalizado.
2. Hash do par `prompt + completion`.
3. Similaridade aproximada por MinHash/LSH.
4. Deduplicação por sessão e origem.

Gerar relatório:

```text
número bruto
número após limpeza
número após deduplicação exata
número após deduplicação aproximada
sobreposição entre datasets
```

Não usar split aleatório por linha.

---

## 12. Divisão treino/validação/teste

Separar por grupos, nesta ordem de preferência:

1. repositório;
2. sessão;
3. família de tarefa;
4. prompt normalizado.

Proporção inicial:

```text
train: 90%
validation: 5%
test: 5%
```

O conjunto de teste deve conter repositórios e sessões ausentes do treino.

Criar também um teste externo com tarefas próprias e executáveis.

Aplicar `GroupShuffleSplit` ou implementação equivalente.

Validar programaticamente que não há interseção de:

```text
session_id
repository_id
normalized_prompt_hash
```

entre splits.

---

## 13. Pontuação de qualidade

Calcular uma pontuação por trajetória:

\[
q = 0.35S + 0.20T + 0.15C + 0.15E + 0.15D
\]

Onde:

- `S`: sucesso em testes ou verificação externa;
- `T`: completude da trajetória;
- `C`: consistência entre raciocínio e ação;
- `E`: eficiência, penalizando ciclos inúteis;
- `D`: diversidade em relação ao corpus.

Implementar uma versão heurística inicial.

Exemplos:

```text
S = 1.0 se verified_success=true
S = 0.5 se build passa, mas testes são inconclusivos
S = 0.0 se falha
```

Usar `quality_score` para:

- filtrar a cauda de baixa qualidade;
- ponderar amostragem;
- criar currículos;
- analisar correlação com avaliação.

Não usar score como substituto de inspeção manual.

---

## 14. Tipos de exemplos de treinamento

Criar três tipos principais.

### 14.1 Next-action

Contexto completo até um ponto da trajetória; target é apenas a próxima ação do assistente.

Proporção inicial:

```text
50%
```

### 14.2 Trajectory chunk

Contexto seguido por uma sequência curta de ações e resultados.

Proporção inicial:

```text
35%
```

Não usar trajetórias completas gigantes no início. Cortar em blocos coerentes.

### 14.3 Final-answer

Contexto final da tarefa; target é o resumo final do agente.

Proporção inicial:

```text
15%
```

---

## 15. Tratamento do raciocínio

Não copiar todo CoT literalmente.

Criar três modalidades:

### 15.1 Reasoning detalhado

```xml
<think>
...
</think>
```

Peso aproximado:

```text
30%
```

### 15.2 Plano comprimido

```xml
<plan>
1. Inspect relevant files.
2. Reproduce the failure.
3. Apply a minimal patch.
4. Run focused tests.
</plan>
```

Peso aproximado:

```text
50%
```

### 15.3 Sem reasoning visível

Somente ação ou resposta final.

Peso aproximado:

```text
20%
```

Implementar um campo:

```text
reasoning_mode = long | compressed | hidden
```

Para reasoning comprimido, começar com regras heurísticas. Uma etapa posterior pode usar outro modelo para resumir CoTs, mas essa geração deve ficar separada do pipeline básico e ser cacheada.

Objetivo: ensinar decisão e planejamento, não verbosidade imitativa.

---

## 16. Estratégia de treinamento adequada à L4 de 24 GB

### 16.1 Método

Usar QLoRA 4-bit com PEFT.

Configuração inicial:

```yaml
model_name: Qwen/Qwen3-8B-Base
load_in_4bit: true
bnb_4bit_quant_type: nf4
bnb_4bit_use_double_quant: true
compute_dtype: bfloat16
use_gradient_checkpointing: true
attn_implementation: sdpa
```

### 16.2 LoRA

Configuração inicial conservadora:

```yaml
r: 32
lora_alpha: 64
lora_dropout: 0.05
bias: none
task_type: CAUSAL_LM
```

Módulos-alvo:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

Após validar estabilidade e memória, testar:

```yaml
r: 64
lora_alpha: 128
```

Não começar com rank 128 na L4.

### 16.3 Comprimento de sequência

Currículo recomendado:

```text
fase A: 4096 tokens
fase B: 8192 tokens
fase C opcional: 16384 tokens
```

Não começar em 16k ou 32k.

Na fase de 16k:

- batch por dispositivo igual a 1;
- gradient accumulation elevado;
- exemplos cuidadosamente selecionados;
- somente melhores trajetórias longas;
- monitorar OOM a cada alteração.

### 16.4 Batch

Configuração inicial para 4k:

```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
effective_batch_size: 16
```

Para 8k:

```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 16 ou 32
```

Ajustar pela memória real observada.

### 16.5 Otimizador

```yaml
optim: paged_adamw_8bit
learning_rate: 0.0001
weight_decay: 0.01
warmup_ratio: 0.03
lr_scheduler_type: cosine
max_grad_norm: 1.0
```

Faixa para busca curta:

```text
5e-5
1e-4
1.5e-4
```

Valor inicial recomendado:

```text
1e-4
```

### 16.6 Épocas

```text
1.0 a 2.0 épocas efetivas
```

Começar com 1 época. Avaliar antes de continuar.

Como há exemplos correlacionados de uma mesma trajetória, várias épocas aumentam o risco de memorização.

### 16.7 Packing

Usar packing somente se:

- não misturar sessões dentro de uma mesma sequência de forma semanticamente incorreta;
- a máscara de loss estiver correta;
- os tokens especiais estiverem validados.

Preferir packing por exemplos independentes curtos. Não concatenar fragmentos de sessões diferentes sem separador explícito.

### 16.8 Loss

Aplicar loss somente em tokens do assistente.

Não aplicar loss em:

- system;
- user;
- tool results;
- padding.

Opcionalmente usar pesos diferentes:

```text
tool call tokens: 1.2
reasoning tokens: 0.7
final answer tokens: 1.0
```

A primeira implementação pode usar peso uniforme em todos os tokens do assistente. Só adicionar loss ponderada após validar a pipeline.

---

## 17. Fases do projeto

## Fase 0 — validação mínima

Objetivo: provar que a pipeline funciona.

Usar:

- 500 a 1.000 exemplos;
- sequência 2k ou 4k;
- LoRA rank 16 ou 32;
- 50 a 200 steps;
- sem preferência/RL;
- sem reasoning compression complexo.

Verificar:

- perda diminui;
- checkpoints são salvos;
- retomada funciona;
- geração contém tags válidas;
- tool calls produzem JSON válido;
- não há OOM.

## Fase 1 — SFT principal em 4k

Dados:

- exemplos de maior qualidade;
- next-action;
- trajetórias curtas;
- respostas finais;
- mistura 65/20/15.

Configuração:

```text
QLoRA 4-bit
LoRA r=32 ou 64
seq_len=4096
1 época
```

Salvar adapters e métricas.

## Fase 2 — continuação em 8k

Selecionar exemplos que realmente precisam de contexto maior.

Configuração:

```text
retomar adapter da fase 1
seq_len=8192
learning rate menor, por exemplo 5e-5
0.25 a 0.75 época
```

Não repetir todo o corpus em 8k.

## Fase 3 — contexto longo opcional em 16k

Somente após a fase 2 passar na avaliação.

Usar:

- melhores trajetórias longas;
- batch 1;
- accumulation alto;
- learning rate baixo;
- poucos steps.

Esta fase é opcional devido à L4 de 24 GB.

## Fase 4 — preferência verificável

Só iniciar depois que o SFT produzir tool calls válidos.

Processo:

1. Criar tarefas executáveis próprias.
2. Gerar múltiplos candidatos por tarefa.
3. Executar em sandbox.
4. Registrar build, testes, diff e chamadas inválidas.
5. Construir pares escolhido/rejeitado.
6. Treinar DPO, ORPO ou método equivalente compatível com a VRAM.

Na L4, preferir DPO com QLoRA e sequências menores.

---

## 18. Recompensa para avaliação e preferências

Usar uma recompensa simples:

\[
r = 2I_{tests} + 0.5I_{build} - 0.1N_{invalid} - 0.01L_{patch}
\]

Onde:

- `I_tests`: 1 se testes passam;
- `I_build`: 1 se build passa;
- `N_invalid`: número de chamadas inválidas;
- `L_patch`: medida normalizada do tamanho do patch.

Adicionar penalidades opcionais:

```text
arquivos desnecessários alterados
regressões
comandos destrutivos
loops repetitivos
resposta final que afirma sucesso sem testes
```

Não usar apenas avaliação por LLM.

---

## 19. Avaliação

### 19.1 Métricas estáticas

- validation loss;
- perplexidade apenas como diagnóstico;
- validade do JSON de tool calls;
- acurácia do nome da ferramenta;
- validade do schema de argumentos;
- taxa de fechamento correto das tags;
- exact match em ações simples;
- similaridade semântica para respostas finais.

### 19.2 Métricas de agente

- taxa de build aprovado;
- taxa de testes aprovados;
- taxa de resolução completa;
- número médio de passos;
- chamadas redundantes;
- chamadas inválidas;
- recuperação após erro;
- tamanho do diff;
- regressões;
- arquivos desnecessários alterados.

### 19.3 Capacidade geral

Executar uma pequena suíte para detectar regressão:

- geração de código sem ferramentas;
- explicação de código;
- debugging textual;
- instruction following;
- português e inglês;
- problemas curtos de raciocínio.

### 19.4 Comparações obrigatórias

Comparar:

1. Qwen3-8B Base original.
2. Adapter após fase 1.
3. Adapter após fase 2.
4. Modelo após preferência, se houver.

Usar os mesmos seeds e parâmetros de geração.

---

## 20. Harness de agente

Criar um harness mínimo separado do treinamento.

Ferramentas permitidas:

```text
list_files
read_file
write_file
apply_patch
search
shell
```

Requisitos:

- diretório de trabalho isolado;
- timeout por comando;
- limite de bytes por output;
- bloqueio de comandos perigosos;
- máximo de passos por tarefa;
- registro JSONL de toda execução;
- seed e configuração registrados;
- possibilidade de replay;
- suporte a testes locais.

Nunca executar diretamente comandos gerados pelo modelo no ambiente principal do Colab sem sandbox ou validação.

---

## 21. Checkpoints e persistência no Colab

Salvar no Google Drive:

```text
/MyDrive/fable-qwen-distillation/
```

Estrutura:

```text
checkpoints/stage1/
checkpoints/stage2/
adapters/
processed_data/
logs/
evaluations/
```

Configuração sugerida:

```yaml
save_strategy: steps
save_steps: 100
save_total_limit: 3
logging_steps: 5
eval_strategy: steps
eval_steps: 100
```

Para experimentos longos, considerar `save_steps=50`.

Cada checkpoint deve incluir:

- adapter;
- optimizer state;
- scheduler state;
- trainer state;
- config completa;
- hash do dataset processado;
- revisão do modelo-base;
- versões dos pacotes.

Implementar retomada automática:

```text
--resume_from_checkpoint auto
```

O script deve localizar o checkpoint mais recente válido.

---

## 22. Controle de memória

Antes do treino, imprimir:

```python
torch.cuda.get_device_name(0)
torch.cuda.get_device_properties(0).total_memory
torch.cuda.mem_get_info()
```

Durante o treino:

- registrar pico de VRAM;
- limpar cache entre avaliação e treino quando necessário;
- evitar manter modelo-base duplicado;
- não carregar modelo de referência simultaneamente em DPO sem quantização;
- usar gradient checkpointing;
- usar `use_cache=False` durante treino;
- usar tokenizer offline após download;
- processar datasets em disco.

Ordem para resolver OOM:

1. reduzir `max_seq_length`;
2. manter batch igual a 1;
3. aumentar gradient accumulation;
4. reduzir LoRA rank;
5. desativar packing;
6. usar SDPA;
7. reduzir tamanho da avaliação;
8. reduzir número de workers;
9. offload apenas como último recurso.

Não usar CPU offload como configuração padrão, pois pode tornar o treino inviavelmente lento no Colab.

---

## 23. Reprodutibilidade

Fixar seeds:

```text
python
numpy
torch
datasets
sampler
```

Registrar:

- seed;
- versão dos dados;
- revisão do modelo;
- hiperparâmetros;
- commit do código;
- GPU;
- versões CUDA e PyTorch;
- tempo por step;
- pico de VRAM;
- número real de tokens vistos.

Salvar um `run_manifest.json` por execução.

---

## 24. Testes automatizados

Implementar testes para:

1. parsing de cada dataset;
2. normalização de ferramentas;
3. JSON de tool calls;
4. remoção de PII;
5. deduplicação;
6. ausência de vazamento entre splits;
7. máscaras de loss;
8. formatação do chat template;
9. truncamento head-tail;
10. retomada de checkpoint;
11. geração curta com adapter;
12. exportação e recarregamento.

Executar testes antes de iniciar uma sessão longa de treino.

---

## 25. Entregáveis do Codex

O Codex deve produzir:

1. Repositório completo com a estrutura especificada.
2. `README.md` com instruções locais e para Colab.
3. Notebooks executáveis em ordem.
4. Scripts CLI equivalentes aos notebooks.
5. Arquivos de configuração YAML.
6. Pipeline de download e auditoria.
7. Pipeline de preprocessing e deduplicação.
8. Split seguro por sessão/repositório.
9. Treino QLoRA com retomada.
10. Avaliação estática.
11. Harness mínimo de agente.
12. Exportação do adapter.
13. Script opcional para merge do adapter em CPU ou ambiente com RAM suficiente.
14. Relatório de VRAM e velocidade.
15. Testes automatizados.
16. Manifestos de dados e execuções.

---

## 26. Critérios de aceitação

A fase inicial será aceita quando:

- o Qwen3-8B Base carregar em 4-bit na L4;
- um treino curto completar sem OOM;
- checkpoint e retomada funcionarem;
- o adapter puder ser recarregado;
- o modelo gerar tool calls no schema correto;
- o pipeline impedir vazamento por sessão;
- os datasets forem rastreáveis por origem;
- a loss for aplicada apenas aos tokens corretos;
- a avaliação comparar base e modelo treinado;
- o repositório puder ser executado novamente do zero.

A fase principal será aceita quando:

- houver ganho sobre o modelo-base em tarefas agentic;
- a taxa de tool calls inválidos cair;
- a taxa de testes aprovados subir;
- o modelo não apresentar regressão generalizada grave;
- resultados forem reproduzíveis a partir dos manifests.

---

## 27. Configuração inicial recomendada

```yaml
model:
  name: Qwen/Qwen3-8B-Base
  load_in_4bit: true
  quant_type: nf4
  double_quant: true
  compute_dtype: bfloat16
  attn_implementation: sdpa
  trust_remote_code: true

lora:
  r: 32
  alpha: 64
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

training:
  max_seq_length: 4096
  per_device_train_batch_size: 1
  per_device_eval_batch_size: 1
  gradient_accumulation_steps: 16
  learning_rate: 1.0e-4
  num_train_epochs: 1.0
  warmup_ratio: 0.03
  weight_decay: 0.01
  max_grad_norm: 1.0
  optimizer: paged_adamw_8bit
  scheduler: cosine
  bf16: true
  fp16: false
  gradient_checkpointing: true
  use_cache: false
  logging_steps: 5
  eval_steps: 100
  save_steps: 100
  save_total_limit: 3
  seed: 42

sampling:
  greghavens_weight: 0.65
  glint_weight: 0.20
  general_code_weight: 0.15
  next_action_weight: 0.50
  trajectory_weight: 0.35
  final_answer_weight: 0.15
  reasoning_long_weight: 0.30
  reasoning_compressed_weight: 0.50
  reasoning_hidden_weight: 0.20
```

---

## 28. Ordem de implementação para o Codex

Executar nesta ordem:

1. Criar estrutura do repositório.
2. Criar ambiente e requirements fixados.
3. Implementar schemas.
4. Implementar download dos datasets.
5. Implementar auditoria e manifests.
6. Implementar parsers por fonte.
7. Implementar normalização das ferramentas.
8. Implementar limpeza e redaction.
9. Implementar deduplicação.
10. Implementar split por grupos.
11. Implementar formatação para Qwen.
12. Implementar máscaras de loss.
13. Criar testes unitários.
14. Criar smoke test com 500 exemplos.
15. Implementar treino QLoRA.
16. Implementar retomada automática.
17. Implementar avaliação estática.
18. Implementar harness de agente.
19. Executar fase 1 em 4k.
20. Avaliar contra o modelo-base.
21. Só então criar fase 2 em 8k.
22. Implementar preferência apenas depois do SFT estável.

Não começar pela fase de RL ou DPO.

---

## 29. Observações finais

- O corpus público é pequeno para uma destilação completa de um modelo 8B.
- O projeto deve ser tratado como pós-treinamento especializado em coding agents.
- A qualidade dos traces importa mais do que o volume bruto.
- Avaliação por execução importa mais do que imitação textual de CoT.
- O melhor ganho adicional viria da geração de novas trajetórias verificáveis pelo professor, caso haja acesso ao Fable 5.
- Na L4 de 24 GB, contexto e rank de LoRA devem ser aumentados somente após medição real de VRAM.
- Um modelo menor com pipeline correto pode superar um treinamento maior mal filtrado.
- Toda decisão de expansão deve ser baseada em ablação e avaliação, não apenas em training loss.
