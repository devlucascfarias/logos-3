# Receita de SFT — Modelo 8B em NVIDIA L4 (Google Colab Pro)

## Decisão principal

**Modelo inicial:** `Qwen/Qwen3-8B`

**Método:** QLoRA 4-bit com Unsloth ou TRL + PEFT.

**GPU-alvo:** NVIDIA L4, 24 GB de VRAM.

O Qwen3-8B é um modelo denso com boa capacidade de raciocínio, código, instruções multilíngues e uso de ferramentas. Para esta primeira etapa, ele é mais apropriado do que tentar adaptar um modelo de 30B na L4.

## Objetivo

Produzir um modelo 8B especializado em:

1. geração e correção de código;
2. raciocínio técnico curto e causal;
3. debugging;
4. uso de ferramentas;
5. tarefas simples de engenharia de software;
6. respostas finais objetivas.

O primeiro treinamento deve validar o pipeline, a formatação, os filtros e as métricas. Não deve tentar consumir integralmente datasets com centenas de milhares de trajetórias.

---

## Mistura de dados

As proporções abaixo são calculadas por **tokens**, não por quantidade de linhas.

| Categoria | Proporção | Dados sugeridos |
|---|---:|---|
| Código executável e verificável | 45% | OpenCodeReasoning, Nemotron Competitive Programming e amostras próprias com testes |
| Raciocínio de código e STEM | 20% | OpenThoughts filtrado, apenas problemas verificáveis |
| Engenharia de software | 20% | Nemotron-SFT-SWE-v2 e trajetórias OpenHands bem-sucedidas |
| Claude Code | 10% | `nlile/misc-merged-claude-code-traces-v1`, fortemente filtrado |
| Instruções gerais e tool calling | 5% | Nemotron Post-Training e exemplos próprios |

### Volume para a primeira execução

**Piloto:** 10–20 milhões de tokens.

**Primeiro treinamento útil:** 30–80 milhões de tokens.

**Após validar o pipeline:** 100–300 milhões de tokens, divididos em múltiplas sessões e checkpoints.

Não é recomendável iniciar com bilhões de tokens no Colab. O custo, a duração das sessões e a interrupção do runtime tornam esse volume impraticável para uma L4.

---

## Critérios de filtragem

### Manter

- código que compila ou executa;
- soluções aprovadas por testes;
- patches que resolvem a tarefa;
- raciocínio que explica causa, evidência e próxima ação;
- sessões com resultado final;
- chamadas de ferramenta válidas;
- exemplos com contexto suficiente;
- diversidade de linguagens e tipos de problema.

### Remover

- trajetórias interrompidas;
- loops de ferramentas;
- dumps de terminal excessivos;
- soluções que não passam nos testes;
- raciocínio repetitivo;
- respostas excessivamente longas para tarefas simples;
- exemplos com proveniência duvidosa;
- conteúdo duplicado;
- benchmarks que serão usados na avaliação;
- segredos, tokens, dados pessoais e credenciais.

---

## Formato recomendado

Cada amostra deve ser convertida para mensagens no padrão do tokenizer do Qwen.

```json
{
  "messages": [
    {
      "role": "system",
      "content": "Você é um assistente especializado em programação. Analise a evidência, produza código correto e valide a solução."
    },
    {
      "role": "user",
      "content": "Corrija a função abaixo e explique brevemente a causa do erro..."
    },
    {
      "role": "assistant",
      "content": "<think>A falha ocorre porque o índice pode ultrapassar o tamanho da lista. Vou validar o limite antes do acesso.</think>\n```python\n...\n```\nA validação evita o acesso fora dos limites."
    }
  ]
}
```

O bloco de raciocínio deve ser curto. A meta é ensinar raciocínio causal, não monólogos.

### Distribuição de comprimento do raciocínio

- 55%: raciocínio curto;
- 25%: raciocínio médio;
- 10%: trajetória agentic mais longa;
- 10%: resposta direta, sem CoT explícito.

---

## Configuração QLoRA recomendada

```yaml
model_name: Qwen/Qwen3-8B
load_in_4bit: true
bnb_4bit_quant_type: nf4
bnb_4bit_use_double_quant: true
bnb_4bit_compute_dtype: bfloat16

max_seq_length: 4096
packing: true

lora_r: 32
lora_alpha: 64
lora_dropout: 0.05
lora_bias: none

target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

per_device_train_batch_size: 1
gradient_accumulation_steps: 16
effective_batch_size: 16

gradient_checkpointing: true
use_reentrant: false

learning_rate: 0.0001
lr_scheduler_type: cosine
warmup_ratio: 0.03
weight_decay: 0.01
max_grad_norm: 1.0

num_train_epochs: 1
optim: paged_adamw_8bit
bf16: true
tf32: true

logging_steps: 5
save_steps: 250
eval_steps: 250
save_total_limit: 3
```

## Ajustes caso falte VRAM

Aplicar nesta ordem:

1. reduzir `max_seq_length` de 4096 para 3072;
2. manter batch físico em 1;
3. ativar gradient checkpointing;
4. reduzir `lora_r` de 32 para 16;
5. desativar avaliação durante o treino;
6. usar atenção otimizada disponível no runtime;
7. evitar tokenização dinâmica durante o treinamento.

Não começar com contexto de 8K ou maior na L4. Traces longos devem ser segmentados semanticamente.

---

## Etapas do projeto

### Etapa 1 — Baseline

Treinar com 10–20 milhões de tokens:

- 60% código verificável;
- 20% raciocínio;
- 10% SWE;
- 5% Claude Code;
- 5% instruções gerais.

Objetivo: verificar se o modelo melhora sem perder estabilidade ou capacidade geral.

### Etapa 2 — SFT principal

Treinar com 30–80 milhões de tokens:

- 45% código verificável;
- 20% raciocínio;
- 20% SWE;
- 10% Claude Code;
- 5% instruções gerais.

Objetivo: consolidar qualidade de código e comportamento de debugging.

### Etapa 3 — Agentic

Continuar a partir do melhor checkpoint, com learning rate entre `2e-5` e `5e-5`.

Usar apenas trajetórias:

- bem-sucedidas;
- curtas;
- sem loops;
- com patch validado;
- com uso consistente de ferramentas.

---

## Avaliação mínima

Avaliar o modelo-base e cada checkpoint com o mesmo conjunto.

### Código

- HumanEval+;
- MBPP+;
- LiveCodeBench ou subconjunto descontaminado;
- testes próprios em Python;
- compilação e testes em C++ quando aplicável.

### Engenharia de software

- tarefas próprias de correção em pequenos repositórios;
- subconjunto de SWE-Bench que não apareceu no treino;
- taxa de patches que passam nos testes;
- número médio de ações até a solução.

### Comportamento

- aderência ao formato;
- tamanho médio do CoT;
- repetição;
- taxa de respostas inválidas;
- uso correto de ferramentas;
- regressão em perguntas gerais.

---

## Critério para escolher o checkpoint

Não selecionar automaticamente o último checkpoint.

O checkpoint escolhido deve maximizar:

```text
score =
  0.40 * código_verificado
+ 0.25 * debugging
+ 0.15 * tarefas_SWE
+ 0.10 * uso_de_ferramentas
+ 0.10 * qualidade_geral
- penalidade_por_repetição
- penalidade_por_overthinking
```

---

## Resultado esperado

A primeira versão não deve tentar competir com modelos de 30B em tarefas de repositório longo. O objetivo realista é obter um 8B que:

- produza código mais correto que o checkpoint original no domínio escolhido;
- explique erros de forma curta;
- siga um protocolo de inspeção, edição e teste;
- use ferramentas sem loops frequentes;
- possa ser posteriormente ampliado para um treinamento em 30B.

## Receita resumida

> `Qwen/Qwen3-8B` + QLoRA NF4 + sequência de 4096 tokens + LoRA rank 32 + 30–80 milhões de tokens filtrados, priorizando código testado e trajetórias SWE bem-sucedidas.
