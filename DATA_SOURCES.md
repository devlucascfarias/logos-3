# Fontes e governança dos dados

As revisões de datasets ficam em `main` por padrão. Antes de abrir cada stream,
o pipeline resolve `main` para um SHA imutável e registra tanto a revisão
solicitada quanto a resolvida no relatório. O modelo Qwen3-8B já está fixado por
SHA em `configs/recipe.yaml`. Antes de distribuir um resultado, reconfirme os
termos dessas revisões exatas.

| Categoria | Fonte | Uso | Filtro adicional |
|---|---|---|---|
| Código verificável | `nvidia/OpenCodeReasoning` | Python competitivo | remove benchmarks/holdouts, comprimento e duplicatas |
| Código verificável | `nvidia/Nemotron-Competitive-Programming-v1` | Python competitivo | splits `competitive_coding_python_part00/01` |
| Raciocínio | `open-thoughts/OpenThoughts-114k` | código/STEM | somente amostras estruturadas e dentro de 4096 tokens |
| SWE | `nvidia/Nemotron-SFT-SWE-v2` | traces OpenHands | split curado `openhands_swe` |
| Claude Code | `nlile/misc-merged-claude-code-traces-v1` | traces reais | exige patch/diff e resposta não vazia |
| Geral/ferramentas | `nvidia/Nemotron-Post-Training-Dataset-v1` | tool calling | split `tool_calling`, somente 5% por tokens |

Links dos cards:

- <https://huggingface.co/datasets/nvidia/OpenCodeReasoning>
- <https://huggingface.co/datasets/nvidia/Nemotron-Competitive-Programming-v1>
- <https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k>
- <https://huggingface.co/datasets/nvidia/Nemotron-SFT-SWE-v2>
- <https://huggingface.co/datasets/nlile/misc-merged-claude-code-traces-v1>
- <https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1>

## Regras aplicadas

O preprocessing rejeita:

- tokens de API/credenciais reconhecíveis;
- IDs presentes em `data/eval_holdout_ids.txt`;
- marcadores de HumanEval, MBPP e LiveCodeBench;
- exemplos sem turnos de usuário e assistente;
- repetições consecutivas e excesso de mensagens de ferramenta;
- traces Claude sem patch ou com resposta vazia;
- amostras que ainda excedem 4096 tokens depois da segmentação;
- conteúdo duplicado após normalização de espaços.

E-mails e caminhos de home são redigidos. Outputs de ferramenta são preservados
em head-tail até 12 mil caracteres. O split usa `repository/session/group_id`,
impedindo que o mesmo grupo apareça em treino e validação.

## Responsabilidade de licença

O campo `license` acompanha cada exemplo, mas não substitui uma revisão
jurídica. A fonte Claude agrega múltiplos datasets e é marcada como `mixed`;
confirme os termos de cada origem antes de redistribuir o corpus ou um modelo
derivado. Não publique os arquivos em `data/` sem essa revisão.
