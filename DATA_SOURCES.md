# Fontes de dados e proveniência

Este arquivo é um registro operacional, não aconselhamento jurídico. O comando
`scripts/audit_datasets.py` atualiza manifests com revisão, fingerprint, campos e
card metadata disponíveis no momento do download.

| Fonte | Papel | Peso inicial | Licença | Política |
|---|---:|---:|---|---|
| `greghavens/fable-5-coding-and-debugging-traces` | traces principais | 0,65 | confirmar no card/revisão baixada | habilitada |
| `Glint-Research/Fable-5-traces` | reasoning/próxima ação | 0,20 | confirmar no card; tratar como potencialmente AGPL | opcional por `--exclude-glint` |
| `bigcode/self-oss-instruct-sc2-exec-filter-50k` | retenção de código geral (máximo inicial de 20 mil linhas) | 0,15 | ODC-By-1.0 no card; confirmar na revisão | habilitada |
| `armand0e/claude-fable-5-claude-code` | auditoria/reconstrução | 0 | confirmar no card | nunca misturar automaticamente |

## Campos preservados

Todo exemplo processado deve conter:

```json
{
  "source_dataset": "owner/name",
  "source_revision": "commit-ou-revisao",
  "source_session_id": "id-na-fonte",
  "source_example_id": "id-na-fonte",
  "license": "valor-confirmado-ou-unknown"
}
```

Não presuma que a licença de um agregador substitui a licença da fonte
original. Exemplos sem procedência clara recebem `license: "unknown"` e podem
ser removidos com `require_known_license: true` em `configs/data.yaml`.

A fonte geral foi escolhida por ser um conjunto único, pequeno/controlável e
filtrado por execução. O pipeline limita a leitura inicial a 20 mil linhas para
evitar que ela domine os traces agentic. ODC-By exige atribuição; desabilite a
fonte se isso não for compatível com a distribuição pretendida.
