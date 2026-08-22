# Plano de Monitoramento — Churn Guard MLP

Versão 1.0 · Owner: time de Advanced Analytics · Revisão: trimestral

O modelo não termina no deploy. Este documento define o que é observado, com que frequência, qual limiar dispara ação e quem age.

---

## 1. Por que monitorar

Três coisas quebram um modelo de churn em produção, e nenhuma delas aparece na métrica de teste:

1. **Data drift** — o perfil de quem entra na base muda (novo plano, nova campanha de aquisição).
2. **Concept drift** — a relação entre features e churn muda (concorrente lança oferta agressiva; `MonthlyCharges` alto deixa de ser sinal de risco).
3. **Degradação silenciosa de infra** — schema de origem muda, um campo vira nulo, a latência sobe. O modelo continua respondendo 200, com lixo.

O rótulo real só chega com ~30 dias de atraso (churn é observado no fim do ciclo de faturamento). Por isso o monitoramento tem duas camadas: **proxies imediatos** e **métricas de performance defasadas**.

---

## 2. Camada 1 — Saúde operacional (tempo real)

| Métrica | Como medir | Verde | Alerta | Crítico | Ação |
|---|---|---|---|---|---|
| Disponibilidade | `GET /health` a cada 30s | `status=ok` | 1 falha | 3 falhas seguidas | Página o on-call; rollback |
| Latência p95 | Campo `latency_ms` da resposta | < 100ms | 100–200ms | > 200ms | Investigar; escalar réplicas |
| Taxa de erro 5xx | Logs de acesso, janela de 5 min | < 0.1% | 0.1–1% | > 1% | Página o on-call |
| Taxa de 422 | Logs de acesso, janela de 1h | < 1% | 1–5% | > 5% | Contrato de entrada mudou — falar com o time consumidor |
| Modelo carregado | `model_loaded` em `/health` | `true` | — | `false` | Container não deve receber tráfego |

O 422 merece atenção: um pico significa que a origem começou a mandar categoria fora do vocabulário de treino. Isso é drift chegando pela porta da frente, e a validação Pydantic o torna visível em vez de deixar o modelo extrapolar em silêncio.

---

## 3. Camada 2 — Data drift (diário)

Comparação da janela dos últimos 7 dias contra a distribuição de referência (conjunto de treino, congelado em `data/processed/`).

| Feature | Teste | Alerta | Crítico |
|---|---|---|---|
| `tenure`, `MonthlyCharges` | Kolmogorov–Smirnov | p < 0.05 | p < 0.01 **e** PSI > 0.2 |
| Categóricas (15) | Population Stability Index | PSI 0.1–0.2 | PSI > 0.25 |
| Todas | % de nulos | > 2× baseline | > 10% absoluto |
| Categóricas | Categoria nova (fora do vocabulário) | qualquer ocorrência | > 1% das requisições |

**Interpretação do PSI.** < 0.1 estável · 0.1–0.25 mudança moderada, investigar · > 0.25 mudança significativa, retreinar.

**Prediction drift.** Distribuição das probabilidades de saída, comparada com a de referência:

| Métrica | Alerta | Crítico |
|---|---|---|
| Média das probabilidades | desvio > 5 p.p. | > 10 p.p. |
| % classificado como risco alto (≥0.70) | desvio > 5 p.p. | > 10 p.p. |
| Divergência de Jensen–Shannon vs. referência | > 0.1 | > 0.2 |

Prediction drift é o sinal mais rápido disponível, porque não depende do rótulo. Se a fração de risco alto salta de 18% para 31% em uma semana sem campanha nova, algo mudou na entrada.

---

## 4. Camada 3 — Performance (mensal, com defasagem de rótulo)

Depois que o ciclo de faturamento fecha e o churn real é observado:

| Métrica | Baseline (teste) | Alerta | Crítico | Ação |
|---|---|---|---|---|
| AUC-ROC | ≥ 0.85 | queda > 3 p.p. | < 0.80 | Retreino |
| Recall | ≥ 0.83 | queda > 5 p.p. | < 0.75 | Retreino imediato |
| F1 | ≥ 0.65 | queda > 5 p.p. | < 0.58 | Retreino |
| Precision | — | queda > 10 p.p. | — | Reavaliar threshold antes de retreinar |

Queda só de precisão com recall estável normalmente é threshold desatualizado, não modelo degradado. Recalibre o threshold na validação antes de gastar um ciclo de retreino.

---

## 5. Camada 4 — Fairness (mensal)

Rodada junto com a avaliação de performance, sobre os mesmos dados rotulados.

| Métrica | Limiar | Ação |
|---|---|---|
| Disparidade de FNR entre grupos (`gender`, `SeniorCitizen`, `Partner`, `Dependents`) | ≤ 10 p.p. | Acima: bloqueia promoção; abre revisão |
| Disparidade de recall entre grupos | ≤ 10 p.p. | Acima: investigar representatividade |
| Volume mínimo por grupo | ≥ 100 observações | Abaixo: reportar sem conclusão (IC largo demais) |

Um gate de fairness violado **bloqueia a promoção do modelo**, mesmo que todas as métricas de performance passem. Modelo que performa bem no agregado e mal para um subgrupo não está pronto.

---

## 6. Camada 5 — Impacto de negócio (mensal)

Métricas técnicas boas com impacto zero significam que o modelo não está sendo usado, ou está sendo usado errado.

| Indicador | Como medir |
|---|---|
| Taxa de retenção na faixa alta | % de contatados que não cancelaram em 60 dias |
| Lift vs. controle | Retenção do grupo tratado − retenção do grupo de controle |
| Custo por retenção | Custo total das ofertas ÷ clientes retidos |
| Cobertura | % dos churns reais do mês que estavam na faixa alta |

**Grupo de controle.** Manter 5–10% dos clientes de risco alto sem contato ativo. Sem isso não há como separar o efeito do modelo do efeito da campanha, e a métrica de negócio vira narrativa.

---

## 7. Gatilhos de retreino

| Gatilho | Tipo | Prazo |
|---|---|---|
| Calendário trimestral | Programado | — |
| Recall < 0.75 | Crítico | 48h |
| AUC-ROC < 0.80 | Crítico | 1 semana |
| PSI > 0.25 em qualquer feature | Crítico | 1 semana |
| Disparidade de FNR > 10 p.p. | Crítico | 1 semana |
| Categoria nova em > 1% das requisições | Crítico | Imediato (schema mudou) |
| PSI 0.1–0.25 sustentado por 2 semanas | Alerta | Próximo ciclo |
| Mudança de portfólio/preço | Manual | Antecipar o ciclo |

### Procedimento de retreino

1. Congelar a janela de dados e versionar (`dataset_version` incrementa).
2. Rodar `python -m churn.models.baselines` — o modelo novo precisa continuar superando os baselines na janela nova.
3. Rodar `python -m churn.models.train_mlp` com o mesmo `SEED`.
4. Verificar `quality_gate` **e** `fairness_gate`. Qualquer um reprovado bloqueia a promoção.
5. Comparar contra o modelo em produção no MLflow.
6. Promover via Model Registry; manter a versão anterior para rollback.
7. Atualizar `docs/model_card.md` com as novas métricas e a data.

---

## 8. Instrumentação — o que ainda falta

Estado atual: métricas de latência e versão do modelo já saem na resposta da API; MLflow guarda o histórico de runs. Para fechar o plano falta:

- [ ] Log estruturado (JSON) de cada requisição: `request_id`, features, probabilidade, versão do modelo, timestamp
- [ ] Job diário de PSI/KS contra a referência congelada
- [ ] Endpoint `/metrics` no formato Prometheus
- [ ] Dashboard (Grafana ou Evidently) com as quatro camadas
- [ ] Join mensal automatizado entre predições e churn observado
- [ ] Alertas roteados para o canal do time

> Nenhuma dessas peças exige mudança no modelo — são todas de plataforma. A ordem acima é a de maior retorno por esforço.

**Privacidade.** O log de requisições contém dados de cliente. Retenção de 90 dias, acesso restrito ao time owner, `customerID` pseudonimizado no armazenamento analítico.
