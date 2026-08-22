# Model Card — Churn Prediction (Fase 1)

> Documento de transparência do modelo seguindo o padrão Mitchell et al. (2019).

---

## 1. Detalhes do Modelo

| Campo | Valor |
|-------|-------|
| Nome | churn-lgbm-optimized v1.0 |
| Tipo | LightGBM Classifier (Gradient Boosting) |
| Otimização | Optuna (hiperparâmetros + SelectKBest k) |
| Versão do dataset | v1.0 |
| Data de treinamento | Junho 2026 |
| Desenvolvido por | Matheus Angelo Filetti (rm373015) |
| Contexto | Tech Challenge Fase 1 — POSTECH/FIAP |

---

## 2. Uso Pretendido

**Uso primário:** Auxiliar equipes de CRM na priorização de campanhas de retenção, identificando clientes com alta probabilidade de cancelamento (churn).

**Usuários pretendidos:** Analistas de CRM, times de retenção, gestores comerciais.

**Uso fora do escopo:**
- Não deve ser usado como única base para decisões de cancelamento de contratos.
- Não foi validado para perfis de clientes fora do segmento de telecomunicações.
- Não deve ser aplicado em contextos onde viés por grupos demográficos seja inaceitável sem mitigação.

---

## 3. Fatores

**Fatores relevantes avaliados:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`

**Fatores de desempenho:** O modelo tende a performar melhor em clientes com `Contract = Month-to-month` e `InternetService = Fiber optic`, que são os grupos com maior concentração de churn no dataset.

---

## 4. Métricas de Desempenho

### Resultados no Conjunto de Teste (15% dos dados, estratificado)

| Modelo | AUC-ROC | Recall | F1-Score | Precision | Accuracy |
|--------|---------|--------|----------|-----------|----------|
| DummyClassifier (piso) | ~0.50 | 0.00 | 0.00 | — | ~0.735 |
| Regressão Logística | 0.8467 | 0.7964 | 0.6169 | ~0.51 | ~0.74 |
| **LightGBM (Optuna)** | **0.8517** | **0.8357** | **0.6509** | ~0.53 | ~0.76 |

### Confusion Matrix — LightGBM (melhor modelo)

```
                  Predito: Não Churn   Predito: Churn
Real: Não Churn        572               205
Real: Churn             46               234
```

- **Falsos Negativos (FN = 46):** churners não detectados — clientes que cancelam sem receber intervenção.
- **Falsos Positivos (FP = 205):** não-churners classificados como risco — custo de campanhas desnecessárias.

**Métrica prioritária:** Recall — minimizar Falsos Negativos é mais importante do que evitar Falsos Positivos neste contexto de negócio.

---

## 5. Dados de Treinamento

| Item | Detalhe |
|------|---------|
| Dataset | Telco Customer Churn — IBM Watson Analytics |
| Fonte | [Kaggle — blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| Volume | 7.043 clientes × 21 features |
| Período | Snapshot histórico (data não informada pelo provedor) |
| Balanceamento | 73.5% Não Churn / 26.5% Churn |
| Split | 70% treino / 15% validação / 15% teste (estratificado, SEED=42) |
| Features excluídas | `customerID` (identificador), `TotalCharges` (colinear com tenure) |
| Pré-processamento | SimpleImputer (median/mode) + StandardScaler + OneHotEncoder + SelectKBest |

---

## 6. Considerações Éticas e Auditoria de Fairness

### Atributos sensíveis auditados (Fairlearn — MetricFrame)

| Atributo | Métrica | Disparidade (FNR) | Status |
|----------|---------|-------------------|--------|
| `gender` | Taxa de Falso Negativo | A verificar na execução | Limiar: ≤ 10 p.p. |
| `SeniorCitizen` | Taxa de Falso Negativo | A verificar na execução | Limiar: ≤ 10 p.p. |
| `Partner` | Taxa de Falso Negativo | A verificar na execução | Limiar: ≤ 10 p.p. |
| `Dependents` | Taxa de Falso Negativo | A verificar na execução | Limiar: ≤ 10 p.p. |

> Executar `notebooks/03_fairness.ipynb` para obter os valores exatos de disparidade. O gate de fairness rejeita o modelo se qualquer atributo ultrapassar 10 pontos percentuais de diferença de FNR entre grupos.

**Possíveis fontes de viés:**
- O dataset não inclui dados de renda, que pode ser correlacionado com `Contract` e `PaymentMethod`.
- `SeniorCitizen` é um proxy de idade (não há coluna de idade contínua).
- O histórico é de uma única operadora — não generaliza para outras regiões ou perfis.

---

## 7. Limitações Conhecidas

- **Dados limitados:** ~7k registros é pequeno para modelos de produção em larga escala.
- **Snapshot único:** o modelo não foi treinado com dados temporais — não captura sazonalidade.
- **Sem dados de comportamento:** não há features de interação (chamadas ao suporte, navegação no app).
- **Não validado para uso pediátrico** ou perfis fora do mercado de telecomunicações.
- **Latência não medida em produção:** a API ainda não foi deployada (Etapa 3).

---

## 8. Plano de Monitoramento

| Sinal | Métrica | Frequência | Ação |
|-------|---------|-----------|------|
| Data Drift | KS test nas features de entrada | Mensal | Alerta se p < 0.05 |
| Model Drift | Recall no conjunto rotulado | Mensal | Retreinar se Recall < 0.75 |
| Fairness Drift | Disparidade FNR por grupo | Trimestral | Auditoria + possível retreino |
| Latência API | P99 < 200ms | Contínuo | Alerta se ultrapassar |

---

## 9. Informações de Reprodutibilidade

- **SEED:** 42 (fixado em todas as etapas)
- **Validação cruzada:** StratifiedKFold (5-fold)
- **Registro MLflow:** experimento `churn-prediction-fase1`, run `optimized-lightgbm`
- **Instalação:** `pip install -e ".[dev]"` a partir do `pyproject.toml`

---

*Última atualização: Junho 2026 — Etapa 1 concluída*
