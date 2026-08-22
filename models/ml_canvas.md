# ML Canvas — Churn Prediction

> Baseado na metodologia ML Canvas (Louis Dorard)  
> Atualizado com resultados reais da Etapa 1

---

## 1. 🎯 Proposta de Valor

**Problema:** Uma operadora de telecomunicações enfrenta alta taxa de cancelamento de clientes (churn), gerando perda de receita recorrente. Ações reativas (após o cancelamento) são custosas e ineficazes.

**Solução ML:** Modelo preditivo que antecipa quais clientes têm alta probabilidade de cancelar, permitindo intervenções proativas e personalizadas de retenção.

---

## 2. 👥 Stakeholders

| Papel | Interesse |
|-------|-----------|
| Diretoria de Negócios | Reduzir churn rate e aumentar LTV dos clientes |
| Time de CRM/Marketing | Lista priorizada de clientes em risco para campanhas |
| Time de Produto | Identificar features do serviço associadas ao cancelamento |
| Engenharia de ML | Construir e manter o modelo em produção |
| Time de Dados | Garantir qualidade e disponibilidade dos dados |

---

## 3. 📥 Inputs (Dados de Entrada)

**Dataset:** Telco Customer Churn — IBM Watson Analytics  
**Fonte:** [Kaggle — blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
**Arquivo:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`

**Estatísticas reais (conforme EDA):**

| Item | Valor |
|------|-------|
| Registros | 7.043 |
| Features | 21 |
| Nulos | 11 (0.16%) — só TotalCharges com tenure=0 |
| Duplicatas | 0 |
| Desbalanceamento | 26.5% churn / 73.5% não-churn |

**Features por grupo:**

| Grupo | Features |
|-------|----------|
| Demográficas | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| Contrato | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod` |
| Serviços | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| Financeiras | `MonthlyCharges`, `TotalCharges` |

**Variável-alvo:** `Churn` (binária: Yes/No → 1/0)

---

## 4. 📤 Outputs (Previsões)

| Output | Tipo | Uso |
|--------|------|-----|
| Probabilidade de churn | Float [0,1] | Score de risco para priorização |
| Classificação binária | 0 ou 1 | Flag de alerta para CRM |

---

## 5. 📊 Métricas de Negócio (KPIs)

| KPI | Definição | Meta |
|-----|-----------|------|
| Churn Rate Evitado | % de churners identificados e retidos | Reduzir em 15% |
| Custo de Retenção | Custo médio de campanha por cliente em risco | < R$ 50/cliente |
| Receita Preservada | Receita mensal salva pela retenção | Maximizar |

**Custo dos erros:**
- **Falso Negativo** (não detectar churner): ALTO — cliente cancela sem intervenção
- **Falso Positivo** (classificar não-churner como churner): BAIXO — custo de campanha desnecessária

> **Conclusão:** Priorizar **Recall** (minimizar falsos negativos) sobre Precisão.

---

## 6. 📈 Métricas Técnicas (SLOs)

| Métrica | Threshold Mínimo | Referência (melhor baseline) | Justificativa |
|---------|-----------------|------------------------------|---------------|
| AUC-ROC | ≥ 0.85 | 0.8517 (LightGBM otimizado) | Capacidade discriminativa geral |
| Recall | ≥ 0.83 | 0.8357 (LightGBM otimizado) | Capturar churners reais |
| F1-Score | ≥ 0.65 | 0.6509 (LightGBM otimizado) | Balanço precisão/recall |
| Latência API | ≤ 200ms | — | Uso em tempo real no CRM |

> A MLP com PyTorch (Etapa 2) precisará superar esses thresholds para ser promovida.

---

## 7. 🔄 Decisões Baseadas nas Previsões

| Cenário | Ação |
|---------|------|
| P(churn) ≥ 0.7 (alto risco) | Contato imediato — oferta de desconto ou upgrade |
| P(churn) entre 0.4 e 0.7 (risco médio) | E-mail de engajamento com benefícios |
| P(churn) < 0.4 (baixo risco) | Nenhuma ação proativa necessária |

---

## 8. 🏗️ Arquitetura de ML

```
Dados Brutos (data/raw/)
    → EDA + Limpeza (01_EDA.ipynb) → data/processed/telco_churn_clean.csv
    → Pipeline sklearn (SimpleImputer + StandardScaler + OneHotEncoder)
    → Feature Selection (SelectKBest, f_classif)
    → Otimização de Hiperparâmetros (Optuna)
    → Treinamento (Regressão Logística / LightGBM / MLP PyTorch)
    → Avaliação + Auditoria Fairlearn
    → MLflow Registry
    → FastAPI /predict → CRM da Operadora
```

**Estratégia de deploy:** Online inference via API REST  
**Retreinamento:** Gatilho de queda de Recall < 0.75 ou drift detectado (KS test)

---

## 9. 🧪 Resultados da Etapa 1 (Baselines)

| Modelo | AUC-ROC | Recall | F1-Score | Papel |
|--------|---------|--------|----------|-------|
| DummyClassifier | ~0.50 | 0.00 | 0.00 | Piso mínimo |
| Regressão Logística | 0.8467 | 0.7964 | 0.6169 | Baseline linear |
| **LightGBM (Optuna)** | **0.8517** | **0.8357** | **0.6509** | Melhor baseline |

**Confusion Matrix — LightGBM (melhor baseline):**

```
              Predito Não-Churn   Predito Churn
Real Não-Churn       572              205
Real Churn            46              234
```

- Falsos Negativos (churners não detectados): **46**
- Falsos Positivos (ações desnecessárias): 205

---

## 10. ⚠️ Riscos e Limitações

| Risco | Impacto | Mitigação |
|-------|---------|-----------|
| Data drift (mudança no perfil dos clientes) | Alto | Monitoramento mensal com KS test |
| Desbalanceamento de classes (26.5% churn) | Médio | class_weight='balanced' + SelectKBest |
| Viés por perfil demográfico | Alto | Auditoria Fairlearn (gender, SeniorCitizen, Partner, Dependents) |
| Dataset limitado (~7k registros) | Médio | Validação cruzada estratificada (5-fold) |
| TotalCharges × tenure: multicolinearidade | Baixo | SelectKBest filtra features redundantes |

---

## 11. 🔧 Stack Técnica

| Componente | Tecnologia | Versão |
|------------|-----------|--------|
| Linguagem | Python | 3.11.1 |
| Dados | pandas, numpy | 2.0.3 / 1.24.1 |
| ML | scikit-learn, LightGBM | ≥1.3 / 4.5.0 |
| Otimização HP | Optuna | ≥3.6 |
| Rede Neural | PyTorch | ≥2.0 (Etapa 2) |
| Tracking | MLflow | ≥2.10 |
| Fairness | Fairlearn | ≥0.10 |
| API | FastAPI + Pydantic | ≥0.110 (Etapa 3) |

---

*Última atualização: Junho 2026 — Etapa 1 concluída*
