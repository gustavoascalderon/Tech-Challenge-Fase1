# ML Canvas — Churn Prediction

> Baseado na metodologia ML Canvas (Louis Dorard)

---

## 1. 🎯 Proposta de Valor

**Problema:** Uma operadora de telecomunicações enfrenta alta taxa de cancelamento de clientes (churn), gerando perda de receita recorrente. Ações reativas (após o cancelamento) são custosas e ineficazes.

**Solução ML:** Modelo preditivo que antecipa quais clientes têm alta probabilidade de cancelar nos próximos 30 dias, permitindo intervenções proativas e personalizadas de retenção.

---

## 2. 👥 Stakeholders

| Papel | Interesse |
|-------|-----------|
| Diretoria de Negócios | Reduzir churn rate e aumentar LTV dos clientes |
| Time de CRM/Marketing | Receber lista priorizada de clientes em risco para campanhas |
| Time de Produto | Identificar features do serviço associadas ao cancelamento |
| Engenharia de ML | Construir e manter o modelo em produção |
| Time de Dados | Garantir qualidade e disponibilidade dos dados |

---

## 3. 📥 Inputs (Dados de Entrada)

**Dataset:** Telco Customer Churn — IBM Watson Analytics  
**Volume:** ~7.043 registros, 21 colunas  
**Fonte:** Histórico de clientes da operadora

**Features disponíveis:**

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
- **Falso Negativo** (não identificar um churner): ALTO — cliente cancela sem intervenção
- **Falso Positivo** (classificar não-churner como churner): BAIXO — custo de campanha desnecessária

> **Conclusão:** Priorizar **Recall** (minimizar falsos negativos) sobre Precisão.

---

## 6. 📈 Métricas Técnicas (SLOs)

| Métrica | Threshold Mínimo | Justificativa |
|---------|-----------------|---------------|
| AUC-ROC | ≥ 0.80 | Capacidade discriminativa geral |
| Recall (classe positiva) | ≥ 0.75 | Capturar a maioria dos churners reais |
| F1-Score | ≥ 0.70 | Balanço entre precisão e recall |
| PR-AUC | ≥ 0.65 | Robustez com classes desbalanceadas |
| Latência de inferência (API) | ≤ 200ms | Uso em tempo real no CRM |

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
Dados Brutos → Pré-processamento → Feature Engineering
    → Treinamento (MLP PyTorch) → Avaliação → MLflow Registry
    → FastAPI /predict → CRM da Operadora
```

**Estratégia de deployment:** Online inference via API REST (decisões em tempo real)  
**Retreinamento:** Mensal ou via gatilho de queda de performance (Recall < 0.70)

---

## 9. ⚠️ Riscos e Limitações

| Risco | Impacto | Mitigação |
|-------|---------|-----------|
| Data drift (mudança no perfil dos clientes) | Alto | Monitoramento mensal da distribuição |
| Desbalanceamento de classes (~26% churn) | Médio | SMOTE ou class_weight no treino |
| Viés por perfil demográfico | Alto | Análise de fairness pós-treino |
| Dataset histórico limitado (~7k registros) | Médio | Validação cruzada estratificada |

---

## 10. 📋 Dados de Treinamento

| Item | Detalhe |
|------|---------|
| Período | Histórico consolidado (snapshot único) |
| Tamanho | ~7.043 clientes |
| Balanceamento | ~73% Não-Churn / ~27% Churn |
| Split | 70% treino / 15% validação / 15% teste |
| Estratificação | Sim (manter proporção de churn em todos os splits) |

---

*Última atualização: Junho 2026*
