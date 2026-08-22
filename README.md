# 📡 Churn Guard —  Retenção de Clientes

Tech Challenge Fase 1 · FIAP/POSTECH — Machine Learning Engineering

Pipeline end-to-end de predição de churn em telecom. Modelo central: **MLP em PyTorch com early stopping**, comparado contra três baselines (Dummy, Regressão Logística e LightGBM otimizado por Optuna), rastreado no MLflow, auditado quanto a fairness e servido por uma **API FastAPI**.

---

## 🎯 Problema de negócio

Uma operadora perde ~26,5% dos clientes. O time de retenção tem capacidade limitada e trabalha a base sem priorização. O modelo pontua cada cliente e ordena a fila de contato: quem está mais perto de cancelar aparece primeiro.

A decisão apoiada é **quem contatar primeiro**, não a quem negar serviço. O score nunca restringe ou encarece o produto.

---

## 📐 Métrica primária: Recall

Falso negativo = cliente cancela sem nunca ter recebido oferta. Perde-se o LTV inteiro.
Falso positivo = desconto oferecido a quem ficaria. Custa o valor do desconto.

Como o LTV supera o desconto em uma ordem de grandeza, o modelo é calibrado para recall alto mesmo ao custo de precisão. O threshold de decisão **não é 0.5** — é buscado no conjunto de validação maximizando F1 sujeito a `recall ≥ 0.83`. O valor selecionado foi **0.43**.

---

## 📊 Resultados

Conjunto de teste: 1.057 clientes, nunca vistos em treino nem na escolha do threshold.

| Modelo | AUC-ROC | Recall | F1 | Precision |
|---|---|---|---|---|
| DummyClassifier | 0.5000 | 0.0000 | 0.0000 | 0.0000 |
| Regressão Logística | 0.8467 | 0.7964 | 0.6169 | 0.5034 |
| LightGBM (Optuna) | **0.8525** | 0.8107 | **0.6421** | 0.5316 |
| **MLP (PyTorch)** | 0.8444 | **0.8714** | 0.6070 | 0.4656 |

A MLP lidera no **recall**, a métrica primária declarada — captura 87,1% dos clientes que cancelam, contra 81,1% do LightGBM. O custo é precisão menor (0.4656 vs 0.5316), consequência direta do threshold de 0.43. A diferença de AUC (0.8444 vs 0.8525) está dentro do ruído esperado para um teste desse tamanho.

**Metas de promoção:** AUC-ROC ≥ 0.85 · Recall ≥ 0.83 · F1 ≥ 0.65 · disparidade de FNR ≤ 10 p.p. · latência p95 ≤ 200ms.

**Resultado:** recall aprovado (0.8714); latência aprovada (~7ms); AUC (0.8444) e F1 (0.6070) abaixo da meta; fairness reprovado em `SeniorCitizen` e `Dependents` — análise completa na [seção 5 do Model Card](docs/model_card.md).

---

## 🧠 Arquitetura do modelo

```
44 features (após one-hot)
  ↓  Linear(44→128) → BatchNorm → ReLU → Dropout(0.3)
  ↓  Linear(128→64) → BatchNorm → ReLU → Dropout(0.3)
  ↓  Linear(64→32)  → BatchNorm → ReLU → Dropout(0.3)
  ↓  Linear(32→1)   → logit
```

16.577 parâmetros treináveis.

| | |
|---|---|
| Loss | `BCEWithLogitsLoss` com `pos_weight = 2.768` (trata o desbalanceamento) |
| Otimizador | AdamW (`lr=1e-3`, `weight_decay=1e-4`) |
| Scheduler | `ReduceLROnPlateau` sobre `val_auc` |
| **Early stopping** | `patience=15`, `min_delta=1e-4`, monitorando `val_auc`, com restauração dos pesos da melhor época |
| Clipping | `max_norm=5.0` |

No treino registrado, o early stopping disparou na **época 36** e restaurou os pesos da **época 21** (melhor `val_auc` = 0.8299). O modelo servido é o da melhor época de validação, **nunca o da última**. Implementação em `src/churn/models/mlp.py`.

Execução determinística: três rodadas consecutivas produziram métricas idênticas até a sexta casa decimal.

---

## 🗂️ Estrutura do projeto

```
tech-challenge-fase1/
├── src/churn/
│   ├── config.py                  # caminhos relativos, seeds, gates, schema
│   ├── data/
│   │   ├── load.py                # ingestão e limpeza
│   │   └── validate.py            # contrato de dados (pandera)
│   ├── features/
│   │   └── preprocess.py          # split estratificado + ColumnTransformer
│   ├── models/
│   │   ├── mlp.py                 # ChurnMLP + EarlyStopping + persistência
│   │   ├── train_mlp.py           # loop de treino + MLflow
│   │   ├── baselines.py           # Dummy + LR + LightGBM/Optuna
│   │   ├── evaluate.py            # métricas, threshold, fairness, gates
│   │   └── fairness_audit.py      # auditoria de disparidade por grupo
│   └── api/
│       ├── main.py                # FastAPI
│       └── schemas.py             # contratos Pydantic
├── tests/                         # 58 testes (dados, modelo, treino, API)
├── docs/
│   ├── model_card.md              # Model Card
│   ├── monitoring_plan.md         # plano de monitoramento
│   ├── ml_canvas.md               # ML Canvas
│   └── fairness_report.csv        # auditoria por grupo
├── notebooks/
│   ├── 01_EDA.ipynb               # análise exploratória
│   └── 02_baselines.ipynb         # baselines (exploratório)
├── data/{raw,processed}/          # não versionado
├── models/                        # artefatos servidos (não versionado)
├── mlruns/                        # tracking MLflow local (não versionado)
├── .github/workflows/ci.yml
└── pyproject.toml
```

---

## ⚙️ Setup

Requer Python 3.11+.

```bash
git clone https://github.com/<seu-usuario>/tech-challenge-fase1.git
cd tech-challenge-fase1

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

### Dataset

Baixe em [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) e coloque em:

```
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Nenhum caminho absoluto é usado no código — tudo deriva da raiz do repositório em `src/churn/config.py`, e pode ser sobrescrito por `CHURN_DATA_DIR` e `CHURN_MODELS_DIR`.

---

## 🚀 Execução

```bash
# 1. Prepara o dataset  →  data/processed/telco_churn_clean.csv
python -m churn.data.load

# 2. Baselines de referência  (Dummy, LR, LightGBM+Optuna)
python -m churn.models.baselines --trials 30

# 3. Treina a MLP  →  models/{preprocessor.joblib, mlp_churn.pt, mlp_metadata.json}
python -m churn.models.train_mlp

# 4. Auditoria de fairness
python -m churn.models.fairness_audit
python -m churn.models.fairness_audit --markdown   # tabelas para o Model Card

# variações de arquitetura
python -m churn.models.train_mlp --hidden 256 128 64 --patience 25 --max-epochs 300
python -m churn.models.train_mlp --no-mlflow       # sem tracking
```

O treino imprime a época em que o early stopping disparou, o threshold escolhido, o resultado dos gates de qualidade e o relatório de fairness.

### MLflow

```bash
export MLFLOW_ALLOW_FILE_STORE=true
python -m mlflow ui --backend-store-uri file:./mlruns
# http://localhost:5000
```

### API

```bash
uvicorn churn.api.main:app --reload --port 8000
# Swagger: http://localhost:8000/docs
```

| Endpoint | Descrição |
|---|---|
| `GET /health` | Liveness — reporta `degraded` se os artefatos não carregaram |
| `GET /model-info` | Arquitetura, threshold, métricas e versão do modelo |
| `POST /predict` | Predição individual |
| `POST /predict/batch` | Até 1000 clientes em uma passada |

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure":3,"MonthlyCharges":89.9,"SeniorCitizen":0,"gender":"Female",
       "Partner":"No","Dependents":"No","PhoneService":"Yes","MultipleLines":"No",
       "InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No",
       "DeviceProtection":"No","TechSupport":"No","StreamingTV":"Yes",
       "StreamingMovies":"Yes","Contract":"Month-to-month","PaperlessBilling":"Yes",
       "PaymentMethod":"Electronic check"}'
```

```json
{
  "churn_probability": 0.8412,
  "churn_prediction": 1,
  "risk_band": "alto",
  "threshold": 0.43,
  "model_version": "v1.0",
  "latency_ms": 7.31
}
```

Categoria fora do vocabulário de treino retorna **422**, não uma predição silenciosa. Isso é intencional: um pico de 422 é drift de schema chegando pela porta da frente.

### Testes e linting

```bash
pytest                      # 58 testes
ruff check src/ tests/
```

---

## 🔬 Decisões técnicas

**`TotalCharges` removida.** Correlação de ~0,83 com `tenure` (é essencialmente `tenure × MonthlyCharges`) e vem quebrada no CSV — clientes com `tenure=0` têm string vazia. Manter as duas adiciona colinearidade sem ganho.

**Reponderação em vez de SMOTE.** Oversampling sintético em espaço one-hot gera combinações categóricas que não existem no mundo real (cliente com `InternetService=No` e `StreamingTV=Yes`). `pos_weight` na loss resolve o desbalanceamento sem inventar dado.

**Threshold buscado na validação.** O conjunto de teste é tocado uma única vez, no fim. Escolher threshold no teste vaza informação e infla a métrica reportada.

**Só o `state_dict` é salvo.** Nunca o objeto pickled inteiro — o carregamento não deve depender da versão exata do código que treinou.

**Fairness por FNR, não por paridade demográfica.** Falso negativo = cliente em risco que não recebeu ação. Disparidade de FNR mede sub-atendimento. Igualar taxas de seleção entre grupos com risco genuinamente diferente prejudicaria o grupo de maior risco.

---

## 📚 Documentação

- [`docs/model_card.md`](docs/model_card.md) — usos pretendidos e fora de escopo, dados, treino, métricas, fairness, limitações, manutenção
- [`docs/monitoring_plan.md`](docs/monitoring_plan.md) — cinco camadas de monitoramento, limiares de drift, gatilhos de retreino
- [`docs/ml_canvas.md`](docs/ml_canvas.md) — ML Canvas

---

## ⚠️ Limitações conhecidas

- 7.043 registros é pouco para uma rede neural. A vantagem da MLP sobre gradient boosting em tabular desse porte é marginal — a escolha é pedagógica e assumida explicitamente.
- Dataset estático, sem componente temporal: nenhuma sazonalidade é capturada.
- Sem features comportamentais (uso de dados, chamadas ao SAC, reclamações), que na prática são os sinais mais fortes de churn.
- `gender` é binário no dataset; clientes não-binários não são representados.
- Gate de fairness reprovado em `SeniorCitizen` e `Dependents` — documentado e analisado, não mitigado. Ver Model Card seção 5.
- Correlação, não causalidade. `Contract=Month-to-month` prediz churn; forçar migração de contrato não necessariamente retém.

---

## 👥 Time

| Nome | RM |
|---|---|
| Gustavo Andres Silva Calderon | rm374865 |
| Lilian Dorini Almagro | rm374368 |
| Matheus Angelo Filetti | rm373015 |


---

## 📖 Referências

- [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- [PyTorch](https://pytorch.org/docs/stable/index.html) · [MLflow](https://mlflow.org/docs/latest/) · [Fairlearn](https://fairlearn.org/) · [Optuna](https://optuna.readthedocs.io/) · [LightGBM](https://lightgbm.readthedocs.io/) · [FastAPI](https://fastapi.tiangolo.com/)
- Mitchell et al. (2019), *Model Cards for Model Reporting*

---

## 📄 Licença

Uso acadêmico — POSTECH / FIAP, Machine Learning Engineering.
