# 🛡️ Churn Guard: Inteligência Artificial para Retenção de Clientes

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?logo=mlflow&logoColor=white)
![Ruff](https://img.shields.io/badge/Linter-Ruff-red.svg)

Este repositório contém a solução completa para o **Tech Challenge - Fase 1** da Pós-Tech em Machine Learning Engineering. O objetivo é prever o *churn* (cancelamento) de clientes de uma operadora de telecomunicações utilizando Redes Neurais Profundas.

---

## 📖 Contexto do Projeto
Uma operadora de telecomunicações enfrenta uma perda acelerada de clientes. Este projeto aplica uma abordagem **End-to-End**, desde a análise exploratória até a disponibilização de uma API de inferência, aplicando as boas práticas de engenharia de ML.

---

## 📂 Estrutura do Repositório
A organização segue rigorosamente os requisitos técnicos da Fase 1:

* `data/`: Datasets utilizados (Raw e Processed).
* `docs/`: Documentação técnica e Model Card.
* `models/`: Artefatos e modelos serializados.
* `notebooks/`: Análise Exploratória de Dados (EDA) e baselines.
* `src/`: Código-fonte modularizado.
* `tests/`: Testes automatizados (Unitários, API, Schema).
* `Makefile`: Atalhos para automação de tarefas (lint, test, run).
* `pyproject.toml`: Single source of truth para dependências e linting.

---

## 🚀 Como Executar

### 1. Configuração do Ambiente
```bash
# Criar ambiente virtual
python -m venv .venv

# Ativar ambiente (Windows)
.\.venv\Scripts\Activate.ps1

# Instalar dependências (Single Source of Truth)
pip install .

## 📒 Passo a passo para refletir as alteralções

# Adiciona todos os ficheiros alterados
git add .

# Regista as alterações (Exemplo de mensagem)
git commit -m "Comente a alteração realizada"

# Envia as suas alterações locais para o repositório remoto
git push origin main

---

## ⚙️ Setup e Instalação

### Pré-requisitos
- Python **3.11+**
- pip

### Instalação

```bash
# 1. Clone o repositório
git clone https://github.com/<seu-usuario>/tech-challenge-fase1.git
cd tech-challenge-fase1

# 2. Crie e ative o ambiente virtual
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

# 3. Instale as dependências (incluindo as de desenvolvimento)
pip install -e ".[dev]"
```

### Dataset

Baixe o dataset no Kaggle e coloque em `data/raw/`:

```
https://www.kaggle.com/datasets/blastchar/telco-customer-churn
```

O arquivo deve ficar em:
```
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

> **Atenção:** Os notebooks usam caminhos absolutos locais. Ao rodar em outra máquina, ajuste a variável `data_path` no início de cada notebook para o caminho correto do seu ambiente.

---

## 🚀 Execução

### Ordem de execução dos notebooks

```bash
# 1. Inicie o Jupyter
jupyter notebook

# 2. Execute nesta ordem:
#    notebooks/01_EDA.ipynb           → gera data/processed/telco_churn_clean.csv
#    notebooks/Modelos_baseline.ipynb → treina LR e LightGBM, registra no MLflow
#    notebooks/03_fairness.ipynb      → auditoria de fairness detalhada
```

### Visualizar experimentos no MLflow

```bash
mlflow ui --backend-store-uri file:./mlruns
# Acesse: http://localhost:5000
```

### Testes

```bash
pytest
```

### Linting

```bash
ruff check src/ tests/
```

---

## 🔬 Detalhes Técnicos

### Pipeline de pré-processamento

```python
# Numéricas: tenure, MonthlyCharges, TotalCharges, SeniorCitizen
SimpleImputer(strategy='median') → StandardScaler()

# Categóricas: 15 colunas (Contract, InternetService, etc.)
SimpleImputer(strategy='most_frequent') → OneHotEncoder(handle_unknown='ignore')
```

### Feature Selection

```python
SelectKBest(f_classif, k=15)
# Aplicado dentro do Pipeline após o pré-processamento
# Fit apenas em X_train para evitar data leakage
```

### Otimização de Hiperparâmetros

```python
# LightGBM otimizado com Optuna
# Melhores parâmetros registrados no MLflow
# AUC-ROC: 0.8517 | Recall: 0.8357 | F1: 0.6509
```

### Split dos dados

```python
# Estratificado para manter 26.5% de churn em todos os conjuntos
# 70% treino | 15% validação | 15% teste
# SEED = 42
```

### Fairness

Atributos sensíveis auditados: `gender`, `SeniorCitizen`, `Partner`, `Dependents`

Métrica principal: **Taxa de Falso Negativo (FNR)** por grupo  
Gate de fairness: disparidade FNR ≤ 10 p.p. entre grupos

---

## 📈 Métricas Alvo

| Métrica | Threshold | Melhor Baseline |
|---------|-----------|-----------------|
| AUC-ROC | ≥ 0.85 | 0.8517 (LightGBM) |
| Recall | ≥ 0.83 | 0.8357 (LightGBM) |
| F1-Score | ≥ 0.65 | 0.6509 (LightGBM) |
| Latência API | ≤ 200ms | — |

---

## ✅ Checklist de Boas Práticas (Etapa 1)

- [x] Seeds fixados para reprodutibilidade (SEED=42)
- [x] Validação cruzada estratificada (StratifiedKFold 5-fold)
- [x] Model Card documentando limitações e vieses (`docs/model_card.md`)
- [x] Testes automatizados — smoke test, schema test, API placeholder (`tests/test_data.py`)
- [x] Logging estruturado (sem print(), usando `logging.getLogger`)
- [x] DummyClassifier como Baseline 1 (piso mínimo)
- [x] Regressão Logística como Baseline 2
- [x] LightGBM + Optuna como Baseline 3
- [x] MLflow com parâmetros, métricas e dataset_version
- [x] ML Canvas preenchido (`docs/ml_canvas.md`)

## 🗺️ Roadmap

- [x] **Etapa 1** — EDA + 3 Baselines (Dummy, LR, LightGBM) + MLflow + Fairlearn

---

## 👥 Time

| Nome | RM |
|------|----|
| Gustavo Andres Silva Calderon | rm374865 |
| Lilian Dorini Almagro | rm374368 |
| Matheus Angelo Filetti | rm373015 |

---

## 📚 Referências

- [Telco Customer Churn Dataset — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [Fairlearn Documentation](https://fairlearn.org/)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)

---

## 📄 Licença

FIAP — Machine Learning Engineering Fase 1.
