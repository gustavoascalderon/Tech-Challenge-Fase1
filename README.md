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