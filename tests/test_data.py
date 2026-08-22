"""
Testes básicos para a Etapa 1 — serão expandidos na Etapa 3.
"""
import pandas as pd
import numpy as np
import pytest


def test_dataset_shape():
    """Smoke test: verifica que o dataset processado tem o shape esperado."""
    try:
        df = pd.read_csv('data/processed/telco_churn_clean.csv')
        assert df.shape[0] > 7000, "Dataset deve ter mais de 7000 linhas"
        assert df.shape[1] == 21, "Dataset deve ter 21 colunas"
        assert 'Churn' in df.columns, "Coluna 'Churn' deve existir"
    except FileNotFoundError:
        pytest.skip("Dataset não encontrado — execute o notebook 01_eda primeiro")


def test_target_is_binary():
    """Verifica que o target é binário (0 ou 1)."""
    try:
        df = pd.read_csv('data/processed/telco_churn_clean.csv')
        valores_unicos = set(df['Churn'].unique())
        assert valores_unicos.issubset({0, 1}), f"Target deve ser binário, encontrado: {valores_unicos}"
    except FileNotFoundError:
        pytest.skip("Dataset não encontrado")


def test_sem_duplicatas():
    """Verifica que não há registros duplicados."""
    try:
        df = pd.read_csv('data/processed/telco_churn_clean.csv')
        assert df.duplicated().sum() == 0, "Dataset não deve ter duplicatas"
    except FileNotFoundError:
        pytest.skip("Dataset não encontrado")


def test_churn_desbalanceado():
    """Verifica que o churn rate está na faixa esperada (~26-28%)."""
    try:
        df = pd.read_csv('data/processed/telco_churn_clean.csv')
        churn_rate = df['Churn'].mean()
        assert 0.20 <= churn_rate <= 0.35, f"Churn rate fora da faixa esperada: {churn_rate:.3f}"
    except FileNotFoundError:
        pytest.skip("Dataset não encontrado")
