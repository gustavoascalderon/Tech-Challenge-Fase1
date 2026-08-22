"""Fixtures compartilhadas.

Os testes geram um dataset sintético com o MESMO schema do Telco, então a
suíte roda no CI sem depender do CSV do Kaggle (que não é versionado).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn.data.load import clean

_CATEGORIAS = {
    "gender": ["Male", "Female"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}


def make_synthetic(n: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Dataset sintético com sinal real de churn (para o modelo aprender algo)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.choice(v, n) for c, v in _CATEGORIAS.items()})
    df["customerID"] = [f"{i:04d}-TESTE" for i in range(n)]
    df["tenure"] = rng.integers(0, 73, n)
    df["MonthlyCharges"] = rng.uniform(18, 119, n).round(2)
    df["SeniorCitizen"] = rng.choice([0, 1], n, p=[0.84, 0.16])

    total = (df["tenure"] * df["MonthlyCharges"]).round(2).astype(object)
    total[df["tenure"] == 0] = " "  # replica o defeito do dataset original
    df["TotalCharges"] = total

    logit = (
        -1.6
        - 0.05 * df["tenure"]
        + 0.02 * df["MonthlyCharges"]
        + 1.3 * (df["Contract"] == "Month-to-month")
        + 0.7 * (df["InternetService"] == "Fiber optic")
    )
    df["Churn"] = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")
    return df


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return make_synthetic()


@pytest.fixture(scope="session")
def clean_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    return clean(raw_df)


@pytest.fixture(scope="module")
def monkeypatch_module():
    """monkeypatch com escopo de módulo (o embutido é function-scoped)."""
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    yield mp
    mp.undo()
