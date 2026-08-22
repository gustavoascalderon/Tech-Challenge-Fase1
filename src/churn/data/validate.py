"""Validação de schema do dataset com pandera.

Roda como gate na ingestão: se o contrato de dados quebrar, o treino falha
antes de produzir um modelo silenciosamente errado.
"""

from __future__ import annotations

import logging

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

from churn.config import CAT_FEATURES, TARGET

logger = logging.getLogger(__name__)

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

CHURN_SCHEMA = DataFrameSchema(
    {
        "customerID": Column(str, unique=True, nullable=False),
        "tenure": Column(int, Check.in_range(0, 100), nullable=False),
        "MonthlyCharges": Column(float, Check.in_range(0, 1000), nullable=False),
        "TotalCharges": Column(float, Check.ge(0), nullable=True),
        "SeniorCitizen": Column(int, Check.isin([0, 1]), nullable=False),
        TARGET: Column(int, Check.isin([0, 1]), nullable=False),
        **{
            col: Column(str, Check.isin(vals), nullable=False)
            for col, vals in _CATEGORIAS.items()
            if col in CAT_FEATURES
        },
    },
    strict=False,  # tolera colunas extras
    coerce=True,
)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Valida o DataFrame contra o contrato. Levanta SchemaError se falhar."""
    validated = CHURN_SCHEMA.validate(df, lazy=True)
    logger.info("Schema validado: %d linhas aprovadas", len(validated))
    return validated


def get_allowed_categories() -> dict[str, list[str]]:
    """Categorias válidas — usado pela API para validar o payload de entrada."""
    return {k: list(v) for k, v in _CATEGORIAS.items()}


__all__ = ["CHURN_SCHEMA", "validate", "get_allowed_categories", "pa"]
