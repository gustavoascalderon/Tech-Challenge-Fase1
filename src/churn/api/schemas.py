"""Contratos de entrada e saída da API (Pydantic v2)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CustomerFeatures(BaseModel):
    """Features de um cliente. Os Literal fazem a validação de domínio:
    categoria fora do vocabulário do treino vira HTTP 422, não predição silenciosa.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "tenure": 3,
                "MonthlyCharges": 89.9,
                "SeniorCitizen": 0,
                "gender": "Female",
                "Partner": "No",
                "Dependents": "No",
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
            }
        }
    )

    tenure: int = Field(..., ge=0, le=100, description="Meses como cliente")
    MonthlyCharges: float = Field(..., ge=0, le=1000, description="Cobrança mensal")
    SeniorCitizen: Literal[0, 1] = Field(..., description="1 se idoso")

    gender: Literal["Male", "Female"]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]


class PredictionResponse(BaseModel):
    """Resposta de uma predição individual."""

    churn_probability: float = Field(..., ge=0, le=1)
    churn_prediction: int = Field(..., description="0 = fica | 1 = risco de churn")
    risk_band: Literal["baixo", "medio", "alto"]
    threshold: float
    model_version: str
    latency_ms: float


class BatchRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(..., min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    n_customers: int
    total_latency_ms: float


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None = None
    n_features: int | None = None
    device: str | None = None


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    architecture: str
    hidden_dims: list[int]
    n_parameters: int
    n_features: int
    threshold: float
    metrics_test: dict[str, float]
    best_epoch: int
    dataset_version: str
