"""API FastAPI que serve a MLP de churn.

Sobe com:
    uvicorn churn.api.main:app --reload --port 8000

Endpoints:
    GET  /health       — liveness/readiness (não carrega modelo se já em memória)
    GET  /model-info   — metadados e métricas do modelo em produção
    POST /predict      — predição individual
    POST /predict/batch— predição em lote (até 1000 clientes)
    GET  /docs         — Swagger UI
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from churn.api.schemas import (
    BatchRequest,
    BatchResponse,
    CustomerFeatures,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)
from churn.config import FEATURES
from churn.features.preprocess import load_preprocessor
from churn.models.mlp import get_device, load_model

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger("churn.api")

# Estado carregado uma única vez no startup — nunca por request
ARTEFATOS: dict = {"model": None, "preprocessor": None, "metadata": None, "device": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega os artefatos no startup e libera no shutdown."""
    try:
        device = get_device()
        model, metadata = load_model(device=device)
        ARTEFATOS.update(
            model=model,
            preprocessor=load_preprocessor(),
            metadata=metadata,
            device=device,
        )
        logger.info("Artefatos carregados | device=%s", device)
    except FileNotFoundError as exc:
        # A API sobe mesmo sem modelo: /health reporta 'degraded' e o
        # orquestrador não promove o container para produção.
        logger.error("Modelo não carregado: %s", exc)
    yield
    ARTEFATOS.clear()
    logger.info("Recursos liberados.")


app = FastAPI(
    title="Churn Guard API",
    description="Predição de churn de clientes de telecom — MLP (PyTorch).",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# --------------------------------------------------------------- helpers ----
def _exigir_modelo() -> None:
    if ARTEFATOS.get("model") is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo indisponível. Rode `python -m churn.models.train_mlp`.",
        )


def _faixa_risco(p: float) -> str:
    if p >= 0.70:
        return "alto"
    if p >= 0.40:
        return "medio"
    return "baixo"


def _predizer(clientes: list[CustomerFeatures]) -> list[tuple[float, int, str]]:
    """Transforma o payload e roda a MLP em batch."""
    df = pd.DataFrame([c.model_dump() for c in clientes])[FEATURES]
    X = ARTEFATOS["preprocessor"].transform(df)
    probas = ARTEFATOS["model"].predict_proba(X, ARTEFATOS["device"])
    threshold = ARTEFATOS["metadata"]["config"]["threshold"]
    return [
        (float(p), int(p >= threshold), _faixa_risco(float(p))) for p in probas
    ]


# ------------------------------------------------------------- endpoints ----
@app.get("/health", response_model=HealthResponse, tags=["infra"])
def health() -> HealthResponse:
    carregado = ARTEFATOS.get("model") is not None
    metadata = ARTEFATOS.get("metadata") or {}
    return HealthResponse(
        status="ok" if carregado else "degraded",
        model_loaded=carregado,
        model_version=metadata.get("dataset_version"),
        n_features=metadata.get("n_features"),
        device=str(ARTEFATOS.get("device")) if carregado else None,
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["infra"])
def model_info() -> ModelInfoResponse:
    _exigir_modelo()
    metadata = ARTEFATOS["metadata"]
    config = metadata["config"]
    return ModelInfoResponse(
        architecture="MLP (PyTorch) — Linear+BatchNorm+ReLU+Dropout",
        hidden_dims=config["hidden_dims"],
        n_parameters=ARTEFATOS["model"].n_parameters(),
        n_features=metadata["n_features"],
        threshold=config["threshold"],
        metrics_test=metadata["metrics_test"],
        best_epoch=metadata["best_epoch"],
        dataset_version=metadata["dataset_version"],
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inferência"])
def predict(customer: CustomerFeatures) -> PredictionResponse:
    """Prediz o risco de churn de um cliente."""
    _exigir_modelo()
    inicio = time.perf_counter()
    proba, pred, faixa = _predizer([customer])[0]
    latencia = (time.perf_counter() - inicio) * 1000

    return PredictionResponse(
        churn_probability=round(proba, 4),
        churn_prediction=pred,
        risk_band=faixa,
        threshold=ARTEFATOS["metadata"]["config"]["threshold"],
        model_version=ARTEFATOS["metadata"]["dataset_version"],
        latency_ms=round(latencia, 2),
    )


@app.post("/predict/batch", response_model=BatchResponse, tags=["inferência"])
def predict_batch(request: BatchRequest) -> BatchResponse:
    """Prediz em lote — uma única passada pela rede, muito mais eficiente."""
    _exigir_modelo()
    inicio = time.perf_counter()
    resultados = _predizer(request.customers)
    latencia = (time.perf_counter() - inicio) * 1000

    threshold = ARTEFATOS["metadata"]["config"]["threshold"]
    versao = ARTEFATOS["metadata"]["dataset_version"]
    por_cliente = round(latencia / len(resultados), 3)

    return BatchResponse(
        predictions=[
            PredictionResponse(
                churn_probability=round(p, 4),
                churn_prediction=pred,
                risk_band=faixa,
                threshold=threshold,
                model_version=versao,
                latency_ms=por_cliente,
            )
            for p, pred, faixa in resultados
        ],
        n_customers=len(resultados),
        total_latency_ms=round(latencia, 2),
    )
