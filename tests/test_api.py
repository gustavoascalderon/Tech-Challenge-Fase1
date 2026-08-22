"""Testes de contrato da API.

Um modelo pequeno é treinado uma vez por sessão e injetado no estado da API,
então a suíte não depende de artefatos versionados nem de um treino completo.
"""

from __future__ import annotations
import torch
import numpy as np
import pytest
from fastapi.testclient import TestClient

from churn.api import main as api_main
from churn.features.preprocess import build_preprocessor, split_data
from churn.models.mlp import ChurnMLP, MLPConfig, get_device, set_seed

PAYLOAD = {
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


@pytest.fixture(scope="module")
def client(clean_df):
    """Cliente com artefatos em memória (sem tocar em models/)."""
    set_seed(42)
    splits = split_data(clean_df)
    preprocessor = build_preprocessor().fit(splits.X_train)
    n_features = preprocessor.transform(splits.X_train).shape[1]

    config = MLPConfig(input_dim=n_features, hidden_dims=[16, 8], threshold=0.5)
    model = ChurnMLP(config).eval()

    api_main.ARTEFATOS.update(
        model=model,
        preprocessor=preprocessor,
        torch.device("cpu"),
        metadata={
            "config": config.to_dict(),
            "metrics_test": {"auc_roc": 0.86, "recall": 0.84, "f1": 0.66},
            "best_epoch": 10,
            "n_features": n_features,
            "dataset_version": "v-teste",
        },
    )
    # lifespan=None impede que o startup sobrescreva os artefatos injetados
    with TestClient(api_main.app) as c:
        api_main.ARTEFATOS.update(
            model=model, preprocessor=preprocessor, device=get_device(),
            metadata=api_main.ARTEFATOS["metadata"],
        )
        yield c


# ------------------------------------------------------------------ infra ---
def test_health_ok(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True


def test_model_info(client):
    body = client.get("/model-info").json()
    assert "MLP" in body["architecture"]
    assert body["n_parameters"] > 0
    assert 0 < body["threshold"] < 1


def test_openapi_disponivel(client):
    assert client.get("/openapi.json").status_code == 200


# -------------------------------------------------------------- inferência ---
def test_predict_contrato(client):
    r = client.post("/predict", json=PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["churn_prediction"] in (0, 1)
    assert body["risk_band"] in ("baixo", "medio", "alto")


def test_predict_deterministico(client):
    """Mesma entrada, mesma saída — dropout desligado em eval."""
    a = client.post("/predict", json=PAYLOAD).json()["churn_probability"]
    b = client.post("/predict", json=PAYLOAD).json()["churn_probability"]
    assert a == b


def test_latencia_abaixo_do_gate(client):
    """Gate de negócio: p95 <= 200ms."""
    latencias = [
        client.post("/predict", json=PAYLOAD).json()["latency_ms"] for _ in range(20)
    ]
    assert float(np.percentile(latencias, 95)) < 200


def test_batch(client):
    r = client.post("/predict/batch", json={"customers": [PAYLOAD] * 10})
    assert r.status_code == 200
    body = r.json()
    assert body["n_customers"] == 10 and len(body["predictions"]) == 10


def test_batch_vazio_rejeitado(client):
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422


# -------------------------------------------------------------- validação ---
@pytest.mark.parametrize(
    "campo,valor",
    [
        ("Contract", "Vitalício"),          # categoria inexistente
        ("tenure", -5),                     # fora do intervalo
        ("MonthlyCharges", 99999),          # fora do intervalo
        ("SeniorCitizen", 2),               # fora do domínio
        ("InternetService", "5G"),          # categoria inexistente
    ],
)
def test_valores_invalidos_retornam_422(client, campo, valor):
    assert client.post("/predict", json={**PAYLOAD, campo: valor}).status_code == 422


def test_campo_faltando_retorna_422(client):
    incompleto = {k: v for k, v in PAYLOAD.items() if k != "Contract"}
    assert client.post("/predict", json=incompleto).status_code == 422


def test_campo_extra_e_ignorado(client):
    """Campo desconhecido não deve derrubar a API."""
    r = client.post("/predict", json={**PAYLOAD, "customerID": "1234-XYZ"})
    assert r.status_code == 200
