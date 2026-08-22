"""Configuração central do projeto.

Todos os caminhos são derivados da raiz do repositório — nunca use caminhos
absolutos da sua máquina no código. Podem ser sobrescritos por variáveis de
ambiente, o que é o que a API usa em container.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

# ---------------------------------------------------------------- caminhos ---
# config.py → churn → src → raiz do repo
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.getenv("CHURN_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = Path(os.getenv("CHURN_MODELS_DIR", ROOT / "models"))
DOCS_DIR = ROOT / "docs"
MLRUNS_DIR = Path(os.getenv("CHURN_MLRUNS_DIR", ROOT / "mlruns"))

RAW_CSV = RAW_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
CLEAN_CSV = PROCESSED_DIR / "telco_churn_clean.csv"

# Artefatos servidos pela API
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
MLP_WEIGHTS_PATH = MODELS_DIR / "mlp_churn.pt"
MLP_METADATA_PATH = MODELS_DIR / "mlp_metadata.json"

# ------------------------------------------------------------- reprodutib. ---
SEED = 42

# ------------------------------------------------------------------ schema ---
TARGET = "Churn"
ID_COL = "customerID"

NUM_FEATURES = ["tenure", "MonthlyCharges", "SeniorCitizen"]

CAT_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

FEATURES = NUM_FEATURES + CAT_FEATURES

# TotalCharges é removida: correlação ~0.83 com tenure (tenure × MonthlyCharges)
DROP_COLS = [ID_COL, "TotalCharges"]

# Atributos sensíveis auditados na análise de fairness
SENSITIVE_FEATURES = ["gender", "SeniorCitizen", "Partner", "Dependents"]

# ------------------------------------------------------------------ splits ---
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# ------------------------------------------------------------------ MLflow ---
EXPERIMENT_NAME = os.getenv("CHURN_EXPERIMENT", "churn-prediction-fase1")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"file://{MLRUNS_DIR}")

# ------------------------------------------------------- gates de qualidade ---
# Thresholds que a MLP precisa bater (definidos pelo melhor baseline: LightGBM)
GATE_AUC_ROC = 0.85
GATE_RECALL = 0.83
GATE_F1 = 0.65
# Disparidade máxima de FNR entre grupos de um atributo sensível (p.p.)
GATE_FNR_DISPARITY = 0.10


def ensure_dirs() -> None:
    """Cria os diretórios de saída se ainda não existirem."""
    for d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, DOCS_DIR, MLRUNS_DIR):
        d.mkdir(parents=True, exist_ok=True)
