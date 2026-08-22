"""Ingestão e limpeza do dataset Telco Customer Churn.

Contém a mesma lógica do notebook 01_EDA, mas testável e reutilizável:
`python -m churn.data.load` gera data/processed/telco_churn_clean.csv.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from churn.config import CLEAN_CSV, RAW_CSV, TARGET, ensure_dirs

logger = logging.getLogger(__name__)


def load_raw(path: Path | str = RAW_CSV) -> pd.DataFrame:
    """Lê o CSV bruto do Kaggle."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset bruto não encontrado em {path}. "
            "Baixe em https://www.kaggle.com/datasets/blastchar/telco-customer-churn "
            "e coloque em data/raw/."
        )
    df = pd.read_csv(path)
    logger.info("Dataset bruto carregado: %d linhas x %d colunas", *df.shape)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica as correções identificadas na EDA.

    1. TotalCharges vem como string com espaços em branco para clientes com
       tenure=0 (recém-adquiridos) -> converte para numérico, virando NaN.
    2. Churn Yes/No -> 1/0.
    3. SeniorCitizen -> int.
    """
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_nulos = int(df["TotalCharges"].isna().sum())
    if n_nulos:
        logger.info(
            "TotalCharges: %d nulos após conversão (clientes com tenure=0)", n_nulos
        )

    if df[TARGET].dtype == object:
        df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})
    df[TARGET] = df[TARGET].astype(int)

    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    logger.info("Taxa de churn: %.1f%%", df[TARGET].mean() * 100)
    return df


def build_clean_dataset(
    raw_path: Path | str = RAW_CSV, out_path: Path | str = CLEAN_CSV
) -> pd.DataFrame:
    """Pipeline completo: lê bruto, limpa e persiste."""
    ensure_dirs()
    df = clean(load_raw(raw_path))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Dataset limpo salvo em %s | shape=%s", out_path, df.shape)
    return df


def load_clean(path: Path | str = CLEAN_CSV) -> pd.DataFrame:
    """Carrega o dataset já tratado, gerando-o se ainda não existir."""
    path = Path(path)
    if not path.exists():
        logger.warning("%s não existe — gerando a partir do bruto.", path)
        return build_clean_dataset(out_path=path)
    return pd.read_csv(path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        force=True,
    )
    build_clean_dataset()
