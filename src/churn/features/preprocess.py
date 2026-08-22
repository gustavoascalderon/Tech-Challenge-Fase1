"""Engenharia de features: split estratificado e pipeline de pré-processamento.

O mesmo `preprocessor` alimenta os baselines sklearn e a MLP em PyTorch —
garantindo que a comparação seja justa e que a API aplique exatamente a mesma
transformação vista no treino.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.config import (
    CAT_FEATURES,
    DROP_COLS,
    NUM_FEATURES,
    PREPROCESSOR_PATH,
    SEED,
    TARGET,
    TEST_SIZE,
    VAL_SIZE,
)

logger = logging.getLogger(__name__)


@dataclass
class DataSplits:
    """Container dos conjuntos de treino/validação/teste."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series

    def summary(self) -> str:
        total = len(self.X_train) + len(self.X_val) + len(self.X_test)
        linhas = []
        for nome, X, y in [
            ("treino", self.X_train, self.y_train),
            ("validação", self.X_val, self.y_val),
            ("teste", self.X_test, self.y_test),
        ]:
            linhas.append(
                f"  {nome:10s}: {len(X):5,d} ({len(X) / total * 100:4.1f}%) "
                f"| churn {y.mean() * 100:.1f}%"
            )
        return "\n".join(linhas)


def split_data(df: pd.DataFrame, seed: int = SEED) -> DataSplits:
    """Split estratificado 70/15/15 mantendo a proporção de churn."""
    X = df.drop(columns=[TARGET] + [c for c in DROP_COLS if c in df.columns])
    y = df[TARGET]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
    )
    # VAL_SIZE é fração do total; aqui vira fração do que sobrou
    val_relativo = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relativo, random_state=seed, stratify=y_temp
    )

    splits = DataSplits(X_train, X_val, X_test, y_train, y_val, y_test)
    logger.info("Split estratificado:\n%s", splits.summary())
    return splits


def build_preprocessor() -> ColumnTransformer:
    """Numéricas: mediana + StandardScaler. Categóricas: moda + OneHot."""
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUM_FEATURES),
            ("cat", categorical, CAT_FEATURES),
        ]
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Nomes das colunas após o one-hot — usado em interpretabilidade."""
    return list(preprocessor.get_feature_names_out())


def fit_transform_splits(
    splits: DataSplits, preprocessor: ColumnTransformer | None = None
) -> tuple[ColumnTransformer, dict[str, np.ndarray]]:
    """Ajusta o preprocessor SOMENTE no treino e transforma os três conjuntos.

    Isso é o que evita data leakage: validação e teste apenas recebem
    `transform`, nunca `fit`.
    """
    preprocessor = preprocessor or build_preprocessor()
    Xtr = preprocessor.fit_transform(splits.X_train)
    arrays = {
        "X_train": np.asarray(Xtr, dtype=np.float32),
        "X_val": np.asarray(preprocessor.transform(splits.X_val), dtype=np.float32),
        "X_test": np.asarray(preprocessor.transform(splits.X_test), dtype=np.float32),
        "y_train": splits.y_train.to_numpy(dtype=np.float32),
        "y_val": splits.y_val.to_numpy(dtype=np.float32),
        "y_test": splits.y_test.to_numpy(dtype=np.float32),
    }
    logger.info(
        "Pré-processamento aplicado | %d features após one-hot",
        arrays["X_train"].shape[1],
    )
    return preprocessor, arrays


def save_preprocessor(
    preprocessor: ColumnTransformer, path: Path | str | None = None
) -> Path:
    path = Path(path or PREPROCESSOR_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)
    logger.info("Preprocessor salvo em %s", path)
    return path


def load_preprocessor(path: Path | str | None = None) -> ColumnTransformer:
    path = Path(path or PREPROCESSOR_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"Preprocessor não encontrado em {path}. "
            "Rode `python -m churn.models.train_mlp` antes de subir a API."
        )
    return joblib.load(path)
