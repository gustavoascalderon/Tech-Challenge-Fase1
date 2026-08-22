"""Baselines de referência: Dummy, Regressão Logística e LightGBM + Optuna.

Refatoração do notebook Modelos_baseline.ipynb. Uso:
    python -m churn.models.baselines --trials 30
"""

from __future__ import annotations

import argparse
import logging

import mlflow
import mlflow.sklearn
import optuna
from lightgbm import LGBMClassifier
from mlflow.models.signature import infer_signature
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from churn.config import EXPERIMENT_NAME, MLFLOW_TRACKING_URI, SEED, ensure_dirs
from churn.data.load import load_clean
from churn.data.validate import validate
from churn.features.preprocess import build_preprocessor, split_data
from churn.models.evaluate import comparison_table, compute_metrics

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _log_run(nome: str, pipeline, params: dict, metricas: dict, X_train, tags: dict):
    """Registra um run no MLflow com params, métricas, modelo e tags."""
    with mlflow.start_run(run_name=nome) as run:
        mlflow.log_params(params)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_metrics(metricas)
        signature = infer_signature(X_train, pipeline.predict(X_train))
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=signature,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
        for k, v in tags.items():
            mlflow.set_tag(k, v)
        return run.info.run_id


def run_dummy(splits, log_mlflow: bool = True) -> dict:
    """Baseline 1 — piso mínimo. Qualquer modelo útil precisa superá-lo."""
    params = {"strategy": "most_frequent", "random_state": SEED}
    pipe = Pipeline(
        [("preprocessor", build_preprocessor()), ("classifier", DummyClassifier(**params))]
    )
    pipe.fit(splits.X_train, splits.y_train)
    proba = pipe.predict_proba(splits.X_test)[:, 1]
    metricas = compute_metrics(
        splits.y_test, pipe.predict(splits.X_test), proba, "DummyClassifier"
    )
    if log_mlflow:
        _log_run(
            "baseline-dummy",
            pipe,
            {**params, "model_type": "DummyClassifier"},
            metricas,
            splits.X_train,
            {"stage": "baseline", "dataset_version": "v1.0"},
        )
    return metricas


def run_logistic(splits, log_mlflow: bool = True) -> dict:
    """Baseline 2 — modelo linear interpretável."""
    params = {
        "C": 1.0,
        "max_iter": 1000,
        "solver": "liblinear",
        "penalty": "l1",
        "class_weight": "balanced",
        "random_state": SEED,
    }
    pipe = Pipeline(
        [("preprocessor", build_preprocessor()), ("classifier", LogisticRegression(**params))]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_res = cross_validate(
        pipe,
        splits.X_train,
        splits.y_train,
        cv=cv,
        scoring=["accuracy", "f1", "recall", "precision", "roc_auc"],
    )
    for m in ["f1", "recall", "roc_auc"]:
        s = cv_res[f"test_{m}"]
        logger.info("CV %s: %.4f ± %.4f", m, s.mean(), s.std())

    pipe.fit(splits.X_train, splits.y_train)
    proba = pipe.predict_proba(splits.X_test)[:, 1]
    metricas = compute_metrics(
        splits.y_test, pipe.predict(splits.X_test), proba, "Regressão Logística"
    )

    if log_mlflow:
        cv_metricas = {
            f"cv_{m}_mean": cv_res[f"test_{m}"].mean()
            for m in ["accuracy", "f1", "recall", "precision", "roc_auc"]
        }
        _log_run(
            "baseline-logistic-regression",
            pipe,
            {**params, "model_type": "LogisticRegression", "cv_folds": 5},
            {**metricas, **cv_metricas},
            splits.X_train,
            {"stage": "baseline", "dataset_version": "v1.0"},
        )
    return metricas


def run_lightgbm(splits, n_trials: int = 30, log_mlflow: bool = True) -> dict:
    """Baseline 3 — LightGBM com hiperparâmetros buscados por Optuna."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "class_weight": "balanced",
            "random_state": SEED,
            "n_jobs": -1,
            "verbose": -1,
        }
        k = trial.suggest_int("k", 10, 35)
        pipe = Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                ("selector", SelectKBest(f_classif, k=k)),
                ("classifier", LGBMClassifier(**params)),
            ]
        )
        res = cross_validate(
            pipe, splits.X_train, splits.y_train, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        return float(res["test_score"].mean())

    logger.info("Otimizando LightGBM com Optuna (%d trials)...", n_trials)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    study.optimize(objective, n_trials=n_trials)
    logger.info("Melhor ROC-AUC (CV): %.4f", study.best_value)

    melhores = study.best_params.copy()
    k = melhores.pop("k")
    melhores |= {
        "class_weight": "balanced",
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": -1,
    }

    pipe = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("selector", SelectKBest(f_classif, k=k)),
            ("classifier", LGBMClassifier(**melhores)),
        ]
    )
    pipe.fit(splits.X_train, splits.y_train)
    proba = pipe.predict_proba(splits.X_test)[:, 1]
    metricas = compute_metrics(
        splits.y_test, pipe.predict(splits.X_test), proba, "LightGBM (Optuna)"
    )

    if log_mlflow:
        _log_run(
            "baseline-lightgbm-optuna",
            pipe,
            {**melhores, "k_features": k, "model_type": "LightGBM", "n_trials": n_trials},
            metricas,
            splits.X_train,
            {"stage": "baseline", "dataset_version": "v1.0", "optuna": "true"},
        )
    return metricas


def run_all(n_trials: int = 30, log_mlflow: bool = True):
    ensure_dirs()
    splits = split_data(validate(load_clean()))
    resultados = {
        "DummyClassifier": run_dummy(splits, log_mlflow),
        "Regressão Logística": run_logistic(splits, log_mlflow),
        "LightGBM (Optuna)": run_lightgbm(splits, n_trials, log_mlflow),
    }
    tabela = comparison_table(resultados)
    logger.info("Comparativo dos baselines:\n%s", tabela.to_string())
    return tabela


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina os baselines de referência.")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        force=True,
    )
    if not args.no_mlflow:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
    run_all(args.trials, log_mlflow=not args.no_mlflow)


if __name__ == "__main__":
    main()
