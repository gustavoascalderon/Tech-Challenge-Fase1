"""Avaliação: métricas, ajuste de threshold e auditoria de fairness."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from churn.config import (
    GATE_AUC_ROC,
    GATE_F1,
    GATE_FNR_DISPARITY,
    GATE_RECALL,
    SENSITIVE_FEATURES,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    nome: str = "Modelo",
) -> dict[str, float]:
    """Métricas principais. Recall é a métrica de negócio prioritária:
    perder um cliente que ia cancelar custa mais do que oferecer retenção
    a quem não ia cancelar.
    """
    metricas: dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metricas["auc_roc"] = roc_auc_score(y_true, y_proba)
        metricas["pr_auc"] = average_precision_score(y_true, y_proba)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metricas["fnr"] = fn / (fn + tp) if (fn + tp) else 0.0
    metricas["fpr"] = fp / (fp + tn) if (fp + tn) else 0.0

    logger.info(
        "%s | AUC %.4f | Recall %.4f | F1 %.4f | Precision %.4f",
        nome,
        metricas.get("auc_roc", float("nan")),
        metricas["recall"],
        metricas["f1"],
        metricas["precision"],
    )
    return metricas


def find_best_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "f1",
    min_recall: float = GATE_RECALL,
) -> tuple[float, float]:
    """Varre thresholds e escolhe o melhor sujeito a um recall mínimo.

    Buscado no conjunto de **validação**, nunca no de teste.
    Retorna (threshold, valor_da_metrica).
    """
    scorers = {
        "f1": f1_score,
        "precision": precision_score,
        "recall": recall_score,
    }
    scorer = scorers[metric]

    candidatos: list[tuple[float, float]] = []
    for t in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_proba >= t).astype(int)
        if recall_score(y_true, y_pred, zero_division=0) < min_recall:
            continue
        candidatos.append((float(t), float(scorer(y_true, y_pred, zero_division=0))))

    if not candidatos:
        logger.warning(
            "Nenhum threshold atinge recall >= %.2f — usando o de recall máximo.",
            min_recall,
        )
        recalls = [
            (float(t), recall_score(y_true, (y_proba >= t).astype(int), zero_division=0))
            for t in np.arange(0.05, 0.96, 0.01)
        ]
        return max(recalls, key=lambda x: x[1])

    melhor = max(candidatos, key=lambda x: x[1])
    logger.info("Melhor threshold: %.2f (%s=%.4f)", melhor[0], metric, melhor[1])
    return melhor


def fairness_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_df: pd.DataFrame,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """FNR, seleção e recall por grupo de cada atributo sensível.

    FNR é a métrica principal: um falso negativo significa um cliente em risco
    que não recebeu ação de retenção. Disparidade sistemática de FNR entre
    grupos significa que um grupo é sub-atendido pelo modelo.
    """
    features = features or [f for f in SENSITIVE_FEATURES if f in sensitive_df.columns]
    linhas = []

    for feat in features:
        for grupo in sorted(sensitive_df[feat].dropna().unique()):
            mask = (sensitive_df[feat] == grupo).to_numpy()
            if mask.sum() == 0:
                continue
            yt, yp = np.asarray(y_true)[mask], np.asarray(y_pred)[mask]
            labels_presentes = set(np.unique(yt)) | set(np.unique(yp))
            if labels_presentes <= {0}:
                continue
            tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
            linhas.append(
                {
                    "atributo": feat,
                    "grupo": str(grupo),
                    "n": int(mask.sum()),
                    "taxa_churn_real": float(yt.mean()),
                    "taxa_selecao": float(yp.mean()),
                    "recall": float(tp / (tp + fn)) if (tp + fn) else np.nan,
                    "fnr": float(fn / (fn + tp)) if (fn + tp) else np.nan,
                    "fpr": float(fp / (fp + tn)) if (fp + tn) else np.nan,
                }
            )

    return pd.DataFrame(linhas)


def fairness_gate(
    report: pd.DataFrame, max_disparity: float = GATE_FNR_DISPARITY
) -> tuple[bool, pd.DataFrame]:
    """Verifica se a disparidade de FNR entre grupos está dentro do limite."""
    disparidades = (
        report.groupby("atributo")["fnr"]
        .agg(fnr_min="min", fnr_max="max")
        .assign(disparidade=lambda d: d["fnr_max"] - d["fnr_min"])
        .assign(aprovado=lambda d: d["disparidade"] <= max_disparity)
        .reset_index()
    )
    aprovado = bool(disparidades["aprovado"].all())
    if not aprovado:
        reprovados = disparidades.loc[~disparidades["aprovado"], "atributo"].tolist()
        logger.warning("Gate de fairness REPROVADO para: %s", reprovados)
    else:
        logger.info("Gate de fairness aprovado (disparidade FNR <= %.2f)", max_disparity)
    return aprovado, disparidades


def quality_gate(metricas: dict[str, float]) -> tuple[bool, dict[str, bool]]:
    """Compara as métricas contra os thresholds definidos no config."""
    checks = {
        "auc_roc": metricas.get("auc_roc", 0) >= GATE_AUC_ROC,
        "recall": metricas.get("recall", 0) >= GATE_RECALL,
        "f1": metricas.get("f1", 0) >= GATE_F1,
    }
    aprovado = all(checks.values())
    logger.info(
        "Quality gate: %s | %s",
        "APROVADO" if aprovado else "REPROVADO",
        {k: ("ok" if v else "falhou") for k, v in checks.items()},
    )
    return aprovado, checks


def comparison_table(resultados: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Tabela comparativa entre modelos, ordenada por AUC-ROC."""
    return (
        pd.DataFrame(resultados)
        .T.round(4)
        .sort_values("auc_roc", ascending=False)
    )
