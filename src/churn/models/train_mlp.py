"""Treino da MLP em PyTorch com Early Stopping e tracking no MLflow.

Uso:
    python -m churn.models.train_mlp
    python -m churn.models.train_mlp --max-epochs 300 --patience 20

Ao final, persiste em models/:
    preprocessor.joblib | mlp_churn.pt | mlp_metadata.json
que são exatamente os artefatos consumidos pela API.
"""

from __future__ import annotations

import argparse
import logging
import time

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from churn.config import (
    EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    SEED,
    SENSITIVE_FEATURES,
    ensure_dirs,
)
from churn.data.load import load_clean
from churn.data.validate import validate
from churn.features.preprocess import (
    build_preprocessor,
    fit_transform_splits,
    save_preprocessor,
    split_data,
)
from churn.models.evaluate import (
    compute_metrics,
    fairness_gate,
    fairness_report,
    find_best_threshold,
    quality_gate,
)
from churn.models.mlp import (
    ChurnMLP,
    EarlyStopping,
    MLPConfig,
    get_device,
    save_model,
    set_seed,
)

logger = logging.getLogger(__name__)


def make_loaders(
    arrays: dict[str, np.ndarray], batch_size: int, seed: int = SEED
) -> tuple[DataLoader, DataLoader]:
    """DataLoaders de treino (shuffle) e validação (ordem fixa)."""
    gerador = torch.Generator().manual_seed(seed)

    train_ds = TensorDataset(
        torch.from_numpy(arrays["X_train"]), torch.from_numpy(arrays["y_train"])
    )
    val_ds = TensorDataset(
        torch.from_numpy(arrays["X_val"]), torch.from_numpy(arrays["y_val"])
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=gerador, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size * 4, shuffle=False)
    return train_loader, val_loader


def compute_pos_weight(y_train: np.ndarray) -> torch.Tensor:
    """Peso da classe positiva = n_negativos / n_positivos.

    Equivale ao class_weight='balanced' usado nos baselines.
    """
    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - n_pos)
    peso = n_neg / max(n_pos, 1.0)
    logger.info("pos_weight = %.3f (neg=%d, pos=%d)", peso, int(n_neg), int(n_pos))
    return torch.tensor(peso, dtype=torch.float32)


def train_one_epoch(
    model: ChurnMLP,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        # Clipping evita explosão de gradiente nas primeiras épocas
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total += loss.item() * len(xb)
        n += len(xb)
    return total / max(n, 1)


@torch.no_grad()
def evaluate_epoch(
    model: ChurnMLP, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> tuple[float, float, np.ndarray]:
    model.eval()
    total, n = 0.0, 0
    probas, alvos = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        total += criterion(logits, yb).item() * len(xb)
        n += len(xb)
        probas.append(torch.sigmoid(logits).cpu().numpy())
        alvos.append(yb.cpu().numpy())

    y_proba = np.concatenate(probas)
    y_true = np.concatenate(alvos)
    auc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.5
    return total / max(n, 1), float(auc), y_proba


def train(config: MLPConfig | None = None, log_mlflow: bool = True) -> dict:
    """Pipeline completo de treino da MLP."""
    ensure_dirs()
    set_seed(SEED)
    device = get_device()
    logger.info("Device: %s", device)

    # ---------------------------------------------------------- dados ----
    df = validate(load_clean())
    splits = split_data(df)
    preprocessor, arrays = fit_transform_splits(splits, build_preprocessor())

    config = config or MLPConfig(input_dim=arrays["X_train"].shape[1])
    config.input_dim = arrays["X_train"].shape[1]

    train_loader, val_loader = make_loaders(arrays, config.batch_size)

    # --------------------------------------------------------- modelo ----
    model = ChurnMLP(config).to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=compute_pos_weight(arrays["y_train"]).to(device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=max(config.patience // 3, 2)
    )
    stopper = EarlyStopping(
        patience=config.patience,
        min_delta=config.min_delta,
        mode="max" if config.monitor == "val_auc" else "min",
        restore_best_weights=True,
    )

    logger.info(
        "MLP: %d features -> %s -> 1 | %d parâmetros",
        config.input_dim,
        config.hidden_dims,
        model.n_parameters(),
    )

    # ---------------------------------------------------------- treino ----
    historico: list[dict] = []
    inicio = time.time()

    for epoca in range(1, config.max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc, _ = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_auc)

        historico.append(
            {"epoch": epoca, "train_loss": train_loss, "val_loss": val_loss, "val_auc": val_auc}
        )
        if epoca % 10 == 0 or epoca == 1:
            logger.info(
                "Época %3d | train_loss %.4f | val_loss %.4f | val_auc %.4f",
                epoca,
                train_loss,
                val_loss,
                val_auc,
            )
        if log_mlflow and mlflow.active_run():
            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss, "val_auc": val_auc},
                step=epoca,
            )

        monitorado = val_auc if config.monitor == "val_auc" else val_loss
        if stopper.step(monitorado, model, epoca):
            break

    stopper.restore(model)
    duracao = time.time() - inicio
    logger.info(
        "Treino concluído em %.1fs | %d épocas | melhor época %d",
        duracao,
        len(historico),
        stopper.best_epoch,
    )

    # ---------------------------------- threshold (na VALIDAÇÃO) ----------
    proba_val = model.predict_proba(arrays["X_val"], device)
    threshold, _ = find_best_threshold(arrays["y_val"], proba_val, metric="f1")
    config.threshold = float(threshold)

    # ---------------------------------------- avaliação final (TESTE) -----
    proba_test = model.predict_proba(arrays["X_test"], device)
    y_pred_test = (proba_test >= threshold).astype(int)
    metricas = compute_metrics(
        arrays["y_test"], y_pred_test, proba_test, "MLP (PyTorch)"
    )
    aprovado_qualidade, checks = quality_gate(metricas)

    # -------------------------------------------------------- fairness ----
    sensiveis = splits.X_test[
        [c for c in SENSITIVE_FEATURES if c in splits.X_test.columns]
    ]
    relatorio_fairness = fairness_report(arrays["y_test"], y_pred_test, sensiveis)
    aprovado_fairness, disparidades = fairness_gate(relatorio_fairness)

    # ------------------------------------------------------ persistência ---
    save_preprocessor(preprocessor)
    metadata = {
        "metrics_test": metricas,
        "best_epoch": stopper.best_epoch,
        "epochs_run": len(historico),
        "training_seconds": round(duracao, 2),
        "quality_gate_passed": aprovado_qualidade,
        "fairness_gate_passed": aprovado_fairness,
        "n_features": int(config.input_dim),
        "train_size": int(len(arrays["y_train"])),
        "dataset_version": "v1.0",
    }
    save_model(model, metadata)

    # ----------------------------------------------------------- MLflow ---
    if log_mlflow and mlflow.active_run():
        mlflow.log_params(config.to_dict())
        mlflow.log_param("model_type", "MLP-PyTorch")
        mlflow.log_param("optimizer", "AdamW")
        mlflow.log_param("early_stopping", True)
        mlflow.log_metrics({f"test_{k}": v for k, v in metricas.items()})
        mlflow.log_metric("best_epoch", stopper.best_epoch)
        mlflow.log_metric("epochs_run", len(historico))
        mlflow.set_tag("stage", "champion-candidate")
        mlflow.set_tag("dataset_version", "v1.0")
        mlflow.set_tag("quality_gate", "pass" if aprovado_qualidade else "fail")
        mlflow.set_tag("fairness_gate", "pass" if aprovado_fairness else "fail")

        relatorio_fairness.to_csv("/tmp/fairness_report.csv", index=False)
        mlflow.log_artifact("/tmp/fairness_report.csv")
        modelo_cpu = model.to("cpu")
        exemplo = torch.from_numpy(arrays["X_val"][:5])
        mlflow.pytorch.log_model(
            modelo_cpu, name="model", input_example=exemplo, serialization_format="pickle"
        )
        model.to(device)

    return {
        "model": model,
        "preprocessor": preprocessor,
        "config": config,
        "metrics": metricas,
        "history": historico,
        "fairness": relatorio_fairness,
        "fairness_disparity": disparidades,
        "quality_checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina a MLP de churn.")
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument(
        "--hidden", type=int, nargs="+", default=[128, 64, 32], help="ex: --hidden 256 128"
    )
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    config = MLPConfig(
        input_dim=0,  # definido após o pré-processamento
        hidden_dims=args.hidden,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )

    if args.no_mlflow:
        train(config, log_mlflow=False)
        return

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="mlp-pytorch-early-stopping"):
        train(config, log_mlflow=True)


if __name__ == "__main__":
    main()
