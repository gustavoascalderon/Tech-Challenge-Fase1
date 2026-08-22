"""MLP (Multi-Layer Perceptron) em PyTorch para predição de churn.

Arquitetura: rede densa totalmente conectada com BatchNorm + Dropout,
saída de 1 logit (classificação binária). O desbalanceamento (~27% de churn)
é tratado via `pos_weight` no BCEWithLogitsLoss, equivalente ao
`class_weight='balanced'` dos baselines sklearn.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from churn.config import MLP_METADATA_PATH, MLP_WEIGHTS_PATH, SEED

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ config ---
@dataclass
class MLPConfig:
    """Hiperparâmetros da MLP. Serializado junto com os pesos."""

    input_dim: int
    hidden_dims: list[int] = field(default_factory=lambda: [128, 64, 32])
    dropout: float = 0.3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 200
    # Early Stopping
    patience: int = 15
    min_delta: float = 1e-4
    monitor: str = "val_auc"  # 'val_auc' (maximiza) ou 'val_loss' (minimiza)
    threshold: float = 0.5  # ajustado depois pela busca de threshold
    seed: int = SEED

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------- reprodutib. ---
def set_seed(seed: int = SEED) -> None:
    """Fixa todas as fontes de aleatoriedade do PyTorch."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------- o modelo ---
class ChurnMLP(nn.Module):
    """MLP densa: [Linear -> BatchNorm -> ReLU -> Dropout] x N -> Linear(1).

    Retorna **logits** (sem sigmoid) — a sigmoid fica no BCEWithLogitsLoss,
    que é numericamente mais estável.
    """

    def __init__(self, config: MLPConfig):
        super().__init__()
        self.config = config

        camadas: list[nn.Module] = []
        dim_entrada = config.input_dim
        for dim_saida in config.hidden_dims:
            camadas += [
                nn.Linear(dim_entrada, dim_saida),
                nn.BatchNorm1d(dim_saida),
                nn.ReLU(),
                nn.Dropout(config.dropout),
            ]
            dim_entrada = dim_saida
        camadas.append(nn.Linear(dim_entrada, 1))

        self.network = nn.Sequential(*camadas)
        self._init_weights()

    def _init_weights(self) -> None:
        """Inicialização He — apropriada para ativações ReLU."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray, device: torch.device | None = None) -> np.ndarray:
        """Probabilidades de churn para um array já pré-processado."""
        device = device or next(self.parameters()).device
        self.eval()
        tensor = torch.as_tensor(np.asarray(X, dtype=np.float32), device=device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return torch.sigmoid(self(tensor)).cpu().numpy()

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# --------------------------------------------------------- early stopping ----
class EarlyStopping:
    """Interrompe o treino quando a métrica monitorada para de melhorar.

    Guarda em memória os pesos da melhor época e os restaura no fim
    (`restore_best_weights`), de modo que o modelo final nunca é o da última
    época — é o da melhor época de validação.
    """

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 1e-4,
        mode: str = "max",
        restore_best_weights: bool = True,
    ):
        if mode not in {"max", "min"}:
            raise ValueError("mode deve ser 'max' ou 'min'")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights

        self.best_score: float | None = None
        self.best_epoch: int = 0
        self.counter: int = 0
        self.should_stop: bool = False
        self._best_state: dict | None = None

    def _melhorou(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "max":
            return score > self.best_score + self.min_delta
        return score < self.best_score - self.min_delta

    def step(self, score: float, model: nn.Module, epoch: int) -> bool:
        """Avalia a época. Retorna True se o treino deve parar."""
        if self._melhorou(score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.restore_best_weights:
                self._best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    "Early stopping na época %d | melhor época: %d (%s=%.4f)",
                    epoch,
                    self.best_epoch,
                    self.mode,
                    self.best_score,
                )
        return self.should_stop

    def restore(self, model: nn.Module) -> None:
        """Devolve ao modelo os pesos da melhor época."""
        if self.restore_best_weights and self._best_state is not None:
            model.load_state_dict(self._best_state)
            logger.info("Pesos da época %d restaurados.", self.best_epoch)


# ------------------------------------------------------------ persistência ---
def save_model(
    model: ChurnMLP,
    metadata: dict,
    weights_path: Path | str | None = None,
    metadata_path: Path | str | None = None,
) -> None:
    """Salva pesos (.pt) e metadados (.json) separadamente.

    Só o `state_dict` é salvo — nunca o objeto pickled inteiro — para que o
    carregamento não dependa da versão exata do código.
    """
    weights_path = Path(weights_path or MLP_WEIGHTS_PATH)
    metadata_path = Path(metadata_path or MLP_METADATA_PATH)
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), weights_path)
    payload = {"config": model.config.to_dict(), **metadata}
    metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.info("Modelo salvo em %s (+ metadados em %s)", weights_path, metadata_path)


def load_model(
    weights_path: Path | str | None = None,
    metadata_path: Path | str | None = None,
    device: torch.device | None = None,
) -> tuple[ChurnMLP, dict]:
    """Recarrega o modelo treinado para inferência (usado pela API)."""
    weights_path = Path(weights_path or MLP_WEIGHTS_PATH)
    metadata_path = Path(metadata_path or MLP_METADATA_PATH)
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Artefatos da MLP não encontrados ({weights_path}). "
            "Rode `python -m churn.models.train_mlp` primeiro."
        )

    metadata = json.loads(metadata_path.read_text())
    config = MLPConfig(**metadata["config"])
    device = device or get_device()

    model = ChurnMLP(config).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    logger.info("Modelo carregado | %d parâmetros", model.n_parameters())
    return model, metadata
