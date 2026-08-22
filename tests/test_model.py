"""Testes da MLP (PyTorch), do Early Stopping e da avaliação."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from churn.models.evaluate import (
    compute_metrics,
    fairness_gate,
    fairness_report,
    find_best_threshold,
)
from churn.models.mlp import (
    ChurnMLP,
    EarlyStopping,
    MLPConfig,
    load_model,
    save_model,
    set_seed,
)


@pytest.fixture
def config() -> MLPConfig:
    return MLPConfig(input_dim=20, hidden_dims=[16, 8], dropout=0.1)


# ------------------------------------------------------------- arquitetura ---
def test_forward_shape(config: MLPConfig):
    """Saída é um logit por amostra."""
    model = ChurnMLP(config)
    out = model(torch.randn(32, config.input_dim))
    assert out.shape == (32,)


def test_saida_e_logit_nao_probabilidade(config: MLPConfig):
    """A rede devolve logits (podem sair de [0,1]) — sigmoid fica na loss."""
    model = ChurnMLP(config)
    model.train()
    out = model(torch.randn(64, config.input_dim) * 10)
    assert out.min() < 0 or out.max() > 1


def test_predict_proba_intervalo(config: MLPConfig):
    model = ChurnMLP(config)
    probas = model.predict_proba(np.random.randn(50, config.input_dim))
    assert probas.shape == (50,)
    assert ((probas >= 0) & (probas <= 1)).all()


def test_predict_proba_amostra_unica(config: MLPConfig):
    """A API envia 1 cliente por vez — BatchNorm em eval não pode quebrar."""
    model = ChurnMLP(config)
    assert model.predict_proba(np.random.randn(config.input_dim)).shape == (1,)


def test_camadas_respeitam_config():
    model = ChurnMLP(MLPConfig(input_dim=10, hidden_dims=[64, 32, 16]))
    lineares = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    assert [m.out_features for m in lineares] == [64, 32, 16, 1]


def test_dropout_e_batchnorm_presentes(config: MLPConfig):
    tipos = {type(m) for m in ChurnMLP(config).modules()}
    assert torch.nn.Dropout in tipos and torch.nn.BatchNorm1d in tipos


def test_gradientes_fluem(config: MLPConfig):
    model = ChurnMLP(config)
    loss = torch.nn.BCEWithLogitsLoss()(
        model(torch.randn(16, config.input_dim)), torch.randint(0, 2, (16,)).float()
    )
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
        if p.requires_grad
    )


def test_reprodutibilidade(config: MLPConfig):
    set_seed(42)
    a = ChurnMLP(config).predict_proba(np.ones((5, config.input_dim), dtype=np.float32))
    set_seed(42)
    b = ChurnMLP(config).predict_proba(np.ones((5, config.input_dim), dtype=np.float32))
    assert np.allclose(a, b)


# ---------------------------------------------------------- early stopping ---
def test_early_stopping_dispara_apos_paciencia(config: MLPConfig):
    model = ChurnMLP(config)
    stopper = EarlyStopping(patience=3, mode="max")
    assert not stopper.step(0.80, model, 1)
    for epoca in range(2, 5):  # três épocas sem melhora
        parou = stopper.step(0.70, model, epoca)
    assert parou and stopper.should_stop
    assert stopper.best_epoch == 1


def test_early_stopping_reseta_contador_ao_melhorar(config: MLPConfig):
    model = ChurnMLP(config)
    stopper = EarlyStopping(patience=2, mode="max")
    stopper.step(0.80, model, 1)
    stopper.step(0.70, model, 2)
    stopper.step(0.90, model, 3)  # melhorou
    assert stopper.counter == 0 and stopper.best_epoch == 3


def test_early_stopping_restaura_melhores_pesos(config: MLPConfig):
    """O modelo final é o da melhor época, não o da última."""
    model = ChurnMLP(config)
    stopper = EarlyStopping(patience=2, mode="max")
    stopper.step(0.90, model, 1)
    melhor = model.network[0].weight.detach().clone()

    with torch.no_grad():  # degrada os pesos de propósito
        model.network[0].weight.add_(5.0)
    stopper.step(0.50, model, 2)
    stopper.step(0.50, model, 3)
    stopper.restore(model)

    assert torch.allclose(model.network[0].weight, melhor)


def test_early_stopping_modo_min(config: MLPConfig):
    """Em mode='min' (val_loss), queda conta como melhora."""
    model = ChurnMLP(config)
    stopper = EarlyStopping(patience=2, mode="min")
    stopper.step(1.0, model, 1)
    stopper.step(0.5, model, 2)
    assert stopper.best_epoch == 2 and stopper.counter == 0


def test_early_stopping_min_delta(config: MLPConfig):
    """Melhora menor que min_delta não conta como melhora."""
    model = ChurnMLP(config)
    stopper = EarlyStopping(patience=5, mode="max", min_delta=0.01)
    stopper.step(0.80, model, 1)
    stopper.step(0.8001, model, 2)
    assert stopper.counter == 1 and stopper.best_epoch == 1


def test_modo_invalido():
    with pytest.raises(ValueError):
        EarlyStopping(mode="talvez")


# ------------------------------------------------------------ persistência ---
def test_save_load_roundtrip(config: MLPConfig, tmp_path):
    """Modelo recarregado produz predições idênticas — o que a API depende."""
    model = ChurnMLP(config)
    model.eval()
    X = np.random.randn(10, config.input_dim).astype(np.float32)
    antes = model.predict_proba(X)

    pesos, meta = tmp_path / "m.pt", tmp_path / "m.json"
    save_model(model, {"metrics_test": {"auc_roc": 0.86}, "best_epoch": 12}, pesos, meta)
    recarregado, metadata = load_model(pesos, meta)

    assert np.allclose(antes, recarregado.predict_proba(X), atol=1e-6)
    assert metadata["best_epoch"] == 12
    assert metadata["config"]["input_dim"] == config.input_dim


def test_load_model_ausente_falha_claro(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "nao_existe.pt", tmp_path / "nao_existe.json")


# --------------------------------------------------------------- avaliação ---
def test_compute_metrics_perfeito():
    y = np.array([0, 1, 0, 1])
    m = compute_metrics(y, y, np.array([0.1, 0.9, 0.2, 0.8]))
    assert m["recall"] == 1.0 and m["f1"] == 1.0 and m["auc_roc"] == 1.0
    assert m["fnr"] == 0.0


def test_find_best_threshold_respeita_recall_minimo():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    proba = np.clip(y * 0.4 + rng.random(500) * 0.5, 0, 1)
    t, _ = find_best_threshold(y, proba, metric="f1", min_recall=0.80)
    assert 0.0 < t < 1.0
    assert (proba >= t).astype(int)[y == 1].mean() >= 0.79


def test_fairness_report_cobre_todos_os_grupos():
    n = 200
    rng = np.random.default_rng(1)
    y_true, y_pred = rng.integers(0, 2, n), rng.integers(0, 2, n)
    sens = pd.DataFrame({"gender": rng.choice(["Male", "Female"], n)})
    rel = fairness_report(y_true, y_pred, sens, ["gender"])
    assert set(rel["grupo"]) == {"Male", "Female"}
    assert rel["fnr"].between(0, 1).all()


def test_fairness_gate_reprova_disparidade_alta():
    rel = pd.DataFrame(
        {"atributo": ["gender", "gender"], "grupo": ["M", "F"], "fnr": [0.10, 0.45]}
    )
    aprovado, disp = fairness_gate(rel, max_disparity=0.10)
    assert not aprovado
    assert disp.loc[0, "disparidade"] == pytest.approx(0.35)
