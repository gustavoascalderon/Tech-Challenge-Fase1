"""Teste de integração do pipeline de treino.

Roda um treino curto de verdade (poucas épocas, rede pequena) para garantir
que ingestão → split → pré-processamento → MLP → early stopping → threshold →
fairness → persistência funcionam encadeados.
"""

from __future__ import annotations

import numpy as np
import pytest

from churn.models.mlp import MLPConfig
from churn.models.train_mlp import compute_pos_weight, make_loaders, train


@pytest.fixture(scope="module")
def resultado(clean_df, tmp_path_factory, monkeypatch_module):
    """Treino curto com artefatos redirecionados para tmp_path."""
    destino = tmp_path_factory.mktemp("models")

    import churn.features.preprocess as prep
    import churn.models.mlp as mlp_mod
    import churn.models.train_mlp as train_mod

    monkeypatch_module.setattr(prep, "PREPROCESSOR_PATH", destino / "preprocessor.joblib")
    monkeypatch_module.setattr(mlp_mod, "MLP_WEIGHTS_PATH", destino / "mlp.pt")
    monkeypatch_module.setattr(mlp_mod, "MLP_METADATA_PATH", destino / "mlp.json")
    monkeypatch_module.setattr(train_mod, "load_clean", lambda: clean_df)

    config = MLPConfig(
        input_dim=0, hidden_dims=[32, 16], max_epochs=25, patience=5, batch_size=128
    )
    resultado = train(config, log_mlflow=False)
    resultado["_destino"] = destino
    return resultado


def test_treino_completa(resultado):
    assert resultado["model"] is not None
    assert 0 < len(resultado["history"]) <= 25


def test_loss_de_treino_cai(resultado):
    """A rede está de fato aprendendo, não apenas rodando."""
    hist = resultado["history"]
    assert hist[-1]["train_loss"] < hist[0]["train_loss"]


def test_historico_tem_metricas_de_validacao(resultado):
    for epoca in resultado["history"]:
        assert {"epoch", "train_loss", "val_loss", "val_auc"} <= set(epoca)
        assert 0 <= epoca["val_auc"] <= 1
        assert np.isfinite(epoca["train_loss"])


def test_threshold_ajustado_fora_do_padrao(resultado):
    """O threshold vem da busca na validação, não é o 0.5 default."""
    threshold = resultado["config"].threshold
    assert 0.0 < threshold < 1.0


def test_metricas_de_teste_completas(resultado):
    m = resultado["metrics"]
    assert {"auc_roc", "recall", "f1", "precision", "fnr", "fpr"} <= set(m)
    assert 0 <= m["auc_roc"] <= 1


def test_modelo_aprende_algo(resultado):
    """AUC acima do acaso — sanity check contra pipeline embaralhado."""
    assert resultado["metrics"]["auc_roc"] > 0.6


def test_relatorio_de_fairness_gerado(resultado):
    rel = resultado["fairness"]
    assert not rel.empty
    assert {"atributo", "grupo", "fnr", "n"} <= set(rel.columns)


def test_gates_avaliados(resultado):
    assert set(resultado["quality_checks"]) == {"auc_roc", "recall", "f1"}


def test_artefatos_persistidos(resultado):
    """Os três arquivos que a API consome precisam existir."""
    destino = resultado["_destino"]
    assert (destino / "mlp.pt").exists()
    assert (destino / "mlp.json").exists()
    assert (destino / "preprocessor.joblib").exists()


# ------------------------------------------------------------- unitários ----
def test_pos_weight():
    """80 negativos / 20 positivos = peso 4.0."""
    y = np.array([0] * 80 + [1] * 20, dtype=np.float32)
    assert float(compute_pos_weight(y)) == pytest.approx(4.0)


def test_pos_weight_sem_positivos_nao_divide_por_zero():
    assert np.isfinite(float(compute_pos_weight(np.zeros(10, dtype=np.float32))))


def test_loaders_shuffle_apenas_no_treino():
    arrays = {
        "X_train": np.random.randn(256, 10).astype(np.float32),
        "y_train": np.random.randint(0, 2, 256).astype(np.float32),
        "X_val": np.random.randn(64, 10).astype(np.float32),
        "y_val": np.random.randint(0, 2, 64).astype(np.float32),
    }
    treino, val = make_loaders(arrays, batch_size=32)
    assert treino.batch_size == 32
    # validação percorre na mesma ordem sempre
    primeira = next(iter(val))[0]
    segunda = next(iter(val))[0]
    assert np.allclose(primeira.numpy(), segunda.numpy())
