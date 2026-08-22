from __future__ import annotations

import argparse
import logging
from pathlib import Path

from churn.config import GATE_FNR_DISPARITY, SENSITIVE_FEATURES
from churn.data.load import load_clean
from churn.data.validate import validate
from churn.features.preprocess import load_preprocessor, split_data
from churn.models.evaluate import fairness_gate, fairness_report
from churn.models.mlp import get_device, load_model

logger = logging.getLogger(__name__)


def audit() -> tuple:
    """Roda a auditoria sobre o conjunto de teste. Retorna (relatório, disparidades)."""
    df = validate(load_clean())
    splits = split_data(df)

    device = get_device()
    model, metadata = load_model(device=device)
    preprocessor = load_preprocessor()

    threshold = metadata["config"]["threshold"]
    X_test = preprocessor.transform(splits.X_test)
    proba = model.predict_proba(X_test, device)
    y_pred = (proba >= threshold).astype(int)
    y_true = splits.y_test.to_numpy()

    presentes = [f for f in SENSITIVE_FEATURES if f in splits.X_test.columns]
    relatorio = fairness_report(y_true, y_pred, splits.X_test[presentes], presentes)
    aprovado, disparidades = fairness_gate(relatorio)

    return relatorio, disparidades, aprovado, threshold, metadata


def to_markdown(relatorio, disparidades) -> str:
    """Formata o relatório como markdown, pronto para colar no Model Card."""
    linhas = ["### Disparidade por atributo sensível", ""]
    linhas.append("| Atributo | Grupo | n | Taxa churn real | Recall | FNR |")
    linhas.append("|---|---|---|---|---|---|")
    for _, r in relatorio.iterrows():
        linhas.append(
            f"| `{r['atributo']}` | {r['grupo']} | {r['n']} | "
            f"{r['taxa_churn_real']:.3f} | {r['recall']:.3f} | {r['fnr']:.3f} |"
        )

    linhas += ["", "### Gate de fairness (disparidade de FNR)", ""]
    linhas.append("| Atributo | FNR mín. | FNR máx. | Disparidade | Situação |")
    linhas.append("|---|---|---|---|---|")
    for _, r in disparidades.iterrows():
        status = "aprovado" if r["aprovado"] else "**reprovado**"
        linhas.append(
            f"| `{r['atributo']}` | {r['fnr_min']:.3f} | {r['fnr_max']:.3f} | "
            f"{r['disparidade']:.3f} | {status} |"
        )
    linhas += ["", f"Limite adotado: disparidade de FNR ≤ {GATE_FNR_DISPARITY:.2f}."]
    return "\n".join(linhas)


def main() -> None:
    parser = argparse.ArgumentParser(description="Auditoria de fairness do modelo.")
    parser.add_argument(
        "--markdown", action="store_true", help="saída em markdown para o Model Card"
    )
    parser.add_argument("--csv", type=str, default=None, help="salva o relatório em CSV")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.markdown else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        force=True,
    )

    relatorio, disparidades, aprovado, threshold, metadata = audit()

    if args.markdown:
        print(to_markdown(relatorio, disparidades))
    else:
        print(f"\nThreshold aplicado: {threshold:.2f} | melhor época: {metadata['best_epoch']}")
        print("\n--- Métricas por grupo ---")
        print(relatorio.round(3).to_string(index=False))
        print("\n--- Gate de fairness ---")
        print(disparidades.round(3).to_string(index=False))
        print(f"\nResultado geral: {'APROVADO' if aprovado else 'REPROVADO'}")

        piores = disparidades.loc[~disparidades["aprovado"], "atributo"].tolist()
        if piores:
            print(
                "\nAtributos acima do limite: "
                + ", ".join(piores)
                + "\nGrupos com n < 100 têm intervalo de confiança largo — "
                "reporte a disparidade como indicativa, não conclusiva."
            )

    if args.csv:
        destino = Path(args.csv)
        destino.parent.mkdir(parents=True, exist_ok=True)
        relatorio.to_csv(destino, index=False)
        print(f"\nRelatório salvo em {destino}")


if __name__ == "__main__":
    main()