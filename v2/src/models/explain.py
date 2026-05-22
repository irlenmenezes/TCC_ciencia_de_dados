"""Interpretabilidade do classificador de gravidade via SHAP.

Carrega o pipeline campeão e gera:
- models/shap_summary.png    → barra com importância média absoluta
- models/shap_beeswarm.png   → distribuição dos shap values por feature
- models/shap_values.parquet → tabela com shap values (para uso no dashboard)

Uso:
    python -m src.models.explain --sample 5000
"""
from __future__ import annotations

import argparse
import logging

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.features.build import build_features, load_processed, split_xy
from src.utils.paths import MODELS_DIR

matplotlib.use("Agg")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DEFAULT_SAMPLE = 5000


def _transform(pipe, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Aplica o preprocessador do pipeline e retorna matriz + nomes das features."""
    preproc = pipe.named_steps["preprocess"]
    X_t = preproc.transform(X)
    try:
        feature_names = list(preproc.get_feature_names_out())
    except Exception:  # noqa: BLE001
        feature_names = [f"f{i}" for i in range(X_t.shape[1])]
    return X_t, feature_names


def main() -> int:
    parser = argparse.ArgumentParser(description="Gera explicações SHAP do classificador.")
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE,
                        help="Tamanho da amostra usada para o explainer (default 5000)")
    args = parser.parse_args()

    model_path = MODELS_DIR / "clf_gravidade.joblib"
    if not model_path.exists():
        logger.error("%s não existe. Rode train_classification primeiro.", model_path)
        return 1

    pipe = joblib.load(model_path)
    clf = pipe.named_steps["clf"]
    if not hasattr(clf, "booster_"):
        logger.error("Explicação SHAP implementada apenas para o LightGBM. Campeão atual: %s",
                     type(clf).__name__)
        return 1

    df = build_features(load_processed()).dropna(subset=["gravidade"])
    sample = df.sample(n=min(args.sample, len(df)), random_state=42)
    X, _, _ = split_xy(sample)
    X_t, feature_names = _transform(pipe, X)

    logger.info("Calculando SHAP values em %d amostras × %d features ...",
                X_t.shape[0], X_t.shape[1])
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_t)

    # Para classificação multiclasse, shap_values é lista por classe.
    if isinstance(shap_values, list):
        mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        mean_abs = np.abs(shap_values).mean(axis=0)

    summary = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    summary.to_parquet(MODELS_DIR / "shap_values.parquet", index=False)

    fig, ax = plt.subplots(figsize=(8, max(4, len(summary.head(20)) * 0.3)))
    top = summary.head(20).iloc[::-1]
    ax.barh(top["feature"], top["mean_abs_shap"])
    ax.set_xlabel("|SHAP| médio")
    ax.set_title("Top 20 features mais influentes (multiclasse)")
    fig.tight_layout()
    fig.savefig(MODELS_DIR / "shap_summary.png", dpi=120)
    plt.close(fig)
    logger.info("shap_summary.png salvo")

    try:
        # beeswarm da classe 'fatal' (mais informativo para o problema).
        plt.figure()
        if isinstance(shap_values, list):
            classes = list(getattr(clf, "classes_", []))
            idx = classes.index("fatal") if "fatal" in classes else 0
            shap.summary_plot(shap_values[idx], X_t, feature_names=feature_names, show=False)
        else:
            shap.summary_plot(shap_values, X_t, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(MODELS_DIR / "shap_beeswarm.png", dpi=120, bbox_inches="tight")
        plt.close()
        logger.info("shap_beeswarm.png salvo")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Falha ao gerar beeswarm: %s", exc)

    logger.info("Top 10 features por |SHAP| médio:\n%s", summary.head(10))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
