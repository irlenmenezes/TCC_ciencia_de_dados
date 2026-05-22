"""Treina classificador da gravidade do acidente (sem_vitimas / com_feridos / fatal).

Compara dois modelos:
- Baseline: LogisticRegression com OneHotEncoder (rápido, interpretável).
- Principal: LightGBM com OrdinalEncoder (melhor em dados tabulares mistos).

Salva:
- models/clf_gravidade.joblib       → pipeline campeão (LightGBM)
- models/clf_baseline.joblib        → baseline (para comparação)
- models/clf_gravidade_metrics.json → métricas dos dois modelos
- models/clf_feature_importance.csv → importância por feature (LightGBM)
"""
from __future__ import annotations

import json
import logging

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.features.build import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_features,
    load_processed,
    split_xy,
)
from src.utils.paths import MODELS_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

RANDOM_STATE = 42


def make_baseline(numeric: list[str], categorical: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), categorical),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocess", pre),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)),
        ]
    )


def make_lightgbm(numeric: list[str], categorical: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical,
            ),
        ],
        remainder="drop",
    )
    clf = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocess", pre), ("clf", clf)])


def _evaluate(name: str, pipe: Pipeline, X_test, y_test) -> dict:
    preds = pipe.predict(X_test)
    logger.info("\n=== %s ===\n%s", name, classification_report(y_test, preds))
    return {
        "model": name,
        "f1_macro": f1_score(y_test, preds, average="macro"),
        "f1_weighted": f1_score(y_test, preds, average="weighted"),
        "report": classification_report(y_test, preds, output_dict=True),
    }


def _feature_importance(pipe: Pipeline, features: list[str]) -> pd.DataFrame:
    clf: LGBMClassifier = pipe.named_steps["clf"]
    importances = clf.feature_importances_
    return (
        pd.DataFrame({"feature": features, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def main() -> int:
    df = load_processed()
    df = build_features(df)
    df = df.dropna(subset=["gravidade"])

    X, y, features = split_xy(df)
    numeric = [c for c in NUMERIC_FEATURES if c in features]
    categorical = [c for c in CATEGORICAL_FEATURES if c in features]
    logger.info("Numéricas: %s", numeric)
    logger.info("Categóricas: %s", categorical)
    logger.info("Distribuição da target:\n%s", y.value_counts(normalize=True))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    baseline = make_baseline(numeric, categorical)
    logger.info("Treinando baseline (LogisticRegression)...")
    baseline.fit(X_train, y_train)
    base_metrics = _evaluate("baseline", baseline, X_test, y_test)

    main_model = make_lightgbm(numeric, categorical)
    logger.info("Treinando LightGBM...")
    main_model.fit(X_train, y_train)
    main_metrics = _evaluate("lightgbm", main_model, X_test, y_test)

    champion = main_model if main_metrics["f1_macro"] >= base_metrics["f1_macro"] else baseline
    champion_name = "lightgbm" if champion is main_model else "baseline"
    logger.info("Modelo campeão: %s", champion_name)

    joblib.dump(champion, MODELS_DIR / "clf_gravidade.joblib")
    joblib.dump(baseline, MODELS_DIR / "clf_baseline.joblib")

    metrics_path = MODELS_DIR / "clf_gravidade_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {"champion": champion_name, "baseline": base_metrics, "lightgbm": main_metrics},
            indent=2,
            ensure_ascii=False,
            default=lambda o: float(o) if isinstance(o, np.floating) else str(o),
        )
    )

    fi = _feature_importance(main_model, numeric + categorical)
    fi.to_csv(MODELS_DIR / "clf_feature_importance.csv", index=False)
    logger.info("Top 10 features:\n%s", fi.head(10))
    logger.info("Modelo salvo em %s", MODELS_DIR / "clf_gravidade.joblib")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
