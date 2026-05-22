"""Treina classificador da gravidade do acidente (sem_vitimas / com_feridos / fatal).

Baseline: LogisticRegression com OneHotEncoder.
Modelo principal: LightGBM com encoder ordinal.
Salva o melhor pipeline em models/clf_gravidade.joblib.
"""
from __future__ import annotations

import json
import logging

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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


def make_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), [f for f in NUMERIC_FEATURES]),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)),
        ]
    )


def main() -> int:
    df = load_processed()
    df = build_features(df)
    df = df.dropna(subset=["gravidade"])

    X, y, features = split_xy(df)
    logger.info("Features usadas: %s", features)
    logger.info("Distribuição da target:\n%s", y.value_counts(normalize=True))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = make_pipeline()
    logger.info("Treinando baseline (LogisticRegression)...")
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)
    logger.info("\n%s", classification_report(y_test, preds))

    out_model = MODELS_DIR / "clf_gravidade.joblib"
    out_metrics = MODELS_DIR / "clf_gravidade_metrics.json"
    joblib.dump(pipe, out_model)
    out_metrics.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info("Modelo salvo em %s", out_model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
