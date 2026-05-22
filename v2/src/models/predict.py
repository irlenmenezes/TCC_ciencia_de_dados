"""Wrapper de inferência usado pelo dashboard."""
from __future__ import annotations

from functools import lru_cache

import joblib
import pandas as pd

from src.utils.paths import MODELS_DIR


@lru_cache(maxsize=1)
def load_classifier():
    path = MODELS_DIR / "clf_gravidade.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} não existe. Rode `python -m src.models.train_classification`."
        )
    return joblib.load(path)


def predict_gravidade(payload: dict) -> dict:
    """Recebe um dict com os campos do acidente e retorna proba por classe."""
    pipe = load_classifier()
    df = pd.DataFrame([payload])
    probs = pipe.predict_proba(df)[0]
    classes = pipe.classes_
    return {cls: float(p) for cls, p in zip(classes, probs)}


@lru_cache(maxsize=1)
def load_forecast_brasil() -> pd.DataFrame:
    path = MODELS_DIR / "forecast_brasil.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} não existe. Rode `python -m src.models.train_forecast`."
        )
    return pd.read_parquet(path)
