"""Feature engineering para os modelos de ML."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.utils.paths import DATA_PROCESSED

logger = logging.getLogger(__name__)

CATEGORICAL_FEATURES = [
    "uf", "br", "dia_semana", "fase_dia", "condicao_metereologica",
    "tipo_pista", "tracado_via", "uso_solo", "causa_acidente", "tipo_acidente",
]
NUMERIC_FEATURES = ["km", "veiculos", "pessoas", "hora", "mes"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "data" in df.columns:
        df["mes"] = df["data"].dt.month
        df["ano"] = df["data"].dt.year
        df["dia_do_ano"] = df["data"].dt.dayofyear
        df["fim_de_semana"] = df["data"].dt.weekday >= 5

    if "hora" in df.columns:
        df["periodo_noturno"] = ((df["hora"] >= 19) | (df["hora"] <= 5)).astype(int)
        df["hora_sin"] = np.sin(2 * np.pi * df["hora"] / 24)
        df["hora_cos"] = np.cos(2 * np.pi * df["hora"] / 24)

    return df


def split_xy(df: pd.DataFrame, target: str = "gravidade"):
    features = [c for c in CATEGORICAL_FEATURES + NUMERIC_FEATURES if c in df.columns]
    X = df[features].copy()
    y = df[target].copy()
    return X, y, features


def load_processed() -> pd.DataFrame:
    path = DATA_PROCESSED / "acidentes.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} não existe. Rode `python -m src.data.clean` primeiro."
        )
    return pd.read_parquet(path)
