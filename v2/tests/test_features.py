"""Testes do feature engineering."""
from __future__ import annotations

import pandas as pd

from src.features.build import build_features, split_xy


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "data": pd.to_datetime(["2024-01-06", "2024-06-15", "2024-12-20"]),
            "hora": [2, 14, 22],
            "uf": ["mg", "sp", "rj"],
            "br": ["381", "116", "101"],
            "km": [100.0, 50.5, 200.0],
            "veiculos": [2, 1, 3],
            "pessoas": [3, 1, 5],
            "gravidade": ["fatal", "sem_vitimas", "com_feridos"],
        }
    )


def test_build_features_adiciona_colunas_temporais():
    out = build_features(_sample_df())
    assert {"mes", "ano", "dia_do_ano", "fim_de_semana"} <= set(out.columns)
    assert out["ano"].tolist() == [2024, 2024, 2024]
    # 2024-01-06 é sábado, 2024-06-15 é sábado, 2024-12-20 é sexta
    assert out["fim_de_semana"].tolist() == [True, True, False]


def test_build_features_cria_periodo_noturno():
    out = build_features(_sample_df())
    # hora=2 (noturno), 14 (não), 22 (noturno)
    assert out["periodo_noturno"].tolist() == [1, 0, 1]


def test_split_xy_separa_target_dos_features():
    df = build_features(_sample_df())
    X, y, features = split_xy(df, target="gravidade")
    assert "gravidade" not in X.columns
    assert y.tolist() == ["fatal", "sem_vitimas", "com_feridos"]
    assert "uf" in features and "km" in features
