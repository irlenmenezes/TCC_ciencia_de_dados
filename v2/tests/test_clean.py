"""Testes da limpeza de dados."""
from __future__ import annotations

import pandas as pd

from src.data.clean import _add_gravidade, _normalize


def test_gravidade_fatal_quando_ha_mortos():
    df = pd.DataFrame({"mortos": [1], "feridos_leves": [0], "feridos_graves": [0]})
    out = _add_gravidade(df)
    assert out.loc[0, "gravidade"] == "fatal"


def test_gravidade_com_feridos_sem_mortos():
    df = pd.DataFrame({"mortos": [0], "feridos_leves": [2], "feridos_graves": [0]})
    assert _add_gravidade(df).loc[0, "gravidade"] == "com_feridos"


def test_gravidade_sem_vitimas():
    df = pd.DataFrame({"mortos": [0], "feridos_leves": [0], "feridos_graves": [0]})
    assert _add_gravidade(df).loc[0, "gravidade"] == "sem_vitimas"


def test_normalize_baixa_caixa_e_lida_com_datas():
    df = pd.DataFrame({
        "DATA_INVERSA": ["2024-01-15"],
        "UF": ["MG"],
        "horario": ["13:45:00"],
        "km": ["123,5"],
    })
    out = _normalize(df)
    assert "data" in out.columns
    assert out.loc[0, "uf"] == "mg"
    assert out.loc[0, "hora"] == 13
    assert out.loc[0, "km"] == 123.5
