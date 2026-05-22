"""Limpa e consolida os CSVs brutos da PRF em um único parquet.

Reproduz e moderniza o fluxo de limpeza do notebook original do TCC:
- normaliza tipos (datas, números)
- corrige encoding e strings
- remove duplicatas
- gera coluna de gravidade categórica (target do modelo de classificação)
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.utils.paths import DATA_PROCESSED, DATA_RAW

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

NUMERIC_COLS = ["km", "pessoas", "mortos", "feridos_leves", "feridos_graves", "ilesos",
                "ignorados", "feridos", "veiculos", "latitude", "longitude"]


def _read_prf_csv(path: Path) -> pd.DataFrame:
    """PRF publica CSVs em latin-1 com separador ';' e vírgula como decimal."""
    return pd.read_csv(
        path,
        sep=";",
        encoding="latin-1",
        decimal=",",
        low_memory=False,
    )


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    if "data_inversa" in df.columns:
        df["data"] = pd.to_datetime(df["data_inversa"], errors="coerce")
    elif "data" in df.columns:
        df["data"] = pd.to_datetime(df["data"], errors="coerce")

    if "horario" in df.columns:
        df["hora"] = pd.to_datetime(df["horario"], format="%H:%M:%S", errors="coerce").dt.hour

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().str.lower()

    return df


def _add_gravidade(df: pd.DataFrame) -> pd.DataFrame:
    """Cria coluna 'gravidade' a partir de mortos/feridos.

    - fatal: pelo menos 1 morto
    - com_feridos: sem mortos, mas com feridos (leves ou graves)
    - sem_vitimas: ninguém ferido nem morto
    """
    df = df.copy()
    mortos = df.get("mortos", 0).fillna(0)
    feridos_leves = df.get("feridos_leves", 0).fillna(0)
    feridos_graves = df.get("feridos_graves", 0).fillna(0)
    feridos_total = feridos_leves + feridos_graves

    df["gravidade"] = "sem_vitimas"
    df.loc[feridos_total > 0, "gravidade"] = "com_feridos"
    df.loc[mortos > 0, "gravidade"] = "fatal"
    return df


def clean_year(path: Path) -> pd.DataFrame:
    logger.info("Limpando %s", path.name)
    df = _read_prf_csv(path)
    df = _normalize(df)
    df = _add_gravidade(df)
    df = df.drop_duplicates()
    return df


def main() -> int:
    csvs = sorted(DATA_RAW.glob("datatran*.csv"))
    if not csvs:
        logger.error("Nenhum CSV em %s. Rode `python -m src.data.ingest` primeiro.", DATA_RAW)
        return 1

    frames = [clean_year(p) for p in csvs]
    consolidated = pd.concat(frames, ignore_index=True)

    out = DATA_PROCESSED / "acidentes.parquet"
    consolidated.to_parquet(out, index=False)
    logger.info("Gravado %s (%d linhas)", out, len(consolidated))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
