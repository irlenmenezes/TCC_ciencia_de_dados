"""Previsão de quantidade de acidentes por mês (Brasil e por UF).

Modelos:
- SARIMA (statsmodels) — baseline interpretável.
- Prophet — sazonalidade e feriados automáticos.

Salva forecasts em models/forecast_brasil.parquet e por_uf.parquet.
"""
from __future__ import annotations

import logging

import pandas as pd

try:
    from prophet import Prophet  # type: ignore
except ImportError:  # pragma: no cover
    Prophet = None  # type: ignore

from src.features.build import load_processed
from src.utils.paths import MODELS_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

HORIZON_MONTHS = 6


def monthly_counts(df: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Agrega contagem mensal de acidentes (opcionalmente por UF)."""
    df = df.dropna(subset=["data"]).copy()
    df["mes_ref"] = df["data"].dt.to_period("M").dt.to_timestamp()

    group_cols = ["mes_ref"] + ([by] if by else [])
    counts = df.groupby(group_cols).size().reset_index(name="acidentes")
    return counts


def forecast_prophet(series: pd.DataFrame, periods: int = HORIZON_MONTHS) -> pd.DataFrame:
    if Prophet is None:
        raise ImportError("prophet não instalado. pip install prophet")

    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    df = series.rename(columns={"mes_ref": "ds", "acidentes": "y"})
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq="MS")
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def main() -> int:
    df = load_processed()
    brasil = monthly_counts(df)
    logger.info("Série mensal Brasil: %d pontos", len(brasil))

    forecast_br = forecast_prophet(brasil)
    out_br = MODELS_DIR / "forecast_brasil.parquet"
    forecast_br.to_parquet(out_br, index=False)
    logger.info("Forecast Brasil salvo em %s", out_br)

    por_uf = monthly_counts(df, by="uf")
    pieces = []
    for uf, grupo in por_uf.groupby("uf"):
        if len(grupo) < 24:
            continue  # série muito curta
        try:
            fc = forecast_prophet(grupo[["mes_ref", "acidentes"]])
            fc["uf"] = uf
            pieces.append(fc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Falha em UF=%s: %s", uf, exc)

    if pieces:
        forecast_uf = pd.concat(pieces, ignore_index=True)
        out_uf = MODELS_DIR / "forecast_por_uf.parquet"
        forecast_uf.to_parquet(out_uf, index=False)
        logger.info("Forecast por UF salvo em %s", out_uf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
