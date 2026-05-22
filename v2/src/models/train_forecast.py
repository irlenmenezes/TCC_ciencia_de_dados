"""Previsão de acidentes mensais por Brasil e por UF.

Compara dois modelos:
- SARIMA (statsmodels) — baseline interpretável, sazonalidade explícita.
- Prophet — sazonalidade e mudanças de tendência automáticas.

Avaliação: hold-out das últimas N_TEST_MONTHS observações (MAE, MAPE).
Salva forecasts em models/forecast_brasil.parquet e forecast_por_uf.parquet,
além de models/forecast_metrics.json com a comparação.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    from prophet import Prophet  # type: ignore
except ImportError:  # pragma: no cover
    Prophet = None  # type: ignore

from src.features.build import load_processed
from src.utils.paths import MODELS_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

HORIZON_MONTHS = 6
N_TEST_MONTHS = 6


def monthly_counts(df: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    df = df.dropna(subset=["data"]).copy()
    df["mes_ref"] = df["data"].dt.to_period("M").dt.to_timestamp()
    group_cols = ["mes_ref"] + ([by] if by else [])
    return df.groupby(group_cols).size().reset_index(name="acidentes")


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def fit_sarima(series: pd.Series, horizon: int) -> pd.DataFrame:
    model = SARIMAX(
        series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    forecast = fitted.get_forecast(steps=horizon)
    summary = forecast.summary_frame()
    summary = summary.rename(
        columns={"mean": "yhat", "mean_ci_lower": "yhat_lower", "mean_ci_upper": "yhat_upper"}
    )
    summary["ds"] = summary.index
    return summary[["ds", "yhat", "yhat_lower", "yhat_upper"]].reset_index(drop=True)


def fit_prophet(series: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if Prophet is None:
        raise ImportError("prophet não instalado. pip install prophet")
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    df = series.rename(columns={"mes_ref": "ds", "acidentes": "y"})
    model.fit(df)
    future = model.make_future_dataframe(periods=horizon, freq="MS")
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def evaluate_models(series: pd.DataFrame, n_test: int = N_TEST_MONTHS) -> dict:
    """Treina nos primeiros N - n_test pontos e avalia nos últimos n_test."""
    if len(series) <= n_test + 6:
        return {"skipped": True, "reason": "série muito curta"}

    train = series.iloc[:-n_test].copy()
    test = series.iloc[-n_test:].copy()
    y_true = test["acidentes"].to_numpy()

    metrics: dict = {}

    try:
        sarima_fc = fit_sarima(train.set_index("mes_ref")["acidentes"], horizon=n_test)
        sarima_pred = sarima_fc["yhat"].to_numpy()
        metrics["sarima"] = {"mae": _mae(y_true, sarima_pred), "mape": _mape(y_true, sarima_pred)}
    except Exception as exc:  # noqa: BLE001
        logger.warning("SARIMA falhou: %s", exc)
        metrics["sarima"] = {"error": str(exc)}

    try:
        prophet_fc = fit_prophet(train, horizon=n_test).tail(n_test)
        prophet_pred = prophet_fc["yhat"].to_numpy()
        metrics["prophet"] = {"mae": _mae(y_true, prophet_pred), "mape": _mape(y_true, prophet_pred)}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Prophet falhou: %s", exc)
        metrics["prophet"] = {"error": str(exc)}

    return metrics


def _pick_champion(metrics: dict) -> str:
    sarima_mae = metrics.get("sarima", {}).get("mae", float("inf"))
    prophet_mae = metrics.get("prophet", {}).get("mae", float("inf"))
    return "prophet" if prophet_mae <= sarima_mae else "sarima"


def main() -> int:
    df = load_processed()
    brasil = monthly_counts(df)
    logger.info("Série mensal Brasil: %d pontos", len(brasil))

    metrics = evaluate_models(brasil)
    champion = _pick_champion(metrics) if "skipped" not in metrics else "prophet"
    logger.info("Métricas Brasil: %s | campeão: %s", metrics, champion)

    if champion == "sarima":
        forecast_br = fit_sarima(
            brasil.set_index("mes_ref")["acidentes"], horizon=HORIZON_MONTHS
        )
    else:
        forecast_br = fit_prophet(brasil, horizon=HORIZON_MONTHS)

    forecast_br["model"] = champion
    out_br = MODELS_DIR / "forecast_brasil.parquet"
    forecast_br.to_parquet(out_br, index=False)
    logger.info("Forecast Brasil (%s) salvo em %s", champion, out_br)

    (MODELS_DIR / "forecast_metrics.json").write_text(
        json.dumps({"brasil": metrics, "champion_brasil": champion}, indent=2, ensure_ascii=False)
    )

    por_uf = monthly_counts(df, by="uf")
    pieces = []
    for uf, grupo in por_uf.groupby("uf"):
        if len(grupo) < 24:
            continue
        try:
            fc = fit_prophet(grupo[["mes_ref", "acidentes"]], horizon=HORIZON_MONTHS)
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
