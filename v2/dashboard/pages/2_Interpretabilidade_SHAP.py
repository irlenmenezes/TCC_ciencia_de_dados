"""Visualiza as explicações SHAP geradas por src/models/explain.py."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.utils.paths import MODELS_DIR

st.set_page_config(page_title="Interpretabilidade", page_icon=":mag:", layout="wide")
st.title(":mag: Interpretabilidade do Classificador (SHAP)")

shap_parquet = MODELS_DIR / "shap_values.parquet"
shap_summary_png = MODELS_DIR / "shap_summary.png"
shap_beeswarm_png = MODELS_DIR / "shap_beeswarm.png"

if not shap_parquet.exists():
    st.warning(
        "Explicações SHAP não geradas ainda. Rode `make explain` (ou "
        "`python -m src.models.explain`) após treinar o modelo."
    )
    st.stop()

st.caption(
    "Importância média absoluta dos valores SHAP — quanto cada feature contribui, "
    "em média, para o deslocamento da predição."
)

df = pd.read_parquet(shap_parquet)
top_n = st.slider("Top N features", 5, min(50, len(df)), 20)

fig = px.bar(
    df.head(top_n).sort_values("mean_abs_shap"),
    x="mean_abs_shap", y="feature", orientation="h",
    title=f"Top {top_n} features mais influentes",
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("Plots adicionais (gerados pelo SHAP)"):
    col1, col2 = st.columns(2)
    if shap_summary_png.exists():
        col1.image(str(shap_summary_png), caption="Resumo SHAP", use_container_width=True)
    if shap_beeswarm_png.exists():
        col2.image(str(shap_beeswarm_png), caption="Beeswarm — classe 'fatal'",
                   use_container_width=True)

st.markdown(
    """
    **Como ler:**
    - Features no topo influenciam mais a decisão do modelo.
    - O beeswarm mostra, para cada amostra, se a feature *empurra* a predição
      em direção à classe `fatal` (à direita, vermelho = valor alto) ou *afasta*
      (à esquerda).
    - Use isso para validar se o modelo está capturando relações esperadas
      (ex.: hora noturna, pista molhada, BR perigosa).
    """
)
