"""Formulário interativo para prever a gravidade de um acidente hipotético."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.features.build import load_processed
from src.models.predict import predict_gravidade

st.set_page_config(page_title="Previsão de Gravidade", page_icon=":bar_chart:", layout="wide")

st.title(":bar_chart: Previsão de Gravidade do Acidente")
st.caption(
    "Informe as características do acidente e veja a probabilidade estimada "
    "pelo classificador (LightGBM)."
)


@st.cache_data(show_spinner=False)
def opcoes_categoricas() -> dict[str, list[str]]:
    df = load_processed()
    cols = [
        "uf", "br", "dia_semana", "fase_dia", "condicao_metereologica",
        "tipo_pista", "tracado_via", "uso_solo", "causa_acidente", "tipo_acidente",
    ]
    return {c: sorted(df[c].dropna().unique().tolist()) for c in cols if c in df.columns}


try:
    opcoes = opcoes_categoricas()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.info("Rode `make data && make clean && make train` antes de usar esta página.")
    st.stop()


with st.form("inferencia"):
    col1, col2, col3 = st.columns(3)

    uf = col1.selectbox("UF", opcoes.get("uf", []))
    br = col1.selectbox("BR", opcoes.get("br", []))
    km = col1.number_input("KM", min_value=0.0, max_value=2000.0, value=100.0)

    dia_semana = col2.selectbox("Dia da semana", opcoes.get("dia_semana", []))
    fase_dia = col2.selectbox("Fase do dia", opcoes.get("fase_dia", []))
    hora = col2.slider("Hora", 0, 23, 14)

    condicao = col3.selectbox("Condição meteorológica", opcoes.get("condicao_metereologica", []))
    tipo_pista = col3.selectbox("Tipo de pista", opcoes.get("tipo_pista", []))
    tracado = col3.selectbox("Traçado da via", opcoes.get("tracado_via", []))

    causa = st.selectbox("Causa do acidente", opcoes.get("causa_acidente", []))
    tipo = st.selectbox("Tipo de acidente", opcoes.get("tipo_acidente", []))

    col_a, col_b, col_c = st.columns(3)
    veiculos = col_a.number_input("Veículos envolvidos", 1, 30, 2)
    pessoas = col_b.number_input("Pessoas envolvidas", 1, 100, 3)
    mes = col_c.slider("Mês", 1, 12, 6)

    submitted = st.form_submit_button("Prever gravidade")

if submitted:
    payload = {
        "uf": uf, "br": br, "km": km, "veiculos": veiculos, "pessoas": pessoas,
        "hora": hora, "mes": mes, "dia_semana": dia_semana, "fase_dia": fase_dia,
        "condicao_metereologica": condicao, "tipo_pista": tipo_pista,
        "tracado_via": tracado, "uso_solo": (opcoes.get("uso_solo") or ["urbano"])[0],
        "causa_acidente": causa, "tipo_acidente": tipo,
    }
    try:
        probs = predict_gravidade(payload)
    except FileNotFoundError as exc:
        st.error(str(exc))
    else:
        df_probs = (
            pd.DataFrame({"classe": list(probs.keys()), "probabilidade": list(probs.values())})
            .sort_values("probabilidade", ascending=False)
        )
        st.subheader("Probabilidade por classe")
        st.dataframe(df_probs, hide_index=True, use_container_width=True)
        fig = px.bar(df_probs, x="classe", y="probabilidade",
                     color="classe", text_auto=".1%",
                     title="Gravidade estimada")
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

        top = df_probs.iloc[0]
        st.success(f"Predição: **{top['classe']}** ({top['probabilidade']:.1%} de confiança)")
