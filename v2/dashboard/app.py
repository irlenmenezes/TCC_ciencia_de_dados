"""Dashboard interativo dos acidentes rodoviários PRF."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.features.build import load_processed

st.set_page_config(
    page_title="Acidentes Rodoviários PRF",
    page_icon=":car:",
    layout="wide",
)


@st.cache_data(show_spinner="Carregando dados...")
def get_data() -> pd.DataFrame:
    return load_processed()


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filtros")

    anos_disponiveis = sorted(df["data"].dt.year.dropna().unique().astype(int))
    anos = st.sidebar.multiselect("Ano", anos_disponiveis, default=anos_disponiveis)

    ufs = sorted(df["uf"].dropna().unique())
    ufs_sel = st.sidebar.multiselect("UF", ufs, default=ufs)

    gravidade_sel = st.sidebar.multiselect(
        "Gravidade", ["sem_vitimas", "com_feridos", "fatal"],
        default=["sem_vitimas", "com_feridos", "fatal"],
    )

    mask = (
        df["data"].dt.year.isin(anos)
        & df["uf"].isin(ufs_sel)
        & df["gravidade"].isin(gravidade_sel)
    )
    return df[mask]


def kpis(df: pd.DataFrame) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Acidentes", f"{len(df):,}".replace(",", "."))
    col2.metric("Mortos", int(df["mortos"].sum()))
    col3.metric("Feridos", int(df.get("feridos", df["feridos_leves"] + df["feridos_graves"]).sum()))
    col4.metric("UFs", df["uf"].nunique())


def main() -> None:
    st.title(":car: Análise de Acidentes Rodoviários — PRF")
    st.caption("Evolução do TCC PUC Minas. Dados oficiais da Polícia Rodoviária Federal.")

    try:
        df = get_data()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Execute o pipeline: `make data && make clean` antes de abrir o dashboard.")
        return

    df_filtrado = sidebar_filters(df)
    kpis(df_filtrado)

    tab1, tab2, tab3, tab4 = st.tabs(["Temporal", "Geográfico", "Causas/Tipos", "Previsão"])

    with tab1:
        st.subheader("Acidentes por mês")
        por_mes = (
            df_filtrado.assign(mes=df_filtrado["data"].dt.to_period("M").dt.to_timestamp())
            .groupby(["mes", "gravidade"])
            .size()
            .reset_index(name="acidentes")
        )
        fig = px.area(por_mes, x="mes", y="acidentes", color="gravidade")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Acidentes por UF")
        por_uf = df_filtrado.groupby("uf").size().reset_index(name="acidentes").sort_values("acidentes")
        fig = px.bar(por_uf, x="acidentes", y="uf", orientation="h")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Top 15 causas")
        if "causa_acidente" in df_filtrado.columns:
            top_causas = (
                df_filtrado["causa_acidente"].value_counts().head(15).reset_index()
            )
            top_causas.columns = ["causa", "acidentes"]
            fig = px.bar(top_causas, x="acidentes", y="causa", orientation="h")
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Previsão de acidentes")
        try:
            from src.models.predict import load_forecast_brasil

            forecast = load_forecast_brasil()
            fig = px.line(forecast, x="ds", y="yhat", title="Forecast Brasil")
            fig.add_scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines", name="upper")
            fig.add_scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines", name="lower")
            st.plotly_chart(fig, use_container_width=True)
        except FileNotFoundError as exc:
            st.warning(str(exc))


if __name__ == "__main__":
    main()
