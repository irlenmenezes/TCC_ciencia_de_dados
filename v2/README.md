# Análise de Acidentes Rodoviários PRF — v2

Evolução do TCC de Ciência de Dados e Big Data (PUC Minas) para um projeto de portfólio com dados atualizados, modelos preditivos e dashboard web interativo.

> **Projeto original (TCC entregue):** [`TCC_ciencia_de_dados`](https://github.com/irlenmenezes/TCC_ciencia_de_dados) — preservado como registro acadêmico.

---

## Objetivo

Refazer e expandir a análise de acidentes rodoviários brasileiros (dados da PRF) com:

1. **Dados atualizados** — janela 2021-2025 (TCC original cobre 2017-2020).
2. **Modelos preditivos**:
   - *Classificação* da gravidade do acidente (sem vítima / com feridos / com fatalidade).
   - *Forecast* (série temporal) da quantidade de acidentes por mês/UF.
3. **Código modular** em `src/` (testável, reutilizável) em vez de notebook monolítico.
4. **Dashboard web** em Streamlit, com deploy público (Streamlit Cloud).

## Stack

- **Python 3.11+**
- **Análise:** pandas, numpy, polars (opcional, para CSVs grandes)
- **Visualização:** plotly, matplotlib, seaborn
- **ML:** scikit-learn, lightgbm
- **Forecast:** statsmodels, prophet
- **Dashboard:** streamlit
- **Qualidade:** pytest, ruff, black

## Estrutura

```
v2/
├── data/
│   ├── raw/            # CSVs PRF originais (gitignored)
│   └── processed/      # parquet limpo e consolidado
├── notebooks/          # EDA e experimentação
├── src/
│   ├── data/           # ingestão e limpeza
│   ├── features/       # feature engineering
│   ├── models/         # treino e inferência
│   └── utils/          # helpers
├── dashboard/
│   ├── app.py          # página principal (EDA + filtros)
│   └── pages/          # páginas adicionais (previsão, SHAP)
├── tests/              # pytest
└── docs/               # documentação técnica
```

## Como rodar

```bash
# 1. Instalar dependências
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Baixar dados da PRF (2021-2025)
#    Tenta scraping automático; se falhar, preencha config/prf_urls.yml
#    com os tokens (instruções no próprio arquivo) e rode de novo.
python -m src.data.ingest --years 2021 2022 2023 2024 2025

#    Para apenas listar os tokens descobertos (sem baixar):
python -m src.data.ingest --list

# 3. Limpar e consolidar
python -m src.data.clean

# 4. Treinar modelos
python -m src.models.train_classification
python -m src.models.train_forecast

# 5. Gerar explicações SHAP (opcional, mas usado no dashboard)
python -m src.models.explain --sample 5000

# 6. Rodar dashboard
streamlit run dashboard/app.py
```

Ou via Makefile:

```bash
make install
make data
make train
make dashboard
```

## Perguntas de pesquisa

Além das perguntas do TCC original (quantidade, causas, tipos, dias, BRs, horários, regiões, condições meteorológicas), o v2 responde:

- Como mudou o perfil dos acidentes entre 2017-2020 (pré-pandemia) e 2021-2025 (pós)?
- Dado um acidente, qual a probabilidade de ter vítimas fatais?
- Quantos acidentes esperar nos próximos 6 meses por UF?
- Quais features são mais preditivas para a gravidade do acidente?

## Roadmap

- [x] Estrutura modular (`src/`, `dashboard/`, `tests/`)
- [x] Script de ingestão parametrizável (PRF 2021-2025)
- [x] Pipeline de limpeza → parquet consolidado
- [x] Modelo de classificação: baseline LogReg + LightGBM com seleção do campeão
- [x] Modelo de forecast: SARIMA + Prophet com avaliação MAE/MAPE
- [x] Dashboard Streamlit com filtros interativos
- [x] Testes automatizados + CI (GitHub Actions)
- [x] Documentação de deploy (Streamlit Cloud)
- [x] SHAP / interpretabilidade do classificador (`src/models/explain.py`)
- [x] EDA comparativa pré/pós pandemia (`notebooks/02_eda.ipynb`)
- [x] Workflow agendado para reingestão mensal (`.github/workflows/monthly_ingest.yml`)
- [x] Páginas Streamlit extras (previsão interativa, interpretabilidade)
- [x] Ingestão configurável (scraping automático + tokens manuais em `config/prf_urls.yml`)
- [ ] Deploy efetivo no Streamlit Cloud
- [ ] Monitoramento de drift (PSI/KS sobre dados novos)

## Autor

Irlen Menezes — TCC PUC Minas, Pós em Ciência de Dados e Big Data.
