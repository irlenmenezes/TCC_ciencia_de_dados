# Arquitetura

## Fluxo de dados

```
PRF (portal aberto)
   │ download
   ▼
data/raw/datatran<ano>.csv   ── src/data/ingest.py
   │ limpeza, normalização, target
   ▼
data/processed/acidentes.parquet   ── src/data/clean.py
   │ feature engineering
   ▼
       ┌──────────────────────┬──────────────────────────┐
       ▼                      ▼                          ▼
  Classificação           Forecast                   Dashboard
 (gravidade)             (acidentes/mês)             (Streamlit)
  LogisticRegression /   Prophet / SARIMA           lê parquet + modelos
  LightGBM
```

## Decisões de design

- **Parquet em vez de CSV** para a camada processada: ~10x menor, leitura mais rápida.
- **Pipeline scikit-learn** (não fit/transform manual): garante que a inferência aplica exatamente a mesma transformação do treino.
- **Cache com `@lru_cache`** no `predict.py` para o modelo não recarregar a cada request do Streamlit.
- **`@st.cache_data`** para o parquet processado — só recarrega quando o arquivo muda.

## Diferenças vs TCC original

| Aspecto | TCC original | v2 |
|---|---|---|
| Janela de dados | 2017-2020 | 2021-2025 |
| Código | Notebook único monolítico | Módulos em `src/` testáveis |
| ML | Não implementado | Classificação + forecast |
| Dashboard | Power BI (download) | Streamlit (web público) |
| Reprodutibilidade | Manual | `make` + `requirements.txt` + CI |

## Próximos passos

- Adicionar testes para `train_classification` (smoke test com fixture pequena).
- Logging estruturado (JSON) em produção.
- Versionar modelos com MLflow ou DVC.
- Deploy do Streamlit no Streamlit Cloud com workflow de redeploy automático.
