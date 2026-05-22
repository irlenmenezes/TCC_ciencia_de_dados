# Deploy do Dashboard

## Streamlit Community Cloud (gratuito)

1. Faça push do repositório para o GitHub (público).
2. Acesse https://share.streamlit.io e logue com sua conta GitHub.
3. Clique em **New app** e selecione:
   - Repositório: `<seu-user>/<seu-repo>`
   - Branch: `main`
   - Main file path: `dashboard/app.py`
   - Python version: `3.11`
4. Em **Advanced settings → Secrets**, adicione variáveis se houver (não há por padrão).
5. Clique em **Deploy**. Em ~3 minutos o app estará em
   `https://<seu-user>-<seu-repo>.streamlit.app`.

### Persistir dados processados

O Streamlit Cloud não persiste arquivos entre deploys. Duas opções:

**A. Commitar o parquet processado** (até ~100 MB):
```bash
git add -f data/processed/acidentes.parquet
git commit -m "snapshot de dados processados"
```

**B. Baixar e processar no startup** (recomendado para datasets grandes):
edite `dashboard/app.py` para chamar `ingest + clean` quando o parquet não
existir, e configure uma URL de download (S3, Google Drive público, etc.).

## Deploy alternativo: Hugging Face Spaces

1. Crie um Space do tipo *Streamlit* em https://huggingface.co/spaces.
2. Clone o Space e copie o conteúdo de `v2/` para a raiz.
3. Renomeie `dashboard/app.py` para `app.py` na raiz (ou ajuste `README.md`
   do Space com `app_file: dashboard/app.py`).
4. `git push` — o Space faz build automático.

## Atualização automática

No GitHub Actions, adicione um job que rode o pipeline mensalmente:

```yaml
on:
  schedule:
    - cron: "0 6 1 * *"  # dia 1 de cada mês, 6h UTC
```

Esse job:
1. Roda `python -m src.data.ingest --years $(date +%Y)`
2. Roda `python -m src.data.clean`
3. Comita o parquet atualizado (se mudou) e dispara o redeploy do Streamlit.
