# Dados brutos (data/raw)

Os CSVs originais da PRF **não são versionados** (são grandes e públicos).

Para baixá-los, rode:

```bash
python -m src.data.ingest --years 2021 2022 2023 2024 2025
```

Ou pegue manualmente em:
https://portal.prf.gov.br/dados-abertos-acidentes

Estrutura esperada após o download:

```
data/raw/
├── datatran2021.csv
├── datatran2022.csv
├── datatran2023.csv
├── datatran2024.csv
└── datatran2025.csv
```
