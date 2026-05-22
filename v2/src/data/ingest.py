"""Baixa CSVs de acidentes da PRF (datatran<ano>.csv) para data/raw/.

Os links da PRF mudam de tempos em tempos. A constante PRF_URLS abaixo
deve ser atualizada conforme o portal oficial:
https://portal.prf.gov.br/dados-abertos-acidentes
"""
from __future__ import annotations

import argparse
import logging
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import requests

from src.utils.paths import DATA_RAW

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Atualizar com URLs reais do portal PRF.
# Quando a PRF disponibiliza em .zip, o download_year extrai o CSV automaticamente.
PRF_URLS: dict[int, str] = {
    2021: "https://drive.google.com/uc?id=PLACEHOLDER_2021",
    2022: "https://drive.google.com/uc?id=PLACEHOLDER_2022",
    2023: "https://drive.google.com/uc?id=PLACEHOLDER_2023",
    2024: "https://drive.google.com/uc?id=PLACEHOLDER_2024",
    2025: "https://drive.google.com/uc?id=PLACEHOLDER_2025",
}


def download_year(year: int, dest_dir: Path = DATA_RAW) -> Path:
    url = PRF_URLS.get(year)
    if url is None or "PLACEHOLDER" in url:
        raise ValueError(
            f"URL não configurada para {year}. Edite PRF_URLS em src/data/ingest.py "
            f"com o link atual de https://portal.prf.gov.br/dados-abertos-acidentes"
        )

    dest_csv = dest_dir / f"datatran{year}.csv"
    if dest_csv.exists():
        logger.info("Já existe %s — pulando.", dest_csv.name)
        return dest_csv

    logger.info("Baixando %s ...", url)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    if "zip" in content_type or url.endswith(".zip"):
        with zipfile.ZipFile(BytesIO(response.content)) as zf:
            csv_name = next(
                (n for n in zf.namelist() if n.lower().endswith(".csv") and "datatran" in n.lower()),
                None,
            )
            if csv_name is None:
                raise RuntimeError(f"Nenhum datatran*.csv encontrado dentro do zip de {year}")
            with zf.open(csv_name) as src, open(dest_csv, "wb") as dst:
                dst.write(src.read())
    else:
        with open(dest_csv, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    logger.info("Salvo em %s", dest_csv)
    return dest_csv


def main() -> int:
    parser = argparse.ArgumentParser(description="Baixa CSVs da PRF.")
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=list(PRF_URLS.keys()),
        help="Anos a baixar (ex.: --years 2021 2022)",
    )
    args = parser.parse_args()

    failures = []
    for year in args.years:
        try:
            download_year(year)
        except Exception as exc:  # noqa: BLE001
            logger.error("Falha em %s: %s", year, exc)
            failures.append(year)

    if failures:
        logger.error("Anos com falha: %s", failures)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
