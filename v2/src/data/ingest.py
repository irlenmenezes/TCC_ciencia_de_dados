"""Baixa CSVs de acidentes da PRF (datatran<ano>.csv) para data/raw/.

A PRF distribui os arquivos via ownCloud em URLs do tipo:
    http://arquivos.prf.gov.br/arquivos/index.php/s/<TOKEN>/download

Os tokens são listados dinamicamente na página
https://portal.prf.gov.br/dados-abertos-acidentes e podem mudar quando
a PRF republica os arquivos.

Este módulo suporta dois modos:

1. **Override manual** (recomendado quando a PRF muda o layout do site):
   crie `config/prf_urls.yml` com o token de cada ano, ex.:

       acidentes_ocorrencia:
         2021: "kgJ0ea8QZrix5Yt"   # apenas o token; a URL é montada
         2022: "EF4uPKCihT0ouXd"
         2023: "sdvJndbl5wLyh3J"
         2024: "su5ocOO8HQGt06D"
         2025: ""                   # quando ainda não estiver disponível

   Para descobrir o token, abra a página da PRF, clique-direito no
   link de download do CSV e copie o endereço.

2. **Scraping automático** (fallback): se `config/prf_urls.yml` não
   existir ou estiver vazio, o script tenta extrair os links direto
   da página HTML usando BeautifulSoup.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import requests
import yaml
from bs4 import BeautifulSoup

from src.utils.paths import DATA_RAW

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ROOT = Path(__file__).resolve().parents[2]
URLS_CONFIG = ROOT / "config" / "prf_urls.yml"

PRF_DOWNLOAD = "http://arquivos.prf.gov.br/arquivos/index.php/s/{token}/download"
PRF_LISTING = "https://portal.prf.gov.br/dados-abertos-acidentes"

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "pt-BR,pt;q=0.9"}

TIPOS_VALIDOS = {"acidentes_ocorrencia", "acidentes_pessoa", "acidentes_agrupados"}


def load_tokens(tipo: str = "acidentes_ocorrencia") -> dict[int, str]:
    """Lê tokens manuais de config/prf_urls.yml, se existir."""
    if not URLS_CONFIG.exists():
        return {}
    data = yaml.safe_load(URLS_CONFIG.read_text()) or {}
    section = data.get(tipo, {}) or {}
    return {int(k): str(v) for k, v in section.items() if v}


def scrape_tokens(tipo: str = "acidentes_ocorrencia") -> dict[int, str]:
    """Extrai tokens da página oficial da PRF.

    Implementação baseada em github.com/nymarya/prf_api.
    """
    logger.info("Buscando tokens em %s ...", PRF_LISTING)
    resp = requests.get(PRF_LISTING, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    pattern = re.compile(r"arquivos\.prf\.gov\.br/arquivos/index\.php/s/([A-Za-z0-9]+)")

    tokens: dict[int, str] = {}
    # Estratégia: percorre todos os links e tenta inferir o ano pelo texto/atributos.
    for link in soup.find_all("a", href=True):
        href = link["href"]
        match = pattern.search(href)
        if not match:
            continue
        text = " ".join([link.get_text(" ", strip=True), link.get("title", "")]).lower()
        year_match = re.search(r"(20\d{2})", text)
        if not year_match:
            continue

        # Filtragem por tipo (heurística no texto do link).
        if tipo == "acidentes_ocorrencia" and "ocorrência" not in text and "ocorrencia" not in text:
            continue
        if tipo == "acidentes_pessoa" and "pessoa" not in text:
            continue

        tokens[int(year_match.group(1))] = match.group(1)

    if not tokens:
        logger.warning("Nenhum token encontrado pelo scraping. Confira o layout da página.")
    return tokens


def resolve_tokens(tipo: str = "acidentes_ocorrencia") -> dict[int, str]:
    """Combina tokens manuais (override) + scraping."""
    manual = load_tokens(tipo)
    if manual:
        logger.info("Tokens carregados de %s: %s", URLS_CONFIG, sorted(manual.keys()))
    try:
        scraped = scrape_tokens(tipo)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Scraping falhou (%s). Usando apenas o YAML.", exc)
        scraped = {}
    # Manual tem prioridade sobre scraping.
    merged = {**scraped, **manual}
    return merged


def download_year(year: int, token: str, dest_dir: Path = DATA_RAW) -> Path:
    url = PRF_DOWNLOAD.format(token=token)
    dest_csv = dest_dir / f"datatran{year}.csv"
    if dest_csv.exists():
        logger.info("Já existe %s — pulando.", dest_csv.name)
        return dest_csv

    logger.info("Baixando %s ...", url)
    resp = requests.get(url, headers=HEADERS, stream=True, timeout=120)
    resp.raise_for_status()

    content_type = resp.headers.get("Content-Type", "").lower()
    is_zip = "zip" in content_type or url.endswith(".zip") or resp.content[:2] == b"PK"

    if is_zip:
        with zipfile.ZipFile(BytesIO(resp.content)) as zf:
            csv_name = next(
                (n for n in zf.namelist() if n.lower().endswith(".csv") and "datatran" in n.lower()),
                None,
            )
            if csv_name is None:
                # fallback: pega o primeiro CSV
                csv_name = next((n for n in zf.namelist() if n.lower().endswith(".csv")), None)
            if csv_name is None:
                raise RuntimeError(f"Nenhum CSV dentro do zip de {year}")
            with zf.open(csv_name) as src, open(dest_csv, "wb") as dst:
                dst.write(src.read())
    else:
        with open(dest_csv, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

    logger.info("Salvo em %s (%.1f MB)", dest_csv, dest_csv.stat().st_size / 1e6)
    return dest_csv


def main() -> int:
    parser = argparse.ArgumentParser(description="Baixa CSVs da PRF (ownCloud).")
    parser.add_argument("--years", nargs="+", type=int, default=[2021, 2022, 2023, 2024, 2025])
    parser.add_argument("--tipo", default="acidentes_ocorrencia", choices=sorted(TIPOS_VALIDOS))
    parser.add_argument("--list", action="store_true",
                        help="Apenas lista os tokens descobertos sem baixar nada.")
    args = parser.parse_args()

    tokens = resolve_tokens(args.tipo)
    if args.list:
        for year in sorted(tokens):
            logger.info("  %s -> %s", year, tokens[year])
        return 0

    if not tokens:
        logger.error(
            "Nenhum token disponível. Crie %s manualmente ou verifique sua conexão.",
            URLS_CONFIG,
        )
        return 1

    failures: list[int] = []
    for year in args.years:
        token = tokens.get(year)
        if not token:
            logger.warning("Sem token para %s. Pule ou adicione em %s.", year, URLS_CONFIG)
            failures.append(year)
            continue
        try:
            download_year(year, token)
        except Exception as exc:  # noqa: BLE001
            logger.error("Falha em %s: %s", year, exc)
            failures.append(year)

    if failures:
        logger.error("Anos com falha: %s", failures)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
