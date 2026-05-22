.PHONY: install data clean train dashboard test lint format help

PYTHON := python
YEARS := 2021 2022 2023 2024 2025

help:
	@echo "Comandos disponíveis:"
	@echo "  make install     - instala dependências em .venv"
	@echo "  make data        - baixa CSVs da PRF para data/raw/"
	@echo "  make clean       - limpa e consolida dados em data/processed/"
	@echo "  make train       - treina modelos de classificação e forecast"
	@echo "  make explain     - gera explicações SHAP do classificador"
	@echo "  make dashboard   - sobe o Streamlit"
	@echo "  make test        - roda pytest"
	@echo "  make lint        - roda ruff"
	@echo "  make format      - formata com black + ruff"

install:
	$(PYTHON) -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

data:
	$(PYTHON) -m src.data.ingest --years $(YEARS)

clean:
	$(PYTHON) -m src.data.clean

train:
	$(PYTHON) -m src.models.train_classification
	$(PYTHON) -m src.models.train_forecast

explain:
	$(PYTHON) -m src.models.explain --sample 5000

dashboard:
	streamlit run dashboard/app.py

test:
	pytest -v tests/

lint:
	ruff check src/ dashboard/ tests/

format:
	black src/ dashboard/ tests/
	ruff check --fix src/ dashboard/ tests/
