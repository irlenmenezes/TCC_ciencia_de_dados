# Contribuindo

## Setup local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Padrões

- **Formatação:** `black .` (linha 100).
- **Lint:** `ruff check .` (auto-fix com `ruff check --fix .`).
- **Testes:** `pytest -v tests/`. Toda nova feature em `src/` deve vir com teste.
- **Notebooks:** sandbox de exploração. Lógica reutilizável vai para `src/`.

## Convenção de commits

Mensagens em português, modo imperativo no presente:

- `Adiciona X`, `Corrige Y`, `Atualiza Z`, `Refatora W`.
- Primeira linha ≤ 72 caracteres.
- Corpo opcional explicando o "porquê", não o "o quê".

## Pull Requests

- Uma feature por PR.
- Inclua descrição do problema, decisão tomada e teste/screenshot quando aplicável.
- CI verde antes de pedir review.
