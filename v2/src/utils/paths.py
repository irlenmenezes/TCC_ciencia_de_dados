from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

for path in (DATA_RAW, DATA_PROCESSED, MODELS_DIR):
    path.mkdir(parents=True, exist_ok=True)
