"""Global paths and settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
SUBMISSIONS_DIR = ROOT / "submissions"

# Raw files (from the integer-dtype parquet dataset)
TRAIN_PATH = RAW_DIR / "train.parquet"
TEST_PATH = RAW_DIR / "test.parquet"
TRAIN_LABELS_PATH = RAW_DIR / "train_labels.csv"

ID_COL = "customer_ID"
TARGET_COL = "target"
DATE_COL = "S_2"

SEED = 42
