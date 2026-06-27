from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

TRAIN_FILE = RAW_DATA_DIR / "train.parquet"
TEST_FILE = RAW_DATA_DIR / "test.parquet"
TRAIN_LABELS_FILE = RAW_DATA_DIR / "train_labels.csv"

CATEGORICAL_COLUMNS = [
    "B_30",
    "B_38",
    "D_114",
    "D_116",
    "D_117",
    "D_120",
    "D_126",
    "D_63",
    "D_64",
    "D_66",
    "D_68",
]

ID_COLUMNS = ["customer_ID", "S_2"]
