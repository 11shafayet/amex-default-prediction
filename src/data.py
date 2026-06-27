from pathlib import Path
from typing import TYPE_CHECKING

from src.config import TEST_FILE, TRAIN_FILE

if TYPE_CHECKING:
    import pandas as pd


def load_parquet(path: str | Path) -> "pd.DataFrame":
    """Load a parquet file and fail with a clear path-specific message."""
    parquet_path = Path(path)
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Could not find {parquet_path}. Download the Kaggle parquet files "
            "into data/raw/ first."
        )

    import pandas as pd

    return pd.read_parquet(parquet_path)


def load_train_data(path: str | Path = TRAIN_FILE) -> "pd.DataFrame":
    return load_parquet(path)


def load_test_data(path: str | Path = TEST_FILE) -> "pd.DataFrame":
    return load_parquet(path)


def print_dataset_overview(data: "pd.DataFrame") -> None:
    row_count = data.shape[0]
    customer_count = data["customer_ID"].nunique()

    print(row_count)
    print(customer_count)
    data.info(memory_usage="deep")
