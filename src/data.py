"""Data loading helpers."""
import pandas as pd

from . import config


def load_train():
    """Load training features and merge with labels."""
    df = pd.read_parquet(config.TRAIN_PATH)
    labels = pd.read_csv(config.TRAIN_LABELS_PATH)
    return df.merge(labels, on=config.ID_COL, how="left")


def load_test():
    """Load test features."""
    return pd.read_parquet(config.TEST_PATH)
