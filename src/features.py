"""Feature engineering.

The raw data has multiple monthly statements per customer. Models need one
row per customer, so we aggregate the time series here.
"""
import pandas as pd

from . import config


def aggregate_by_customer(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the per-statement rows into one feature row per customer."""
    feature_cols = [c for c in df.columns if c not in (config.ID_COL, config.DATE_COL)]
    num_cols = df[feature_cols].select_dtypes("number").columns.tolist()

    agg = df.groupby(config.ID_COL)[num_cols].agg(["mean", "std", "min", "max", "last"])
    agg.columns = ["_".join(c) for c in agg.columns]
    return agg.reset_index()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Entry point for turning raw statements into model-ready features."""
    return aggregate_by_customer(df)
