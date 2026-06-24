"""Official American Express evaluation metric.

The score is the mean of two measures:
  - G: normalized Gini coefficient
  - D: default rate captured at the top 4% of predictions
with a 20x weight on the negative (non-default) class.
"""
import numpy as np
import pandas as pd


def amex_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    df = pd.DataFrame({"target": np.asarray(y_true), "prediction": np.asarray(y_pred)})

    def top_four_percent_captured(df):
        df = df.sort_values("prediction", ascending=False)
        df["weight"] = df["target"].apply(lambda x: 20 if x == 0 else 1)
        cutoff = int(0.04 * df["weight"].sum())
        within_cutoff = df["weight"].cumsum() <= cutoff
        return df.loc[within_cutoff, "target"].sum() / df["target"].sum()

    def weighted_gini(df):
        df = df.sort_values("prediction", ascending=False)
        df["weight"] = df["target"].apply(lambda x: 20 if x == 0 else 1)
        random = (df["weight"] / df["weight"].sum()).cumsum()
        total_pos = (df["target"] * df["weight"]).sum()
        df["cum_pos_found"] = (df["target"] * df["weight"]).cumsum()
        lorentz = df["cum_pos_found"] / total_pos
        return ((lorentz - random) * df["weight"]).sum()

    def normalized_weighted_gini(df):
        ideal = df.assign(prediction=df["target"])
        return weighted_gini(df) / weighted_gini(ideal)

    g = normalized_weighted_gini(df)
    d = top_four_percent_captured(df)
    return 0.5 * (g + d)
