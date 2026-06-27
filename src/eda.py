import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from src.config import CATEGORICAL_COLUMNS, ID_COLUMNS, TRAIN_FILE
from src.data import load_train_data, print_dataset_overview

if TYPE_CHECKING:
    import pandas as pd


def missingness_by_prefix(
    data: "pd.DataFrame",
) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    import pandas as pd

    missing_pct = data.isnull().mean() * 100
    miss_df = pd.DataFrame(
        {"column": missing_pct.index, "missing_pct": missing_pct.values}
    )
    miss_df = miss_df[~miss_df["column"].isin(ID_COLUMNS)].copy()
    miss_df["prefix"] = miss_df["column"].str[0]

    prefix_summary = miss_df.groupby("prefix")["missing_pct"].agg(
        ["mean", "max", "min", "count"]
    )
    top_missing = miss_df.sort_values("missing_pct", ascending=False).head(15)
    return prefix_summary, top_missing


def print_missingness_report(data: "pd.DataFrame") -> None:
    prefix_summary, top_missing = missingness_by_prefix(data)

    print("Missingness by prefix group:\n")
    print(prefix_summary.round(2))

    print("\nTop 15 most-missing columns:\n")
    print(top_missing.to_string(index=False))


def statement_count_summary(data: "pd.DataFrame") -> "pd.Series":
    return data.groupby("customer_ID").size()


def print_statement_count_report(data: "pd.DataFrame") -> None:
    stmt_counts = statement_count_summary(data)

    print("\nStatement-count distribution (how many customers have N statements):\n")
    print(stmt_counts.value_counts().sort_index())

    full_13 = (stmt_counts == 13).mean() * 100
    print(f"\n% of customers with full 13 statements: {full_13:.2f}%")
    print(f"% with fewer than 13: {100 - full_13:.2f}%")
    print(f"\nSummary stats of statement counts:\n{stmt_counts.describe().round(2)}")


def categorical_column_summary(
    data: "pd.DataFrame", columns: list[str] | None = None
) -> "pd.DataFrame":
    import pandas as pd

    rows = []
    selected_columns = columns or CATEGORICAL_COLUMNS

    for col in selected_columns:
        if col not in data.columns:
            continue

        uniques = sorted(data[col].dropna().unique().tolist())
        shown = uniques if len(uniques) <= 6 else uniques[:6] + ["..."]
        rows.append(
            {
                "column": col,
                "dtype": str(data[col].dtype),
                "n_unique": data[col].nunique(dropna=True),
                "values": shown,
            }
        )

    return pd.DataFrame(rows)


def print_categorical_column_report(data: "pd.DataFrame") -> None:
    summary = categorical_column_summary(data)

    print("\nCategorical columns - dtype, unique count, and the actual values:\n")
    for row in summary.itertuples(index=False):
        print(
            f"{row.column:6} | dtype={row.dtype:8} | "
            f"n_unique={row.n_unique:3} | values={row.values}"
        )


def run_basic_eda(train_file: str | Path = TRAIN_FILE) -> None:
    train_data = load_train_data(train_file)

    print_dataset_overview(train_data)
    print_missingness_report(train_data)
    print_statement_count_report(train_data)
    print_categorical_column_report(train_data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AMEX notebook EDA split.")
    parser.add_argument(
        "--train-file",
        type=Path,
        default=TRAIN_FILE,
        help="Path to train.parquet.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_basic_eda(args.train_file)


if __name__ == "__main__":
    main()
