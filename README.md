# American Express - Default Prediction

Solution for the [American Express - Default Prediction](https://www.kaggle.com/competitions/amex-default-prediction) Kaggle competition.

## Goal

Predict whether a customer will default on their credit card balance in the future, based on their monthly customer profile.

## Dataset

This project uses the **integer-dtype Parquet** version of the data, which is much smaller and faster to load than the original CSVs:

- Dataset: https://www.kaggle.com/datasets/raddar/amex-data-integer-dtypes-parquet-format

Download the Parquet files into `data/raw/` (e.g. `train.parquet`, `test.parquet`, `train_labels.csv`).

## Project Structure

```
.
├── data/
│   ├── raw/          # original parquet files (from the dataset above)
│   └── processed/    # cleaned / feature-engineered data
├── notebooks/        # exploratory analysis
├── src/
│   ├── config.py     # paths and global settings
│   ├── data.py       # data loading
│   ├── features.py   # feature engineering
│   ├── metric.py     # official Amex evaluation metric
│   ├── train.py      # model training
│   └── predict.py    # inference / submission
├── models/           # saved model artifacts
└── submissions/      # generated submission files
```

## Setup

```bash
pip install -r requirements.txt
```

## Run Current Notebook Split

The current notebook work has been split into Python modules under `src/`.

```bash
python3 main.py
```

By default this loads `data/raw/train.parquet` and runs the same EDA checks from the notebook:

- dataset row/customer count and memory info
- missingness by feature prefix
- statement-count distribution by customer
- categorical column value summaries

You can also run the EDA module directly with a custom file:

```bash
python3 -m src.eda --train-file data/raw/train.parquet
```

Stage 2 has been split into `src/metric.py`. It includes the top-4% capture
function from the notebook plus the full AMEX metric:

```bash
python3 -m unittest tests/test_metric.py
```

## Links

- Competition: https://www.kaggle.com/competitions/amex-default-prediction
- Dataset (Parquet): https://www.kaggle.com/datasets/raddar/amex-data-integer-dtypes-parquet-format
