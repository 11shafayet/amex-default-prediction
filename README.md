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

## Links

- Competition: https://www.kaggle.com/competitions/amex-default-prediction
- Dataset (Parquet): https://www.kaggle.com/datasets/raddar/amex-data-integer-dtypes-parquet-format
