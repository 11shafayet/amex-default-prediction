"""Inference and submission generation."""
import pandas as pd

from . import config, data, features


def main():
    df = data.load_test()
    X = features.build_features(df)
    # TODO: load model from config.MODELS_DIR, predict, and write a submission.
    submission = pd.DataFrame({config.ID_COL: X[config.ID_COL], "prediction": 0.0})
    config.SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.SUBMISSIONS_DIR / "submission.csv"
    submission.to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
