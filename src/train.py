"""Model training entry point."""
from . import config, data, features
from .metric import amex_metric


def main():
    df = data.load_train()
    X = features.build_features(df)
    # TODO: merge target, split folds, fit model, evaluate with amex_metric,
    #       and save the trained model to config.MODELS_DIR.
    print(f"Built features: {X.shape}")


if __name__ == "__main__":
    main()
