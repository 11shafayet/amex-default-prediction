import unittest

import numpy as np

from src.metric import amex_metric, normalized_weighted_gini, top_four_percent


class AmexMetricTest(unittest.TestCase):
    def test_top_four_percent_matches_notebook_toy_example(self) -> None:
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([0.99, 0.95, 0.90, 0.40, 0.20, 0.10])

        self.assertAlmostEqual(top_four_percent(y_true, y_pred), 2 / 3)

    def test_perfect_predictions_have_perfect_normalized_gini(self) -> None:
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([0.99, 0.95, 0.90, 0.40, 0.20, 0.10])

        self.assertAlmostEqual(normalized_weighted_gini(y_true, y_pred), 1.0)

    def test_amex_metric_combines_gini_and_top_four_capture(self) -> None:
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([0.99, 0.95, 0.90, 0.40, 0.20, 0.10])

        self.assertAlmostEqual(amex_metric(y_true, y_pred), 5 / 6)

    def test_rejects_non_binary_labels(self) -> None:
        with self.assertRaises(ValueError):
            amex_metric([0, 1, 2], [0.1, 0.9, 0.5])


if __name__ == "__main__":
    unittest.main()
