import unittest

import numpy as np
import pandas as pd

from src.restaurant_rating_prediction import (
    CUISINE_FEATURE_PREFIX,
    apply_display_uplift,
    build_feature_metadata,
    parse_rating,
    prepare_data,
    train_and_evaluate,
)


class RestaurantRatingPredictionTests(unittest.TestCase):
    def test_display_uplift_adds_point_three_and_caps_at_five(self):
        self.assertAlmostEqual(apply_display_uplift(3.2), 3.5)
        self.assertEqual(apply_display_uplift(5.0), 5.0)
        self.assertEqual(apply_display_uplift(4.9), 5.0)

    def test_parse_rating_variants(self):
        self.assertEqual(parse_rating("4.1/5"), 4.1)
        self.assertEqual(parse_rating(" 3.5 "), 3.5)
        self.assertTrue(np.isnan(parse_rating("NEW")))
        self.assertTrue(np.isnan(parse_rating("-")))

    def test_prepare_data_cleans_rating_and_excludes_target_leakage(self):
        frame = pd.DataFrame(
            {
                "rate": ["4.1/5", "3.5/5", "NEW", "4.0/5", "2.8/5", "4.5/5"],
                "aggregate_rating": [4.1, 3.5, None, 4.0, 2.8, 4.5],
                "votes": ["1,200", "100", "0", "45", "300", "600"],
                "approx_cost(for two people)": [
                    "₹1,000",
                    "₹500",
                    "₹300",
                    "₹700",
                    "₹400",
                    "₹900",
                ],
                "location": ["A", "B", "A", "C", "B", "A"],
                "local address": ["North", "South", "North", "East", "South", "West"],
                "reviews_list": ["review"] * 6,
            }
        )

        features, target, target_name = prepare_data(frame)

        self.assertEqual(target_name, "rate")
        self.assertEqual(len(target), 5)
        self.assertNotIn("rate", features.columns)
        self.assertNotIn("aggregate_rating", features.columns)
        self.assertNotIn("local address", features.columns)
        self.assertNotIn("reviews_list", features.columns)
        self.assertEqual(features["votes"].iloc[0], 1200)
        self.assertEqual(features["approx_cost(for two people)"].iloc[0], 1000)

    def test_cuisines_are_split_into_independent_features(self):
        frame = pd.DataFrame(
            {
                "rating": [3.0, 3.2, 3.5, 3.8, 4.0, 4.2],
                "cuisines type": [
                    "South Indian, Chinese",
                    "North Indian",
                    "Chinese",
                    "South Indian",
                    "North Indian, Chinese",
                    "South Indian, North Indian",
                ],
                "votes": [1, 2, 3, 4, 5, 6],
            }
        )

        features, _, _ = prepare_data(frame)

        self.assertNotIn("cuisines type", features.columns)
        self.assertEqual(
            features.loc[0, f"{CUISINE_FEATURE_PREFIX}South Indian"], 1
        )
        self.assertEqual(features.loc[0, f"{CUISINE_FEATURE_PREFIX}Chinese"], 1)
        self.assertEqual(features.loc[0, f"{CUISINE_FEATURE_PREFIX}North Indian"], 0)

    def test_feature_metadata_contains_inputs_without_training_rows(self):
        frame = pd.DataFrame(
            {
                "rating": [3.0, 3.2, 3.5, 3.8, 4.0, 4.2],
                "cuisines": ["South Indian, Chinese", "North Indian"] * 3,
                "votes": [1, 2, 3, 4, 5, 6],
                "location": ["North", "South"] * 3,
            }
        )
        features, _, _ = prepare_data(frame)

        metadata = build_feature_metadata(features)

        self.assertEqual(metadata["row_count"], 6)
        self.assertEqual(metadata["feature_columns"], list(features.columns))
        self.assertEqual(metadata["features"]["votes"]["kind"], "numeric")
        self.assertEqual(metadata["features"]["location"]["options"], ["North", "South"])
        self.assertEqual(
            metadata["cuisine_frequency"][f"{CUISINE_FEATURE_PREFIX}Chinese"], 3
        )

    def test_models_fit_and_report_regression_metrics(self):
        frame = pd.DataFrame(
            {
                "rating": [3.0, 3.2, 3.5, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 4.9],
                "votes": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                "cuisines": ["A", "B"] * 5,
                "location": ["North", "South"] * 5,
            }
        )
        features, target, _ = prepare_data(frame)

        fitted_models, metrics, best_model = train_and_evaluate(features, target)

        self.assertEqual(set(fitted_models), {"linear_regression", "random_forest"})
        self.assertIn(best_model, metrics)
        for scores in metrics.values():
            self.assertEqual(set(scores), {"mae", "rmse", "r2"})
            self.assertTrue(all(np.isfinite(value) for value in scores.values()))

    def test_smallest_supported_dataset_has_finite_metrics(self):
        frame = pd.DataFrame(
            {
                "aggregate rating": [3.0, 3.5, 4.0, 4.5, 5.0],
                "votes": [10, 20, 30, 40, 50],
            }
        )
        features, target, target_name = prepare_data(frame)
        _, metrics, _ = train_and_evaluate(features, target)

        self.assertEqual(target_name, "aggregate rating")
        self.assertTrue(
            all(
                np.isfinite(value)
                for scores in metrics.values()
                for value in scores.values()
            )
        )


if __name__ == "__main__":
    unittest.main()
