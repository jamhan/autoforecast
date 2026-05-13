import math
import unittest

import pandas as pd

from autoforecast.features import FeatureSpec, build_feature_matrix
from autoforecast.features.dsl import FeatureSpecError, validate_spec
from autoforecast.features.validate import validate_feature_spec


class FeatureBuildTests(unittest.TestCase):
    def test_lag_and_rolling_are_strict_past(self):
        df = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=6, freq="D"),
                "y": [10, 20, 30, 40, 50, 60],
            }
        )

        matrix = build_feature_matrix(
            df,
            date_col="date",
            target_col="y",
            features=["lag_1", "rolling_3_mean", "diff_1"],
        )

        self.assertTrue(math.isnan(matrix.loc[0, "lag_1"]))
        self.assertEqual(matrix.loc[1, "lag_1"], 10)
        self.assertEqual(matrix.loc[3, "rolling_3_mean"], 20)
        self.assertEqual(matrix.loc[3, "diff_1"], 10)

    def test_grouped_lags_do_not_cross_entities(self):
        df = pd.DataFrame(
            {
                "store": ["a", "a", "b", "b"],
                "date": pd.to_datetime(["2025-01-01", "2025-01-02"] * 2),
                "y": [1, 2, 100, 200],
            }
        )

        matrix = build_feature_matrix(
            df,
            date_col="date",
            target_col="y",
            id_cols=["store"],
            features=["lag_1"],
        )

        self.assertTrue(math.isnan(matrix.loc[0, "lag_1"]))
        self.assertEqual(matrix.loc[1, "lag_1"], 1)
        self.assertTrue(math.isnan(matrix.loc[2, "lag_1"]))
        self.assertEqual(matrix.loc[3, "lag_1"], 100)

    def test_dsl_composes_builtin_features(self):
        df = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=5, freq="D"),
                "y": [1, 2, 4, 8, 16],
            }
        )
        spec = FeatureSpec.from_dict(
            {
                "op": "interaction",
                "args": [
                    {"op": "lag", "args": [1]},
                    {"op": "binary_threshold", "args": [{"op": "raw", "args": ["day_of_week"]}, "ge", 2]},
                ],
            }
        )

        matrix = build_feature_matrix(
            df,
            date_col="date",
            target_col="y",
            specs={"lag_x_late_week": spec},
        )

        self.assertTrue(math.isnan(matrix.loc[0, "lag_x_late_week"]))
        self.assertEqual(matrix.loc[1, "lag_x_late_week"], 1)

    def test_validator_rejects_malformed_spec(self):
        with self.assertRaises(FeatureSpecError):
            validate_spec(FeatureSpec.from_dict({"op": "rolling_mean", "args": [1]}))

    def test_validator_accepts_basic_safe_feature(self):
        df = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=80, freq="D"),
                "y": list(range(80)),
            }
        )
        result = validate_feature_spec(
            df,
            spec={"op": "rolling_mean", "args": [7]},
            name="r7",
            date_col="date",
            target_col="y",
        )
        self.assertTrue(result.checks["determinism"].ok)
        self.assertTrue(result.checks["coverage"].ok)


if __name__ == "__main__":
    unittest.main()
