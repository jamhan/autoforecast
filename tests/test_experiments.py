import tempfile
import unittest
from pathlib import Path

from autoforecast.experiments.journal import Journal
from autoforecast.experiments.metrics import diebold_mariano_squared_error, rmse
from autoforecast.experiments.promote import promotion_gate


class ExperimentTests(unittest.TestCase):
    def test_rmse(self):
        self.assertAlmostEqual(rmse([1, 2, 3], [1, 2, 5]), 1.154700538, places=6)

    def test_dm_guard_returns_one_when_candidate_not_better(self):
        result = diebold_mariano_squared_error(
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
        )
        self.assertEqual(result.p_value, 1.0)

    def test_promotion_gate(self):
        accepted = promotion_gate(candidate_metric=80, baseline_metric=100)
        rejected = promotion_gate(candidate_metric=98, baseline_metric=100)

        self.assertTrue(accepted.accepted)
        self.assertAlmostEqual(accepted.improvement_pct, 0.20)
        self.assertFalse(rejected.accepted)

    def test_journal_logs_and_finishes_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = Journal(Path(tmp) / "journal.sqlite")
            experiment_id = journal.log_experiment(
                target="sales",
                family="lightgbm",
                config={"learning_rate": 0.05},
                feature_set=["lag_7"],
                split_signature="demo",
                metric_name="rmse",
            )
            journal.finish_experiment(experiment_id, status="done", metric_val=10.0, metric_holdout=12.0)

            with journal.connect() as conn:
                row = conn.execute(
                    "SELECT status, metric_val, metric_holdout FROM experiments WHERE experiment_id = ?",
                    [experiment_id],
                ).fetchone()

            self.assertEqual(row["status"], "done")
            self.assertEqual(row["metric_val"], 10.0)
            self.assertEqual(row["metric_holdout"], 12.0)


if __name__ == "__main__":
    unittest.main()
