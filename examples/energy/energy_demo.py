"""Energy demand demo for autoforecast.

This is a small industry-shaped example: daily peak demand with weather,
calendar, and lag features. It avoids external data and heavyweight model
dependencies so the core repo remains easy to run.
"""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from autoforecast.experiments.journal import Journal
from autoforecast.experiments.metrics import diebold_mariano_squared_error, rmse
from autoforecast.experiments.promote import promotion_gate
from autoforecast.features import FeatureSpec, build_feature_matrix
from autoforecast.features.validate import validate_feature_spec


ROOT = Path(__file__).resolve().parent


def make_energy_panel(n_days: int = 730, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    date = pd.date_range("2024-01-01", periods=n_days, freq="D")
    day = np.arange(n_days)

    annual = np.sin(2 * np.pi * day / 365.25)
    weekly = np.sin(2 * np.pi * day / 7)
    tmax = 25 + 8 * annual + 3 * weekly + rng.normal(0, 1.7, n_days)
    tmin = tmax - 9 + rng.normal(0, 1.0, n_days)
    cdd = np.maximum(tmax - 24, 0)
    hdd = np.maximum(15 - tmin, 0)
    is_weekend = (date.dayofweek >= 5).astype(float)

    demand = (
        7300
        + 120 * cdd
        + 75 * hdd
        - 360 * is_weekend
        + 95 * weekly
        + rng.normal(0, 95, n_days)
    )
    for idx in range(7, n_days):
        demand[idx] = 0.72 * demand[idx] + 0.28 * demand[idx - 7]

    return pd.DataFrame(
        {
            "region": "NSW1",
            "date": date,
            "peak_demand": demand,
            "tmax": tmax,
            "tmin": tmin,
            "cdd": cdd,
            "hdd": hdd,
        }
    )


def fit_linear(train: pd.DataFrame, feature_cols: list[str], target_col: str):
    clean = train.dropna(subset=feature_cols + [target_col])
    x = clean[feature_cols].to_numpy(dtype=float)
    y = clean[target_col].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    return coef


def predict_linear(frame: pd.DataFrame, feature_cols: list[str], coef) -> np.ndarray:
    x = frame[feature_cols].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(x)), x])
    return x @ coef


def main() -> None:
    panel = make_energy_panel()

    weather_shock_spec = FeatureSpec.from_dict(
        {
            "op": "interaction",
            "args": [
                {"op": "raw", "args": ["cdd"]},
                {"op": "raw", "args": ["is_weekend"]},
            ],
        }
    )
    validation = validate_feature_spec(
        panel,
        spec=weather_shock_spec,
        name="cdd_x_weekend",
        date_col="date",
        target_col="peak_demand",
        id_cols=["region"],
    )

    features = [
        "lag_1",
        "lag_7",
        "rolling_7_mean",
        "rolling_28_mean",
        "day_of_week",
        "is_weekend",
        "tmax",
        "cdd",
        "hdd",
    ]
    matrix = build_feature_matrix(
        panel,
        date_col="date",
        target_col="peak_demand",
        id_cols=["region"],
        features=features,
        specs={"cdd_x_weekend": weather_shock_spec},
    )
    feature_cols = features + ["cdd_x_weekend"]

    split_date = matrix["date"].max() - pd.Timedelta(days=90)
    train = matrix[matrix["date"] <= split_date].copy()
    holdout = matrix[matrix["date"] > split_date].dropna(subset=feature_cols + ["peak_demand"]).copy()

    coef = fit_linear(train, feature_cols, "peak_demand")
    model_pred = predict_linear(holdout, feature_cols, coef)
    baseline_pred = holdout["lag_7"].to_numpy(dtype=float)
    y_true = holdout["peak_demand"].to_numpy(dtype=float)

    baseline_rmse = rmse(y_true, baseline_pred)
    model_rmse = rmse(y_true, model_pred)
    dm = diebold_mariano_squared_error(y_true, model_pred, baseline_pred)
    decision = promotion_gate(candidate_metric=model_rmse, baseline_metric=baseline_rmse)

    output_dir = ROOT / "output"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        output_dir = Path(tempfile.gettempdir()) / "autoforecast-energy"
    journal = Journal(output_dir / "journal.sqlite")
    eid = journal.log_experiment(
        target="NSW1_peak_demand",
        family="linear_numpy",
        config={"features": feature_cols, "split": str(split_date.date())},
        feature_set=feature_cols,
        split_signature="synthetic-energy-v1",
        metric_name="rmse",
        hypothesis="Weather and strict-past demand features improve daily peak demand.",
    )
    journal.finish_experiment(eid, status="done", metric_val=model_rmse, metric_holdout=model_rmse)

    print("autoforecast energy demo")
    print(f"rows: train={len(train)} holdout={len(holdout)}")
    print(f"feature validation passed: {validation.passed}")
    print(f"seasonal naive RMSE: {baseline_rmse:,.1f}")
    print(f"feature model RMSE:  {model_rmse:,.1f}")
    print(f"DM p-value:          {dm.p_value:.4f}")
    print(f"promotion:           {decision.accepted} ({decision.reason})")
    print(f"journal:             {journal.path}")


if __name__ == "__main__":
    main()
