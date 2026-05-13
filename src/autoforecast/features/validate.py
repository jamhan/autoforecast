"""Feature validation checks before expensive model evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from autoforecast.features.build import build_feature_matrix
from autoforecast.features.dsl import FeatureSpec, FeatureSpecError, validate_spec


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    note: str


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    checks: dict[str, CheckResult]


def validate_feature_values(
    values: pd.Series,
    target: pd.Series,
    *,
    persistence: pd.Series | None = None,
    active_features: dict[str, pd.Series] | None = None,
    min_non_null_fraction: float = 0.5,
    min_variance: float = 1e-8,
    max_duplicate_corr: float = 0.98,
    leakage_corr: float = 0.999,
    leakage_rmse_ratio: float = 0.80,
) -> ValidationResult:
    checks: dict[str, CheckResult] = {}
    clean = values.dropna()
    if len(values) == 0 or len(clean) / len(values) < min_non_null_fraction:
        checks["coverage"] = CheckResult(False, f"non-null fraction {len(clean) / max(len(values), 1):.1%}")
    else:
        checks["coverage"] = CheckResult(True, f"non-null fraction {len(clean) / len(values):.1%}")

    variance = float(clean.var()) if len(clean) else 0.0
    checks["variance"] = CheckResult(
        variance >= min_variance,
        f"variance {variance:.3g}",
    )

    pair = pd.DataFrame({"y": target, "f": values}).dropna()
    if len(pair) >= 30 and pair["f"].std() > 0:
        corr = abs(float(pair["y"].corr(pair["f"])))
        checks["leakage_corr"] = CheckResult(
            corr < leakage_corr,
            f"|corr(y, feature)|={corr:.4f}",
        )
        if persistence is not None:
            base = pd.DataFrame({"y": target, "p": persistence}).dropna()
            if len(base) >= 30:
                b, a = np.polyfit(pair["f"].to_numpy(), pair["y"].to_numpy(), 1)
                pred = a + b * pair["f"].to_numpy()
                feat_rmse = float(np.sqrt(np.mean((pair["y"].to_numpy() - pred) ** 2)))
                base_rmse = float(np.sqrt(np.mean((base["y"].to_numpy() - base["p"].to_numpy()) ** 2)))
                ratio = feat_rmse / base_rmse if base_rmse > 0 else 1.0
                checks["leakage_strength"] = CheckResult(
                    ratio >= leakage_rmse_ratio,
                    f"feature-alone rmse ratio vs persistence={ratio:.2f}",
                )
    else:
        checks["leakage_corr"] = CheckResult(True, f"too few rows for leakage corr ({len(pair)})")

    worst_name = None
    worst_corr = 0.0
    for name, other in (active_features or {}).items():
        cmp = pd.DataFrame({"a": values, "b": other}).dropna()
        if len(cmp) < 30 or cmp["a"].std() == 0 or cmp["b"].std() == 0:
            continue
        corr = abs(float(cmp["a"].corr(cmp["b"])))
        if corr > worst_corr:
            worst_name, worst_corr = name, corr
    checks["duplicate_signal"] = CheckResult(
        worst_corr <= max_duplicate_corr,
        f"max |corr| vs active={worst_corr:.3f}" + (f" ({worst_name})" if worst_name else ""),
    )

    return ValidationResult(
        passed=all(check.ok for check in checks.values()),
        checks=checks,
    )


def validate_feature_spec(
    frame: pd.DataFrame,
    *,
    spec: FeatureSpec | dict[str, Any],
    name: str,
    date_col: str,
    target_col: str,
    id_cols: list[str] | tuple[str, ...] | None = None,
    active_features: dict[str, pd.Series] | None = None,
) -> ValidationResult:
    feature_spec = spec if isinstance(spec, FeatureSpec) else FeatureSpec.from_dict(spec)
    try:
        validate_spec(feature_spec)
    except FeatureSpecError as exc:
        return ValidationResult(False, {"shape": CheckResult(False, str(exc))})

    matrix_a = build_feature_matrix(
        frame,
        date_col=date_col,
        target_col=target_col,
        id_cols=id_cols,
        specs={name: feature_spec},
    )
    matrix_b = build_feature_matrix(
        frame,
        date_col=date_col,
        target_col=target_col,
        id_cols=id_cols,
        specs={name: feature_spec},
    )
    deterministic = matrix_a[name].fillna(-9.99e99).equals(matrix_b[name].fillna(-9.99e99))
    result = validate_feature_values(
        matrix_a[name],
        matrix_a[target_col],
        persistence=matrix_a[target_col].shift(1),
        active_features=active_features,
    )
    checks = dict(result.checks)
    checks["determinism"] = CheckResult(deterministic, "deterministic" if deterministic else "values changed")
    return ValidationResult(all(check.ok for check in checks.values()), checks)
