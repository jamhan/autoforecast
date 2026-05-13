"""Strict-past feature matrix builder."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from autoforecast.features.catalog import parse_builtin_name
from autoforecast.features.dsl import FeatureSpec, evaluate_spec


def _group_keys(id_cols: Sequence[str] | None) -> list[str]:
    return list(id_cols or [])


def _sorted_frame(
    frame: pd.DataFrame,
    *,
    date_col: str,
    id_cols: Sequence[str] | None,
) -> pd.DataFrame:
    keys = _group_keys(id_cols) + [date_col]
    out = frame.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    return out.sort_values(keys).reset_index(drop=True)


def _series_grouped(
    frame: pd.DataFrame,
    target_col: str,
    id_cols: Sequence[str] | None,
):
    keys = _group_keys(id_cols)
    if not keys:
        return [(None, frame[target_col])]
    return frame.groupby(keys, sort=False, dropna=False)[target_col]


def _lag(frame: pd.DataFrame, target_col: str, id_cols: Sequence[str] | None, lag: int) -> pd.Series:
    keys = _group_keys(id_cols)
    if not keys:
        return frame[target_col].shift(lag)
    return frame.groupby(keys, sort=False, dropna=False)[target_col].shift(lag)


def _rolling(
    frame: pd.DataFrame,
    target_col: str,
    id_cols: Sequence[str] | None,
    window: int,
    agg: str,
) -> pd.Series:
    shifted = _lag(frame, target_col, id_cols, 1)
    keys = _group_keys(id_cols)
    if not keys:
        rolled = shifted.rolling(window=window, min_periods=max(2, window // 3))
    else:
        rolled = shifted.groupby([frame[k] for k in keys], sort=False, dropna=False).rolling(
            window=window,
            min_periods=max(2, window // 3),
        )
    if agg == "mean":
        values = rolled.mean()
    elif agg == "std":
        values = rolled.std(ddof=0)
    elif agg == "min":
        values = rolled.min()
    elif agg == "max":
        values = rolled.max()
    else:
        raise ValueError(f"unknown rolling aggregation {agg!r}")
    if keys:
        values = values.reset_index(level=list(range(len(keys))), drop=True)
    return values.reindex(frame.index)


def _calendar_feature(frame: pd.DataFrame, date_col: str, name: str) -> pd.Series:
    dt = pd.to_datetime(frame[date_col])
    if name == "day_of_week":
        return dt.dt.dayofweek.astype(float)
    if name == "day_of_year":
        return dt.dt.dayofyear.astype(float)
    if name == "week_of_year":
        return dt.dt.isocalendar().week.astype(float)
    if name == "month":
        return dt.dt.month.astype(float)
    if name == "quarter":
        return dt.dt.quarter.astype(float)
    if name == "is_weekend":
        return (dt.dt.dayofweek >= 5).astype(float)
    if name == "day_of_week_sin":
        return np.sin(2 * math.pi * dt.dt.dayofweek / 7)
    if name == "day_of_week_cos":
        return np.cos(2 * math.pi * dt.dt.dayofweek / 7)
    if name == "day_of_year_sin":
        return np.sin(2 * math.pi * dt.dt.dayofyear / 365.25)
    if name == "day_of_year_cos":
        return np.cos(2 * math.pi * dt.dt.dayofyear / 365.25)
    raise ValueError(f"unknown calendar feature {name!r}")


def build_builtin_feature(
    frame: pd.DataFrame,
    name: str,
    *,
    date_col: str,
    target_col: str,
    id_cols: Sequence[str] | None = None,
) -> pd.Series:
    parsed = parse_builtin_name(name)
    if parsed is None:
        if name in frame.columns:
            return frame[name]
        raise KeyError(f"unknown feature {name!r}")

    if parsed.kind == "calendar":
        return _calendar_feature(frame, date_col, str(parsed.params[0]))
    if parsed.kind == "lag":
        return _lag(frame, target_col, id_cols, int(parsed.params[0]))
    if parsed.kind == "rolling":
        return _rolling(frame, target_col, id_cols, int(parsed.params[0]), str(parsed.params[1]))
    if parsed.kind == "diff":
        lag = int(parsed.params[0])
        return frame[target_col] - _lag(frame, target_col, id_cols, lag)
    if parsed.kind == "pct_change":
        lag = int(parsed.params[0])
        base = _lag(frame, target_col, id_cols, lag)
        return (frame[target_col] - base) / base.replace(0, np.nan)
    raise ValueError(f"unhandled parsed feature {parsed}")


def build_feature_matrix(
    frame: pd.DataFrame,
    *,
    date_col: str,
    target_col: str,
    id_cols: Sequence[str] | None = None,
    features: Iterable[str] | None = None,
    specs: dict[str, FeatureSpec | dict] | None = None,
    include_target: bool = True,
) -> pd.DataFrame:
    """Build a feature matrix with strict-past target-derived features.

    `features` names built-ins or existing non-target columns. `specs`
    supplies DSL-generated columns. The returned frame preserves the
    identifying columns, date, optional target, and generated columns.
    """
    sorted_frame = _sorted_frame(frame, date_col=date_col, id_cols=id_cols)
    keys = _group_keys(id_cols)
    out_cols = keys + [date_col]
    if include_target:
        out_cols.append(target_col)
    out = sorted_frame[out_cols].copy()

    for name in features or []:
        out[name] = build_builtin_feature(
            sorted_frame,
            name,
            date_col=date_col,
            target_col=target_col,
            id_cols=id_cols,
        ).to_numpy()

    for name, spec in (specs or {}).items():
        feature_spec = spec if isinstance(spec, FeatureSpec) else FeatureSpec.from_dict(spec)
        out[name] = evaluate_spec(
            feature_spec,
            sorted_frame,
            date_col=date_col,
            target_col=target_col,
            id_cols=id_cols,
        ).to_numpy()

    return out
