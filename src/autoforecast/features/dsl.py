"""Declarative feature DSL.

The DSL gives an agent room to compose features without letting it
write arbitrary Python. Target-derived operations are strict-past.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


ALLOWED_OPS = {
    "raw",
    "lag",
    "rolling_mean",
    "rolling_std",
    "rolling_min",
    "rolling_max",
    "diff",
    "pct_change",
    "log1p",
    "z_score",
    "binary_threshold",
    "interaction",
}
MAX_DEPTH = 4


class FeatureSpecError(ValueError):
    pass


@dataclass(frozen=True)
class FeatureSpec:
    op: str
    args: tuple[Any, ...]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FeatureSpec":
        if not isinstance(payload, dict):
            raise FeatureSpecError("feature spec must be a dict")
        op = payload.get("op")
        args = payload.get("args")
        if not isinstance(op, str) or not isinstance(args, list):
            raise FeatureSpecError("feature spec requires string op and list args")
        return cls(op=op, args=tuple(_coerce_arg(a) for a in args))

    def to_dict(self) -> dict[str, Any]:
        return {"op": self.op, "args": [_arg_to_dict(a) for a in self.args]}


def _coerce_arg(value: Any) -> Any:
    if isinstance(value, dict) and "op" in value:
        return FeatureSpec.from_dict(value)
    return value


def _arg_to_dict(value: Any) -> Any:
    if isinstance(value, FeatureSpec):
        return value.to_dict()
    return value


def validate_spec(spec: FeatureSpec, depth: int = 0) -> None:
    if depth > MAX_DEPTH:
        raise FeatureSpecError(f"spec depth exceeds {MAX_DEPTH}")
    if spec.op not in ALLOWED_OPS:
        raise FeatureSpecError(f"unknown op {spec.op!r}")
    args = spec.args
    if spec.op == "raw":
        if len(args) != 1 or not isinstance(args[0], str):
            raise FeatureSpecError("raw expects [column_or_builtin_name]")
    elif spec.op == "lag":
        _expect_window(args, "lag", minimum=1, maximum=365)
    elif spec.op.startswith("rolling_"):
        _expect_window(args, spec.op, minimum=2, maximum=365)
    elif spec.op in {"diff", "pct_change"}:
        _expect_window(args, spec.op, minimum=1, maximum=365)
    elif spec.op in {"log1p"}:
        _expect_nested(args, spec.op, 1, depth)
    elif spec.op == "z_score":
        if len(args) != 2 or not isinstance(args[0], FeatureSpec) or not isinstance(args[1], int):
            raise FeatureSpecError("z_score expects [spec, window]")
        validate_spec(args[0], depth + 1)
        if args[1] < 7 or args[1] > 365:
            raise FeatureSpecError("z_score window must be in [7, 365]")
    elif spec.op == "binary_threshold":
        if len(args) != 3 or not isinstance(args[0], FeatureSpec):
            raise FeatureSpecError("binary_threshold expects [spec, comparator, threshold]")
        validate_spec(args[0], depth + 1)
        if args[1] not in {"gt", "ge", "lt", "le", "eq"}:
            raise FeatureSpecError("unknown comparator")
        if not isinstance(args[2], (int, float)):
            raise FeatureSpecError("threshold must be numeric")
    elif spec.op == "interaction":
        _expect_nested(args, spec.op, 2, depth)


def _expect_window(args: tuple[Any, ...], op: str, *, minimum: int, maximum: int) -> None:
    if len(args) != 1 or not isinstance(args[0], int):
        raise FeatureSpecError(f"{op} expects [window]")
    if args[0] < minimum or args[0] > maximum:
        raise FeatureSpecError(f"{op} window must be in [{minimum}, {maximum}]")


def _expect_nested(args: tuple[Any, ...], op: str, count: int, depth: int) -> None:
    if len(args) != count or not all(isinstance(arg, FeatureSpec) for arg in args):
        raise FeatureSpecError(f"{op} expects {count} nested spec(s)")
    for arg in args:
        validate_spec(arg, depth + 1)


def evaluate_spec(
    spec: FeatureSpec,
    frame: pd.DataFrame,
    *,
    date_col: str,
    target_col: str,
    id_cols: list[str] | tuple[str, ...] | None = None,
) -> pd.Series:
    validate_spec(spec)
    from autoforecast.features.build import build_builtin_feature

    keys = list(id_cols or [])

    def lag(window: int) -> pd.Series:
        if not keys:
            return frame[target_col].shift(window)
        return frame.groupby(keys, sort=False, dropna=False)[target_col].shift(window)

    def rolling(window: int, agg: str) -> pd.Series:
        shifted = lag(1)
        if not keys:
            rolled = shifted.rolling(window=window, min_periods=max(2, window // 3))
        else:
            rolled = shifted.groupby([frame[k] for k in keys], sort=False, dropna=False).rolling(
                window=window,
                min_periods=max(2, window // 3),
            )
        values = getattr(rolled, agg)()
        if keys:
            values = values.reset_index(level=list(range(len(keys))), drop=True)
        return values.reindex(frame.index)

    if spec.op == "raw":
        return build_builtin_feature(
            frame,
            str(spec.args[0]),
            date_col=date_col,
            target_col=target_col,
            id_cols=id_cols,
        )
    if spec.op == "lag":
        return lag(int(spec.args[0]))
    if spec.op == "rolling_mean":
        return rolling(int(spec.args[0]), "mean")
    if spec.op == "rolling_std":
        return rolling(int(spec.args[0]), "std")
    if spec.op == "rolling_min":
        return rolling(int(spec.args[0]), "min")
    if spec.op == "rolling_max":
        return rolling(int(spec.args[0]), "max")
    if spec.op == "diff":
        return frame[target_col] - lag(int(spec.args[0]))
    if spec.op == "pct_change":
        base = lag(int(spec.args[0]))
        return (frame[target_col] - base) / base.replace(0, np.nan)
    if spec.op == "log1p":
        inner = evaluate_spec(spec.args[0], frame, date_col=date_col, target_col=target_col, id_cols=id_cols)
        return inner.where(inner >= -1).map(np.log1p)
    if spec.op == "z_score":
        inner = evaluate_spec(spec.args[0], frame, date_col=date_col, target_col=target_col, id_cols=id_cols)
        window = int(spec.args[1])
        shifted = inner.shift(1) if not keys else inner.groupby([frame[k] for k in keys], sort=False, dropna=False).shift(1)
        if not keys:
            rolled = shifted.rolling(window=window, min_periods=max(5, window // 3))
            mean = rolled.mean()
            std = rolled.std(ddof=0)
        else:
            rolled = shifted.groupby([frame[k] for k in keys], sort=False, dropna=False).rolling(
                window=window,
                min_periods=max(5, window // 3),
            )
            mean = rolled.mean().reset_index(level=list(range(len(keys))), drop=True).reindex(frame.index)
            std = rolled.std(ddof=0).reset_index(level=list(range(len(keys))), drop=True).reindex(frame.index)
        return (inner - mean) / std.replace(0, np.nan)
    if spec.op == "binary_threshold":
        inner = evaluate_spec(spec.args[0], frame, date_col=date_col, target_col=target_col, id_cols=id_cols)
        comparator, threshold = str(spec.args[1]), float(spec.args[2])
        if comparator == "gt":
            mask = inner > threshold
        elif comparator == "ge":
            mask = inner >= threshold
        elif comparator == "lt":
            mask = inner < threshold
        elif comparator == "le":
            mask = inner <= threshold
        else:
            mask = inner == threshold
        return mask.astype(float).where(inner.notna())
    if spec.op == "interaction":
        a = evaluate_spec(spec.args[0], frame, date_col=date_col, target_col=target_col, id_cols=id_cols)
        b = evaluate_spec(spec.args[1], frame, date_col=date_col, target_col=target_col, id_cols=id_cols)
        return a * b
    raise FeatureSpecError(f"unhandled op {spec.op!r}")
