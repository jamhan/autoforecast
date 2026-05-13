"""Forecast metrics and paired statistical checks."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    return y[mask], p[mask]


def rmse(y_true, y_pred) -> float:
    y, p = _arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((y - p) ** 2))) if len(y) else float("nan")


def mae(y_true, y_pred) -> float:
    y, p = _arrays(y_true, y_pred)
    return float(np.mean(np.abs(y - p))) if len(y) else float("nan")


def mape(y_true, y_pred) -> float:
    y, p = _arrays(y_true, y_pred)
    mask = y != 0
    return float(np.mean(np.abs((y[mask] - p[mask]) / y[mask]))) if mask.any() else float("nan")


def smape(y_true, y_pred) -> float:
    y, p = _arrays(y_true, y_pred)
    denom = (np.abs(y) + np.abs(p)) / 2
    mask = denom != 0
    return float(np.mean(np.abs(y[mask] - p[mask]) / denom[mask])) if mask.any() else float("nan")


@dataclass(frozen=True)
class DieboldMarianoResult:
    p_value: float
    mean_loss_diff: float
    statistic: float
    n: int


def diebold_mariano_squared_error(y_true, candidate_pred, baseline_pred) -> DieboldMarianoResult:
    """One-sided paired test for candidate beating baseline.

    Loss diff is candidate squared error minus baseline squared error.
    Negative mean is good. Guards return p=1.0 when the test is not
    meaningful.
    """
    y = np.asarray(y_true, dtype=float)
    cand = np.asarray(candidate_pred, dtype=float)
    base = np.asarray(baseline_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(cand) & np.isfinite(base)
    diff = (y[mask] - cand[mask]) ** 2 - (y[mask] - base[mask]) ** 2
    n = int(len(diff))
    if n < 5:
        return DieboldMarianoResult(1.0, float("nan"), float("nan"), n)
    mean = float(np.mean(diff))
    if mean >= 0:
        return DieboldMarianoResult(1.0, mean, 0.0, n)
    sd = float(np.std(diff, ddof=1))
    if sd == 0 or not math.isfinite(sd):
        return DieboldMarianoResult(1.0, mean, float("nan"), n)
    stat = mean / (sd / math.sqrt(n))
    if not math.isfinite(stat):
        return DieboldMarianoResult(1.0, mean, stat, n)
    # Normal approximation; good enough for the gate and avoids scipy.
    p_value = 0.5 * math.erfc(abs(stat) / math.sqrt(2))
    return DieboldMarianoResult(float(p_value), mean, float(stat), n)
