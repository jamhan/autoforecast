"""Promotion gate for candidate forecasting configs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromotionDecision:
    accepted: bool
    reason: str
    improvement_pct: float


def promotion_gate(
    *,
    candidate_metric: float | None,
    baseline_metric: float | None,
    current_champion_metric: float | None = None,
    min_baseline_improvement_pct: float = 0.05,
    hysteresis_pct: float = 0.02,
) -> PromotionDecision:
    """Return whether a lower-is-better candidate should be champion."""
    if candidate_metric is None:
        return PromotionDecision(False, "candidate has no holdout metric", 0.0)
    if baseline_metric is None:
        return PromotionDecision(False, "baseline has no holdout metric", 0.0)
    if baseline_metric <= 0:
        return PromotionDecision(False, "baseline metric must be positive", 0.0)
    improvement_pct = (baseline_metric - candidate_metric) / baseline_metric
    if improvement_pct < min_baseline_improvement_pct:
        return PromotionDecision(
            False,
            f"candidate improves baseline by {improvement_pct:.1%}, below {min_baseline_improvement_pct:.1%}",
            improvement_pct,
        )
    if current_champion_metric is not None:
        champion_gap = (current_champion_metric - candidate_metric) / current_champion_metric
        if champion_gap < hysteresis_pct:
            return PromotionDecision(
                False,
                f"candidate beats current champion by {champion_gap:.1%}, below hysteresis {hysteresis_pct:.1%}",
                improvement_pct,
            )
    return PromotionDecision(True, "accepted", improvement_pct)
