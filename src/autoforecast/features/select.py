"""Feature selection utilities."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass


MetricEvaluator = Callable[[Sequence[str]], float]


@dataclass(frozen=True)
class AblationResult:
    feature: str
    metric_with: float
    metric_without: float
    delta: float
    keep: bool


def paired_ablation(
    *,
    champion_features: Sequence[str],
    candidate_features: Sequence[str],
    evaluate: MetricEvaluator,
    min_improvement: float = 0.0,
) -> list[AblationResult]:
    """Evaluate each candidate as champion ± one feature.

    Metrics are assumed lower-is-better. `delta = with - without`, so
    negative is improvement.
    """
    base = list(champion_features)
    results: list[AblationResult] = []
    for feature in candidate_features:
        without = [f for f in base if f != feature]
        with_feature = list(dict.fromkeys(without + [feature]))
        metric_without = float(evaluate(without))
        metric_with = float(evaluate(with_feature))
        delta = metric_with - metric_without
        results.append(
            AblationResult(
                feature=feature,
                metric_with=metric_with,
                metric_without=metric_without,
                delta=delta,
                keep=delta < -abs(min_improvement),
            )
        )
    return sorted(results, key=lambda row: row.delta)


def prune_by_ablation(
    features: Sequence[str],
    evaluate: MetricEvaluator,
    *,
    max_removed: int | None = None,
    tolerance: float = 0.0,
) -> list[str]:
    """Remove features whose absence does not hurt the metric."""
    kept = list(features)
    removed = 0
    for feature in list(features):
        if max_removed is not None and removed >= max_removed:
            break
        metric_with = float(evaluate(kept))
        candidate = [f for f in kept if f != feature]
        metric_without = float(evaluate(candidate))
        if metric_without <= metric_with + tolerance:
            kept = candidate
            removed += 1
    return kept
