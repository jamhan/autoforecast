"""Built-in feature names and parser helpers.

The catalog is intentionally small and boring. These are the features
industry users expect before any agent gets creative.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


CALENDAR_FEATURES = (
    "day_of_week",
    "day_of_year",
    "week_of_year",
    "month",
    "quarter",
    "is_weekend",
)

CYCLICAL_FEATURES = (
    "day_of_week_sin",
    "day_of_week_cos",
    "day_of_year_sin",
    "day_of_year_cos",
)

DEFAULT_LAGS = (1, 2, 3, 7, 14, 28)
DEFAULT_ROLLING_WINDOWS = (7, 14, 28)


@dataclass(frozen=True)
class ParsedFeature:
    kind: str
    params: tuple[object, ...]


def default_feature_names() -> list[str]:
    names: list[str] = list(CALENDAR_FEATURES) + list(CYCLICAL_FEATURES)
    names.extend(f"lag_{lag}" for lag in DEFAULT_LAGS)
    for window in DEFAULT_ROLLING_WINDOWS:
        for agg in ("mean", "std", "min", "max"):
            names.append(f"rolling_{window}_{agg}")
    names.extend(f"diff_{lag}" for lag in (1, 7))
    names.extend(f"pct_change_{lag}" for lag in (1, 7))
    return names


def parse_builtin_name(name: str) -> ParsedFeature | None:
    if name in CALENDAR_FEATURES or name in CYCLICAL_FEATURES:
        return ParsedFeature("calendar", (name,))

    match = re.fullmatch(r"lag_(\d+)", name)
    if match:
        return ParsedFeature("lag", (int(match.group(1)),))

    match = re.fullmatch(r"rolling_(\d+)_(mean|std|min|max)", name)
    if match:
        return ParsedFeature("rolling", (int(match.group(1)), match.group(2)))

    match = re.fullmatch(r"diff_(\d+)", name)
    if match:
        return ParsedFeature("diff", (int(match.group(1)),))

    match = re.fullmatch(r"pct_change_(\d+)", name)
    if match:
        return ParsedFeature("pct_change", (int(match.group(1)),))

    return None
