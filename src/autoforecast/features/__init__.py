"""Feature engineering primitives."""

from autoforecast.features.build import build_feature_matrix
from autoforecast.features.dsl import FeatureSpec, FeatureSpecError

__all__ = ["FeatureSpec", "FeatureSpecError", "build_feature_matrix"]
