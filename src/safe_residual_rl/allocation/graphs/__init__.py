"""A3 heterogeneous graph construction with train-only preprocessing."""

from .core import (
    A3Graph,
    FeatureNormalizer,
    FeatureVocabulary,
    GraphTeacherTarget,
    build_a3_graph,
    build_teacher_target,
    fit_feature_normalizer,
    fit_feature_vocabulary,
)
from .dataset import A3GraphRecord, discover_a3_records, load_a3_record

__all__ = [
    "A3Graph",
    "A3GraphRecord",
    "FeatureNormalizer",
    "FeatureVocabulary",
    "GraphTeacherTarget",
    "build_a3_graph",
    "build_teacher_target",
    "discover_a3_records",
    "fit_feature_normalizer",
    "fit_feature_vocabulary",
    "load_a3_record",
]
