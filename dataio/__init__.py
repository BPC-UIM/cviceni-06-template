"""Verejne API balicku ``dataio`` pro cviceni 06.

Nacitani dat (Breast Cancer Wisconsin), predzpracovani (standardizace fit/apply,
podvzorkovani pro nevyvazenost, demonstrace uniku dat), typovana sprava
konfigurace a bohate vykreslovani.

I/O, konfigurace a grafy jsou PREDVYPLNENE. V ``dataio/preprocessing.py`` jsou
tri studentske ukoly (``standardize_fit``, ``standardize_apply``,
``subsample_imbalance``); ``demonstrate_leakage`` zustava predvyplnena.

**Tento soubor (__init__.py) neupravujte** — re-exporty verejneho API zustavaji
beze zmeny.
"""

from __future__ import annotations

from dataio.config_manager import (
    DataConfig,
    ExperimentConfig,
    FeatureSelectionConfig,
    KNNConfig,
    ValidationConfig,
    load_config,
    validate_config,
)
from dataio.loader import load_breast_cancer_data
from dataio.plotting import (
    plot_confusion_matrix,
    plot_cv_scores,
    plot_decision_boundary_2d,
    plot_feature_tradeoff,
    plot_overfitting_curve,
    plot_roc_curve,
)
from dataio.preprocessing import (
    demonstrate_leakage,
    standardize_apply,
    standardize_fit,
    subsample_imbalance,
)

__all__ = [
    "load_breast_cancer_data",
    "standardize_fit",
    "standardize_apply",
    "subsample_imbalance",
    "demonstrate_leakage",
    "load_config",
    "validate_config",
    "DataConfig",
    "KNNConfig",
    "ValidationConfig",
    "FeatureSelectionConfig",
    "ExperimentConfig",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_overfitting_curve",
    "plot_cv_scores",
    "plot_feature_tradeoff",
    "plot_decision_boundary_2d",
]
