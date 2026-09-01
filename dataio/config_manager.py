"""Typovana sprava konfigurace nad ``config.yaml`` pro cviceni 06.

Modul definuje vnorene dataclassy odpovidajici sekcim ``config.yaml`` a
dve funkce: ``load_config`` (naparsuje YAML, sestavi dataclassy, zvaliduje
a vrati) a ``validate_config`` (rozsahove kontroly s ceskymi chybovymi
hlaskami).

K hodnotam se pristupuje pres atributy (napr. ``cfg.knn.k``), nikdy ne
pres klice slovniku.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import yaml


@dataclass
class DataConfig:
    """Nastaveni dat (sekce ``data`` v ``config.yaml``)."""

    imbalance_ratio: float
    random_state: int


@dataclass
class KNNConfig:
    """Nastaveni kNN (sekce ``knn``)."""

    k: int
    k_values: list[int]


@dataclass
class ValidationConfig:
    """Nastaveni validace modelu (sekce ``validation``)."""

    test_size: float
    n_folds: int
    n_bootstrap: int


@dataclass
class FeatureSelectionConfig:
    """Nastaveni experimentu s poctem priznaku (sekce ``feature_selection``)."""

    n_features_grid: list[int]


@dataclass
class ExperimentConfig:
    """Korenova konfigurace experimentu slozena ze vsech dilcich sekci."""

    data: DataConfig
    knn: KNNConfig
    validation: ValidationConfig
    feature_selection: FeatureSelectionConfig


def load_config(filepath: str = "config.yaml") -> ExperimentConfig:
    """Nacte a zvaliduje konfiguraci z YAML souboru.

    Parametry
    ---------
    filepath:
        Cesta k YAML souboru s konfiguraci.

    Navratova hodnota
    -----------------
    ``ExperimentConfig`` s vnorenymi dataclassami ``DataConfig``,
    ``KNNConfig``, ``ValidationConfig`` a ``FeatureSelectionConfig``.

    Vyjimky
    -------
    ``FileNotFoundError``:
        Pokud soubor neexistuje.
    ``ValueError``:
        Pokud nektera hodnota nesplnuje rozsahove kontroly ve
        ``validate_config``.
    """
    with open(filepath, "r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)

    cfg = ExperimentConfig(
        data=DataConfig(
            imbalance_ratio=float(raw["data"]["imbalance_ratio"]),
            random_state=int(raw["data"]["random_state"]),
        ),
        knn=KNNConfig(
            k=int(raw["knn"]["k"]),
            k_values=[int(v) for v in raw["knn"]["k_values"]],
        ),
        validation=ValidationConfig(
            test_size=float(raw["validation"]["test_size"]),
            n_folds=int(raw["validation"]["n_folds"]),
            n_bootstrap=int(raw["validation"]["n_bootstrap"]),
        ),
        feature_selection=FeatureSelectionConfig(
            n_features_grid=[int(v) for v in raw["feature_selection"]["n_features_grid"]],
        ),
    )

    validate_config(cfg)
    return cfg


def validate_config(cfg: ExperimentConfig) -> None:
    """Zkontroluje rozsahy hodnot v konfiguraci.

    Pri poruseni nektere podminky vyhodi ``ValueError`` se srozumitelnou
    ceskou hlaskou. Kontroluji se:

    - ``0 < imbalance_ratio < 0.5``
    - ``k >= 1``
    - vsechny hodnoty v ``k_values`` jsou ``>= 1``
    - ``0 < test_size < 1``
    - ``n_folds >= 2``
    - ``n_bootstrap >= 1``
    - kazda polozka ``n_features_grid`` je v rozsahu ``1..30``

    Navratova hodnota je ``None`` -- funkce pouze validuje.
    """
    data = cfg.data
    knn = cfg.knn
    validation = cfg.validation
    fs = cfg.feature_selection

    if not 0.0 < data.imbalance_ratio < 0.5:
        raise ValueError(
            f"imbalance_ratio musi byt v intervalu (0, 0.5), zadano: {data.imbalance_ratio}"
        )
    if knn.k < 1:
        raise ValueError(f"k musi byt >= 1, zadano: {knn.k}")
    if any(v < 1 for v in knn.k_values):
        raise ValueError(
            f"vsechny hodnoty k_values musi byt >= 1, zadano: {knn.k_values}"
        )
    if not 0.0 < validation.test_size < 1.0:
        raise ValueError(
            f"test_size musi byt v intervalu (0, 1), zadano: {validation.test_size}"
        )
    if validation.n_folds < 2:
        raise ValueError(
            f"n_folds musi byt >= 2, zadano: {validation.n_folds}"
        )
    if validation.n_bootstrap < 1:
        raise ValueError(
            f"n_bootstrap musi byt >= 1, zadano: {validation.n_bootstrap}"
        )
    if any(not 1 <= v <= 30 for v in fs.n_features_grid):
        raise ValueError(
            f"kazda polozka n_features_grid musi byt v rozsahu 1..30, zadano: {fs.n_features_grid}"
        )
