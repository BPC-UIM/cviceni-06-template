"""Verejne API balicku ``src`` pro cviceni 06.

Obsahuje:

* ``Distance`` a jeji potomky — brana z cviceni 01 (kNN hleda sousedy pres
  ``calculate``); v tomto cviceni je to STUB k doplneni.
* ``KNNClassifier`` — induktivni klasifikator k nejblizsich sousedu
  (jedina studentska metoda: ``predict``).
* ``confusion_matrix`` a metriky z ni ODVOZENE (``accuracy``, ``precision``,
  ``recall_sensitivity``, ``specificity``, ``f1_score``) — vse bere kontingecni
  tabulku, ne surove vektory popisku.
* ``Validator`` (ABC) a jeho tri zamenne strategie ``HoldOut`` / ``KFold`` /
  ``Bootstrap`` plus orchestracni funkce ``cross_validate``.

**Tento soubor neupravujte.**
"""

from __future__ import annotations

from src.distance import (
    CosineCoeficient,
    Distance,
    EuclideanDistance,
    ManhattanDistance,
)
from src.knn import KNNClassifier
from src.metrics import (
    accuracy,
    confusion_matrix,
    f1_score,
    precision,
    recall_sensitivity,
    specificity,
)
from src.validation import Bootstrap, HoldOut, KFold, Validator, cross_validate

__all__ = [
    "Distance",
    "EuclideanDistance",
    "ManhattanDistance",
    "CosineCoeficient",
    "KNNClassifier",
    "confusion_matrix",
    "accuracy",
    "precision",
    "recall_sensitivity",
    "specificity",
    "f1_score",
    "Validator",
    "HoldOut",
    "KFold",
    "Bootstrap",
    "cross_validate",
]
