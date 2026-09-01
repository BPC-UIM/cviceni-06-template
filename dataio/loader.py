"""Nacitani dat Breast Cancer Wisconsin pro cviceni 06.

Modul poskytuje jedinou funkci ``load_breast_cancer_data``, ktera vrati
priznakovou matici, binarni cilovou promennou a nazvy priznaku.

Na rozdil od cviceni 05 se zde data **nestandardizuji**. Standardizace se
v cviceni 06 provadi az PO rozdeleni na trenovaci a testovaci mnozinu
(viz ``dataio.preprocessing.standardize_fit`` / ``standardize_apply``),
aby bylo mozne demonstrovat unik dat (data leakage).
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_breast_cancer


def load_breast_cancer_data(
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Nacte dataset Breast Cancer Wisconsin a vrati ``(X, y, feature_names)``.

    Parametry
    ---------
    random_state:
        Je v podpisu pouze kvuli konzistenci s ostatnimi nacitaci funkcemi
        v kurzu. Dataset je pevny a deterministicky, takze tento parametr
        nic neovlivnuje (zadne michani ani vzorkovani se zde nedeje).

    Navratova hodnota
    -----------------
    X:
        ``np.ndarray`` tvaru ``(569, 30)`` typu ``float64``. Priznaky se
        zamerne **nestandardizuji** -- viz modul-docstring.
    y:
        ``np.ndarray`` tvaru ``(569,)`` typu ``int64`` s hodnotami
        ``{0, 1}``, kde **1 = maligni (zhoubny)** a **0 = benigni
        (nezhoubny)**. Plati ``y.sum() == 212`` (pocet malignich vzorku).
    feature_names:
        Seznam 30 nazvu priznaku (``list[str]``).

    Poznamka ke kodovani cilove promenne
    ------------------------------------
    ``sklearn.datasets.load_breast_cancer`` koduje ``target`` **opacne**,
    nez potrebujeme: 0 = malignant (zhoubny), 1 = benign (nezhoubny).
    Kurz vsak pracuje s konvenci "vystup 1 -> maligni", proto se stitky
    prohazuji vztahem ``y = 1 - dataset.target``. Po prohozeni plati
    ``y.sum() == 212``.

    Pozitivni trida pro Se/Sp
    -------------------------
    Ve vyhodnoceni klasifikace je **pozitivni tridou maligni nador
    (y = 1)**. Z toho plyne cteni metrik:

    - ``Se = recall = sensitivita = TPR`` -- podil spravne odhalenych
      malignich nadoru; "nepropasnout zhoubny nador".
    - ``Sp = specificita = TNR`` -- podil spravne oznacenych benignich
      pripadu.

    V medicinskem kontextu obvykle sensitivita (Se) prevazuje nad
    precizi -- cena za propasnuty maligni nador je vyssi nez za falesny
    poplach.
    """
    dataset = load_breast_cancer()
    x = np.asarray(dataset.data, dtype=np.float64)
    # Prohozeni stitku: sklearn ma 0 = malignant, 1 = benign; kurz chce 1 = maligni.
    y = 1 - np.asarray(dataset.target, dtype=np.int64)
    feature_names = [str(name) for name in dataset.feature_names]

    return x, y, feature_names
