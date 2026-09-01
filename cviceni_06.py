# -*- coding: utf-8 -*-

"""
Created on 30. 08. 2026 at 21:51:03

Author: Richard Redina
Email: 195715@vut.cz
Affiliation:
         International Clinical Research Center, Brno
         Brno University of Technology, Brno
GitHub: RicRedi

(._.)
 <|>
_/|_

Description:
 Popis:
   Vstupni bod ulohy. Nacte konfiguraci a data (Breast Cancer Wisconsin) a
   projde peti fazemi, ktere dohromady tvori "metodicky" pohled na strojove
   uceni s ucitelem:

     1. prechod k uceni s ucitelem: kNN + jedno rozdeleni na train/test,
        + save/load natrenovaneho modelu do .npz (model = ulozena trenovaci mnozina),
     2. spravna prace s daty: unik dat (data leakage) a nevyvazenost trid,
     3. vyhodnoceni z kontingecni tabulky: accuracy / precision / Se / Sp / F1
        a ROC krivka + AUC (referencni sklearn kNN),
     4. robustni validace: k-fold, bootstrap a krivka preuceni pres ``k``,
     5. navaznost na cviceni 05: pocet priznaku vs. presnost a cas predikce.

   Repozitar bezi v kazde fazi. Dokud nejsou ukoly hotove, kazda faze se
   zastavi jen hlaskou "Úkol: ..." — nikdy nezpracovanym tracebackem.
================================================================================
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

# --- Import guard: srozumitelna hlaska misto holeho ImportError -----------------
try:
    from dataio.config_manager import load_config
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
    from src.distance import EuclideanDistance
    from src.knn import KNNClassifier
    from src.metrics import (
        accuracy,
        confusion_matrix,
        f1_score,
        precision,
        recall_sensitivity,
        specificity,
    )
    from src.validation import Bootstrap, HoldOut, KFold, cross_validate
except ImportError as exc:  # pragma: no cover - jen ochranna hlaska
    print(f"[CHYBA IMPORTU] Nepodarilo se nacist moduly projektu: {exc}")
    print("Zkontrolujte, ze spoustite skript z korene repozitare a mate "
          "nainstalovane zavislosti (pip install -r requirements.txt).")
    sys.exit(1)

GRAPHS_DIR = "graphs"       # vystupni grafy (.png)
MODELS_DIR = "models"       # natrenovany a ulozeny kNN model (.npz)
MODEL_PATH = f"{MODELS_DIR}/knn_model.npz"


def _banner(text: str) -> None:
    """Vypise oddelovaci nadpis faze pipeline."""
    print("\n" + "=" * 78)
    print(f"  {text}")
    print("=" * 78)


def _faze_neni_hotova(exc: NotImplementedError) -> None:
    """Vypise pratelskou hlasku, kdyz faze narazi na nedokonceny ukol."""
    print(f"  [NENI HOTOVO] {exc}")
    print("  -> Tuto cast dokoncite v ramci ukolu; pipeline pokracuje dal.")


def _holdout_standardized(
    cfg, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Jeden ``HoldOut`` split + standardizace se scalerem naucenym JEN z trenovaci casti.

    Vraci ``(X_train_std, X_test_std, y_train, y_test)``. Toto je spravny vzor
    (kontrast s unikem dat ve fazi 2): parametry standardizace se pocitaji
    pouze z trenovacich radku a pak se stejne aplikuji na test.
    """
    train_idx, test_idx = next(iter(
        HoldOut(cfg.validation.test_size, random_state=cfg.data.random_state).split(x, y)
    ))
    params = standardize_fit(x[train_idx])
    x_train_std = standardize_apply(x[train_idx], params)
    x_test_std = standardize_apply(x[test_idx], params)
    return x_train_std, x_test_std, y[train_idx], y[test_idx]


def faze_supervised_knn(cfg, x: np.ndarray, y: np.ndarray, names: list[str]) -> None:
    """Faze 1 — uceni s ucitelem pres kNN a jedno rozdeleni na train/test."""
    _banner("Faze 1: Uceni s ucitelem + kNN (jedno rozdeleni HoldOut)")

    dist = EuclideanDistance()
    try:
        x_tr, x_te, y_tr, y_te = _holdout_standardized(cfg, x, y)
        print(f"  HoldOut: train {x_tr.shape[0]} / test {x_te.shape[0]} vzorku "
              f"(test_size={cfg.validation.test_size}), {x.shape[1]} priznaku.")
        print("  Testovaci mnozina se drzi stranou — model ji pri uceni nikdy nevidi.")

        knn = KNNClassifier(cfg.knn.k, dist).fit(x_tr, y_tr)
        y_pred = knn.predict(x_te)
        cm = confusion_matrix(y_te, y_pred)
        print(f"  kNN (k={cfg.knn.k}, eukleidovska vzdalenost) presnost na testu: "
              f"{accuracy(cm):.3f}")
    except NotImplementedError as exc:
        _faze_neni_hotova(exc)


def faze_persistence(cfg, x: np.ndarray, y: np.ndarray) -> None:
    """Model = ulozene naucene parametry: save -> load -> overeni shody predikci.

    U kNN je "natrenovanym modelem" cela trenovaci mnozina (``x_train_``,
    ``y_train_``) plus hyperparametr ``k`` — kontrast s hrstkou malych poli
    natrenovane PCA z cviceni 05. Metrika ``Distance`` se do znovunacteneho
    modelu injektuje az pri vytvoreni instance, do souboru se neuklada.
    """
    _banner("Model jako ulozene parametry: save / load (.npz)")
    try:
        x_tr, x_te, y_tr, y_te = _holdout_standardized(cfg, x, y)
        knn = KNNClassifier(cfg.knn.k, EuclideanDistance()).fit(x_tr, y_tr)
        y_pred_puvod = knn.predict(x_te)

        os.makedirs(MODELS_DIR, exist_ok=True)
        knn.save(MODEL_PATH)
        obnovena = KNNClassifier(cfg.knn.k, EuclideanDistance()).load(MODEL_PATH)
        y_pred_obnov = obnovena.predict(x_te)

        shoda = bool(np.array_equal(np.asarray(y_pred_puvod).ravel(),
                                    np.asarray(y_pred_obnov).ravel()))
        print(f"  Model ulozen do {MODEL_PATH} ({x_tr.shape[0]} trenovacich vzorku), "
              f"znovu nacten.")
        print(f"  restored.predict(x_te) == puvodni predikce:  {shoda}")
        print("  -> 'nauc ted / pouzij pozdeji': u kNN model NESE celou trenovaci mnozinu.")
    except NotImplementedError as exc:
        _faze_neni_hotova(exc)


def faze_data_handling(cfg, x: np.ndarray, y: np.ndarray, names: list[str]) -> None:
    """Faze 2 — unik dat (leakage) a nevyvazenost trid."""
    _banner("Faze 2: Prace s daty — unik dat a nevyvazenost trid")

    # --- Unik dat: standardizace PRED vs. PO rozdeleni (PREDVYPLNENO) ----------
    try:
        leak = demonstrate_leakage(x, y, cfg.knn.k, cfg.data.random_state)
        print(f"  Unik dat (k={leak['k']}, test_size={leak['test_size']}):")
        print(f"    acc_wrong (standardizace PRED splitem) = {leak['acc_wrong']:.4f}")
        print(f"    acc_right (scaler jen z trenovaci)     = {leak['acc_right']:.4f}")
        print("    -> unik (statistiky testu prosaknou do preprocessingu) nadhodnocuje")
        print("       skore; nikdy jej tise neopravujeme, ukazujeme obe cisla vedle sebe.")
    except NotImplementedError as exc:
        _faze_neni_hotova(exc)

    # --- Nevyvazenost trid: accuracy drzi vysoko, recall/Se se propada --------
    try:
        x_imb, y_imb = subsample_imbalance(
            x, y, cfg.data.imbalance_ratio, cfg.data.random_state
        )
        podil = float(np.mean(y_imb == 1))
        print(f"  Podvzorkovani na cca {cfg.data.imbalance_ratio:.0%} malignich: "
              f"{x_imb.shape[0]} vzorku, skutecny podil malignich {podil:.1%}.")
        x_tr, x_te, y_tr, y_te = _holdout_standardized(cfg, x_imb, y_imb)
        knn = KNNClassifier(cfg.knn.k, EuclideanDistance()).fit(x_tr, y_tr)
        cm = confusion_matrix(y_te, knn.predict(x_te))
        print(f"    accuracy  = {accuracy(cm):.3f}  (drzi se vysoko diky prevaze benignich)")
        print(
            f"    recall/Se = {recall_sensitivity(cm):.3f}  (propada se — maligni nadory unikaji)"
            )
        cesta = f"{GRAPHS_DIR}/kontingencni_tabulka_nevyvazenost.png"
        plot_confusion_matrix(cm, ["benigní (0)", "maligní (1)"], save_path=cesta)
        print(f"    Graf ulozen: {cesta}")
    except NotImplementedError as exc:
        _faze_neni_hotova(exc)


def faze_metriky(cfg, x: np.ndarray, y: np.ndarray, names: list[str]) -> None:
    """Faze 3 — metriky odvozene z kontingecni tabulky + ROC krivka."""
    _banner("Faze 3: Vyhodnoceni klasifikace z kontingecni tabulky")

    try:
        x_tr, x_te, y_tr, y_te = _holdout_standardized(cfg, x, y)
    except NotImplementedError as exc:
        _faze_neni_hotova(exc)
        return

    # --- Metriky z vlastniho kNN (studentsky kod) ----------------------------
    try:
        knn = KNNClassifier(cfg.knn.k, EuclideanDistance()).fit(x_tr, y_tr)
        cm = confusion_matrix(y_te, knn.predict(x_te))
        print(f"  accuracy    = {accuracy(cm):.3f}")
        print(f"  precision   = {precision(cm):.3f}")
        print(f"  recall/Se   = {recall_sensitivity(cm):.3f}   (sensitivita = recall = TPR)")
        print(f"  specificita = {specificity(cm):.3f}   (Sp = TNR)")
        print(f"  F1          = {f1_score(cm):.3f}")
    except NotImplementedError as exc:
        _faze_neni_hotova(exc)

    # --- ROC / AUC: referencni sklearn kNN (knihovna vs. vlastni) ------------
    # Vlastni KNNClassifier zamerne nema predict_proba (jedina studentska metoda
    # je predict), spojite skore pro ROC proto bere referencni sklearn model.
    try:
        from sklearn.metrics import roc_auc_score
        from sklearn.neighbors import KNeighborsClassifier

        ref = KNeighborsClassifier(n_neighbors=cfg.knn.k).fit(x_tr, y_tr)
        scores = ref.predict_proba(x_te)[:, 1]
        auc_val = float(roc_auc_score(y_te, scores))
        assert 0.0 <= auc_val <= 1.0, f"AUC mimo interval [0, 1]: {auc_val}"
        cesta = f"{GRAPHS_DIR}/roc_krivka.png"
        plot_roc_curve(y_te, scores, save_path=cesta)
        print("  ROC krivka (referencni sklearn kNN, skore = podil malignich sousedu)")
        print(f"  AUC = {auc_val:.3f}  (0.5 = nahodne hadani, 1.0 = dokonale oddeleni trid)")
        print(f"  krivka ulozena: {cesta}  (AUC je i v legende grafu)")
    except NotImplementedError as exc:
        _faze_neni_hotova(exc)


def faze_robustni_validace(cfg, x: np.ndarray, y: np.ndarray, names: list[str]) -> None:
    """Faze 4 — k-fold a bootstrap jako alternativy jednoho splitu + krivka preuceni."""
    _banner("Faze 4: Robustni validace — k-fold, bootstrap, krivka preuceni")

    # Pro krizovou validaci standardizujeme jednou nad celym X (kompromis kvuli
    # jednoduchosti demonstrace); spravny "per-fold" scaler by patril dovnitr
    # cross_validate, coz uz je nad ramec tohoto cviceni.
    metric_fns = {"accuracy": accuracy, "recall_Se": recall_sensitivity}
    validators = (
        ("kfold", KFold(cfg.validation.n_folds, random_state=cfg.data.random_state)),
        ("bootstrap", Bootstrap(cfg.validation.n_bootstrap,
                                random_state=cfg.data.random_state)),
    )
    try:
        params = standardize_fit(x)
        x_std = standardize_apply(x, params)
    except NotImplementedError as exc:
        _faze_neni_hotova(exc)
        x_std = None

    for nazev, validator in (validators if x_std is not None else ()):
        try:
            res = cross_validate(
                KNNClassifier(cfg.knn.k, EuclideanDistance()),
                validator, x_std, y, metric_fns,
            )
            print(f"  {nazev} ({res['n_splits']} splitu):")
            for m in metric_fns:
                print(f"    {m:9s}: mean={res[f'{m}_mean']:.3f}  std={res[f'{m}_std']:.3f}")
            cesta = f"{GRAPHS_DIR}/rozptyl_skore_{nazev}.png"
            plot_cv_scores(res["per_split_scores"], save_path=cesta)
            print(f"    Graf ulozen: {cesta}")
        except NotImplementedError as exc:
            _faze_neni_hotova(exc)

    # --- Krivka preuceni: train vs. test presnost pres cfg.knn.k_values ------
    try:
        x_tr, x_te, y_tr, y_te = _holdout_standardized(cfg, x, y)
        train_scores: list[float] = []
        test_scores: list[float] = []
        for k in cfg.knn.k_values:
            model = KNNClassifier(k, EuclideanDistance()).fit(x_tr, y_tr)
            train_scores.append(float(np.mean(model.predict(x_tr) == y_tr)))
            test_scores.append(float(np.mean(model.predict(x_te) == y_te)))
        cesta = f"{GRAPHS_DIR}/krivka_preuceni.png"
        plot_overfitting_curve(cfg.knn.k_values, train_scores, test_scores, save_path=cesta)
        for k, tr_s, te_s in zip(cfg.knn.k_values, train_scores, test_scores):
            print(f"    k={k:>3}: train={tr_s:.3f}  test={te_s:.3f}")
        print(f"  Krivka preuceni ulozena: {cesta}  (male k = preuceni, velke k = podteceni)")
    except NotImplementedError as exc:
        _faze_neni_hotova(exc)

    # --- Volitelne: rozhodovaci hranice ve 2D (prvni 2 standardizovane priznaky) ---
    try:
        x_tr, x_te, y_tr, y_te = _holdout_standardized(cfg, x, y)
        model = KNNClassifier(cfg.knn.k, EuclideanDistance()).fit(x_tr[:, :2], y_tr)
        cesta = f"{GRAPHS_DIR}/rozhodovaci_hranice_2d.png"
        plot_decision_boundary_2d(model, x_tr[:, :2], y_tr, save_path=cesta)
        print(f"  Rozhodovaci hranice (2D, {names[0]} vs. {names[1]}) ulozena: {cesta}")
    except NotImplementedError as exc:
        _faze_neni_hotova(exc)


def faze_cv5_link(cfg, x: np.ndarray, y: np.ndarray, names: list[str]) -> None:
    """Faze 5 — navaznost na cv5: pocet priznaku vs. presnost a cas predikce."""
    _banner("Faze 5: Navaznost na cv5 — pocet priznaku vs. presnost a cas")

    # Jednoduche self-contained univariatni poradi priznaku podle |Cohenova d|
    # (velikost ucinku mezi tridami). Zadny import z cv5 — pocitame inline.
    a = x[y == 0]
    b = x[y == 1]
    pooled_sd = np.sqrt((a.var(axis=0, ddof=1) + b.var(axis=0, ddof=1)) / 2.0)
    pooled_sd = np.where(pooled_sd == 0.0, 1.0, pooled_sd)
    cohen_d = np.abs(a.mean(axis=0) - b.mean(axis=0)) / pooled_sd
    poradi = np.argsort(cohen_d)[::-1]
    print("  Poradi priznaku podle |Cohenova d| (top 5): "
          + ", ".join(names[i] for i in poradi[:5]))

    try:
        x_tr, x_te, y_tr, y_te = _holdout_standardized(cfg, x, y)
        n_list: list[int] = []
        accuracies: list[float] = []
        times: list[float] = []
        for n in cfg.feature_selection.n_features_grid:
            sel = poradi[:n]
            model = KNNClassifier(cfg.knn.k, EuclideanDistance()).fit(x_tr[:, sel], y_tr)
            t0 = time.perf_counter()
            y_pred = model.predict(x_te[:, sel])
            dt = time.perf_counter() - t0
            n_list.append(int(n))
            accuracies.append(float(np.mean(y_pred == y_te)))
            times.append(dt)
            print(f"    n={n:>2}: presnost {accuracies[-1]:.3f}, "
                  f"cas predikce {dt * 1e3:.1f} ms")
        cesta = f"{GRAPHS_DIR}/kompromis_priznaky.png"
        plot_feature_tradeoff(n_list, accuracies, times, save_path=cesta)
        print(f"  Graf ulozen: {cesta}")
    except NotImplementedError as exc:
        _faze_neni_hotova(exc)


def main() -> None:
    """Spusti celou pipeline cviceni 06 s ochrannymi bloky u kazde faze."""
    _banner("CVICENI 06 — kNN & validace modelu — start")

    # --- Config guard -------------------------------------------------------
    try:
        cfg = load_config()
    except (ValueError, AssertionError, FileNotFoundError) as exc:
        print(f"[CHYBA KONFIGURACE] {exc}")
        sys.exit(1)

    # --- Data loading -------------------------------------------------------
    x, y, names = load_breast_cancer_data(cfg.data.random_state)
    print(f"  Data: x {x.shape}, malignich vzorku {int(y.sum())} / {len(y)} "
          f"(pozitivni trida = maligni).")

    faze_supervised_knn(cfg, x, y, names)
    faze_persistence(cfg, x, y)
    faze_data_handling(cfg, x, y, names)
    faze_metriky(cfg, x, y, names)
    faze_robustni_validace(cfg, x, y, names)
    faze_cv5_link(cfg, x, y, names)

    _banner("CVICENI 06 — konec")


if __name__ == "__main__":
    main()
