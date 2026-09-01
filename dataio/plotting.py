"""Vykreslovani vysledku cviceni 06 -- vse PREDVYPLNENE, bohate.

Grafy jsou v tomto cviceni explicitni priorita: kontingencni tabulka jako
anotovany heatmap, ROC krivka s AUC, krivka preuceni (train vs. test pres
``k``), rozptyl skore pres foldy, kompromis presnost/cas podle poctu
priznaku a rozhodovaci hranice kNN ve 2D.

Vsechny funkce pouzivaji neinteraktivni backend ``Agg``: figuru sestavi,
volitelne ulozi do ``save_path`` (vcetne vytvoreni nadrazeneho adresare) a
vzdy figuru zavrou. Funkce ``plt.show`` se nikdy nevola.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")  # neinteraktivni backend, vykreslujeme jen do souboru

import matplotlib.pyplot as plt  # noqa: E402  (musi az po matplotlib.use)
import numpy as np  # noqa: E402


def _save_and_close(fig: plt.Figure, save_path: str | None) -> None:
    """Pomocna funkce: ulozi figuru do ``save_path`` a zavre ji.

    Pokud je ``save_path`` ``None``, figura se pouze zavre. Nadrazeny
    adresar se v pripade potreby vytvori.
    """
    if save_path is not None:
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str] | None = None,
    save_path: str | None = None,
) -> None:
    """Vykresli 2x2 kontingencni tabulku jako anotovany heatmap.

    Parametry
    ---------
    cm:
        Kontingencni tabulka tvaru ``(2, 2)`` v orientaci
        ``[[TN, FP], [FN, TP]]`` (radek = skutecnost, sloupec = predikce;
        index 1 = pozitivni = maligni).
    class_names:
        Volitelne popisky trid (poradi ``[negativni, pozitivni]``).
        Vychozi ``["benigni (0)", "maligni (1)"]``.
    save_path:
        Cesta k vystupnimu PNG, nebo ``None`` (pak se figura jen zavre).

    Do kazde bunky se vypise absolutni pocet i podil z celku. Barevne
    skalovani zvyraznuje, kde je hmota tabulky -- u nevyvazenych dat je
    typicky drtiva vetsina v TN, coz je presne duvod, proc samotna
    presnost (accuracy) klame.
    """
    cm = np.asarray(cm, dtype=np.float64)
    if class_names is None:
        class_names = ["benigni (0)", "maligni (1)"]
    total = cm.sum()

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="pocet vzorku")

    cell_labels = [["TN", "FP"], ["FN", "TP"]]
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(2):
        for j in range(2):
            count = int(round(cm[i, j]))
            frac = cm[i, j] / total if total > 0 else 0.0
            ax.text(
                j,
                i,
                f"{cell_labels[i][j]}\n{count}\n({frac:.1%})",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=11,
            )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("predikce")
    ax.set_ylabel("skutecnost")
    ax.set_title("Kontingencni tabulka (pozitivni = maligni)")

    _save_and_close(fig, save_path)


def plot_roc_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    save_path: str | None = None,
) -> None:
    """Vykresli ROC krivku s plochou pod krivkou (AUC) v legende.

    Parametry
    ---------
    y_true:
        Skutecne binarni stitky delky ``n`` (1 = maligni = pozitivni).
    scores:
        Spojite skore pozitivni tridy delky ``n`` (napr.
        ``predict_proba(...)[:, 1]`` referencniho kNN). Cim vyssi, tim
        vetsi duvera v maligni.
    save_path:
        Cesta k vystupnimu PNG, nebo ``None``.

    ROC vznika prohanenim rozhodovaciho prahu pres vsechny hodnoty
    ``scores``: na ose x je ``1 - Sp`` (podil falesnych poplachu, FPR),
    na ose y ``Se = recall = TPR``. Diagonala odpovida nahodnemu hadani
    (AUC = 0.5), idealni klasifikator ma AUC = 1.
    """
    from sklearn.metrics import auc, roc_curve

    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=np.float64)
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#7f7f7f", lw=1, linestyle="--",
            label="nahodne hadani (AUC = 0.5)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("1 - specificita  (FPR = podil falesnych poplachu)")
    ax.set_ylabel("sensitivita  (Se = TPR = recall)")
    ax.set_title("ROC krivka: kompromis Se vs. Sp pres rozhodovaci prah")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_aspect("equal", adjustable="box")

    _save_and_close(fig, save_path)


def plot_overfitting_curve(
    k_values: Sequence[int],
    train_scores: Sequence[float],
    test_scores: Sequence[float],
    save_path: str | None = None,
) -> None:
    """Vykresli krivku preuceni: skore na train vs. test pres hodnoty ``k``.

    Parametry
    ---------
    k_values:
        Hodnoty hyperparametru ``k`` (pocet sousedu) na ose x.
    train_scores:
        Skore (napr. presnost) na trenovaci mnozine pro jednotliva ``k``.
    test_scores:
        Skore na testovaci mnozine pro jednotliva ``k``.
    save_path:
        Cesta k vystupnimu PNG, nebo ``None``.

    Male ``k`` (zvlaste ``k = 1``) je ucebnicovy priklad preuceni: model
    je na trenovacich datech temer dokonaly, ale na testu vyrazne horsi
    -- viditelne jako mezera mezi obema krivkami. S rostoucim ``k`` se
    hranice vyhlazuje, mezera se zaviraji, prilis velke ``k`` uz ale
    podteceni (obe krivky klesnou). Osa x je v logaritmickem meritku.
    """
    k_values = list(k_values)
    train_scores = np.asarray(train_scores, dtype=np.float64)
    test_scores = np.asarray(test_scores, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(k_values, train_scores, marker="o", color="#1f77b4",
            label="trenovaci skore")
    ax.plot(k_values, test_scores, marker="s", color="#d62728",
            label="testovaci skore")
    ax.fill_between(k_values, train_scores, test_scores, color="#d62728",
                    alpha=0.12, label="mezera (mi, preuceni)")
    ax.set_xscale("log")
    ax.set_xticks(k_values)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("k (pocet sousedu) -- log meritko")
    ax.set_ylabel("skore")
    ax.set_title("Krivka preuceni kNN: male k = preuceni, velke k = podteceni")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    _save_and_close(fig, save_path)


def plot_cv_scores(
    fold_scores: Mapping[str, Sequence[float]] | Sequence[float],
    save_path: str | None = None,
) -> None:
    """Vykresli rozptyl skore pres jednotlive foldy (boxplot + jednotlive body).

    Parametry
    ---------
    fold_scores:
        Bud slovnik ``{nazev_metriky: [skore_fold_1, skore_fold_2, ...]}``
        (napr. klic ``"per_split_scores"`` z ``cross_validate``), nebo
        proste jedna sekvence skore pres foldy.
    save_path:
        Cesta k vystupnimu PNG, nebo ``None``.

    Kazdy box shrnuje rozdeleni skore jedne metriky pres foldy, prekryte
    jednotlivymi body (strip). Smysl grafu: **jeden** ``train_test_split``
    da jedno cislo bez informace o jeho nejistote. k-fold / bootstrap
    daji cele rozdeleni -- sirka boxu ukazuje, jak moc by se odhad lisil
    pri jinem deleni dat.
    """
    if isinstance(fold_scores, Mapping):
        names = list(fold_scores.keys())
        data = [np.asarray(fold_scores[name], dtype=np.float64) for name in names]
    else:
        names = ["skore"]
        data = [np.asarray(fold_scores, dtype=np.float64)]

    fig, ax = plt.subplots(figsize=(1.6 * len(names) + 3.0, 4.8))
    ax.boxplot(data, tick_labels=names, showmeans=True, widths=0.5)

    rng = np.random.default_rng(0)
    for i, values in enumerate(data, start=1):
        jitter = rng.uniform(-0.08, 0.08, size=values.size)
        ax.scatter(np.full(values.size, i) + jitter, values, s=24,
                   color="#1f77b4", alpha=0.7, zorder=3)
        ax.text(i, values.mean(), f"  prum={values.mean():.3f}\n  sd={values.std(ddof=0):.3f}",
                va="center", ha="left", fontsize=8)

    ax.set_ylabel("skore na foldu")
    ax.set_title("Rozptyl skore pres foldy (proc jeden split nestaci)")
    ax.grid(True, axis="y", alpha=0.3)

    _save_and_close(fig, save_path)


def plot_feature_tradeoff(
    n_features: Sequence[int],
    accuracies: Sequence[float],
    times: Sequence[float],
    save_path: str | None = None,
) -> None:
    """Vykresli kompromis mezi presnosti a casem predikce podle poctu priznaku.

    Parametry
    ---------
    n_features:
        Pocet pouzitych priznaku na ose x (navazuje na vyber priznaku
        z cviceni 05).
    accuracies:
        Presnost kNN pro jednotlive pocty priznaku (leva osa y).
    times:
        Doba predikce v sekundach pro jednotlive pocty priznaku
        (prava osa y).
    save_path:
        Cesta k vystupnimu PNG, nebo ``None``.

    Graf ma dve osy y: modrou presnost a cervenou dobu predikce, obe
    proti poctu priznaku. Vice priznaku nemusi znamenat vyssi presnost
    (sum, "prokleti dimenzionality"), ale temer vzdy znamena pomalejsi
    predikci -- kNN pocita vzdalenosti ke vsem trenovacim bodum.
    """
    n_features = list(n_features)
    acc = np.asarray(accuracies, dtype=np.float64)
    tim = np.asarray(times, dtype=np.float64)

    fig, ax_acc = plt.subplots(figsize=(7, 4.5))
    color_acc = "#1f77b4"
    color_time = "#d62728"

    ax_acc.plot(n_features, acc, marker="o", color=color_acc, label="presnost")
    ax_acc.set_xlabel("pocet priznaku")
    ax_acc.set_ylabel("presnost", color=color_acc)
    ax_acc.tick_params(axis="y", labelcolor=color_acc)
    ax_acc.grid(True, alpha=0.3)

    ax_time = ax_acc.twinx()
    ax_time.plot(n_features, tim, marker="s", color=color_time,
                 label="doba predikce [s]")
    ax_time.set_ylabel("doba predikce [s]", color=color_time)
    ax_time.tick_params(axis="y", labelcolor=color_time)

    ax_acc.set_title("Kompromis: presnost vs. doba predikce podle poctu priznaku")

    _save_and_close(fig, save_path)


def plot_decision_boundary_2d(
    model: Any,
    X2d: np.ndarray,
    y: np.ndarray,
    save_path: str | None = None,
) -> None:
    """Vykresli rozhodovaci hranici klasifikatoru ve 2D.

    Parametry
    ---------
    model:
        Nafitovany klasifikator s metodou ``predict`` (vlastni
        ``KNNClassifier`` i ``sklearn`` model). Musi byt trenovany na
        dvourozmernych datech odpovidajicich ``X2d``.
    X2d:
        Priznakova matice tvaru ``(n_samples, 2)`` -- dva priznaky nebo
        2D projekce.
    y:
        Binarni stitky delky ``n_samples`` (1 = maligni).
    save_path:
        Cesta k vystupnimu PNG, nebo ``None``.

    Pres rovinu se polozi jemna mrizka, model kazdy jeji bod
    klasifikuje a vysledek se vykresli jako barevne pozadi; pres nej
    jsou trenovaci body. Male ``k`` da roztrhanou, "ostrovkovtou" hranici
    (preuceni), velke ``k`` hladkou hranici. Vizualni protejsek krivky
    preuceni.
    """
    X2d = np.asarray(X2d, dtype=np.float64)
    y = np.asarray(y)

    x_min, x_max = X2d[:, 0].min() - 0.5, X2d[:, 0].max() + 0.5
    y_min, y_max = X2d[:, 1].min() - 0.5, X2d[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = np.asarray(model.predict(grid)).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.contourf(xx, yy, zz, alpha=0.25, cmap="coolwarm", levels=[-0.5, 0.5, 1.5])
    ax.contour(xx, yy, zz, colors="#555555", linewidths=0.8, levels=[0.5])

    colors = {0: "#2ca02c", 1: "#d62728"}
    labels = {0: "benigni (0)", 1: "maligni (1)"}
    for cls in np.unique(y):
        mask = y == cls
        ax.scatter(X2d[mask, 0], X2d[mask, 1], s=18, alpha=0.8,
                   color=colors.get(int(cls)), edgecolor="white", linewidth=0.3,
                   label=labels.get(int(cls), f"trida {cls}"))

    ax.set_xlabel("priznak 1")
    ax.set_ylabel("priznak 2")
    ax.set_title("Rozhodovaci hranice kNN: k ridi hladkost hranice")
    ax.legend(loc="best")

    _save_and_close(fig, save_path)
