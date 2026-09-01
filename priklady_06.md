# Doplňkové příklady 06 — kNN a hodnocení klasifikace

Následující příklady jsou určeny k samostatnému řešení ručním výpočtem (bez
použití počítače). Slouží k procvičení postupů, které následně implementujete
v kódu (`KNNClassifier.predict`, `confusion_matrix` a metriky z ní odvozené,
`KFold.split`, `Bootstrap.split`). Hodnoty jsou zvoleny tak, aby mezivýsledky
zpravidla vycházely v celých číslech nebo jednoduchých zlomcích. Řešení nejsou
součástí repozitáře; ověřte je s vyučujícím nebo přepočtem v knihovnách
numpy / scikit-learn.

Značení a konvence (shodné s kódem):

- **Pozitivní třída = maligní nádor = `1`.** Negativní třída = benigní = `0`.
- Matici záměn (kontingenční tabulku) zapisujeme v orientaci
  `[[TN, FP], [FN, TP]]` — řádek = skutečnost, sloupec = predikce.
- `Se = recall = senzitivita = TPR` (míra záchytu pozitivních).
- `Sp = specificita = TNR` (míra záchytu negativních).
- Není-li uvedeno jinak, vzdálenost mezi body je **eukleidovská**.

---

## Příklad 1 — Kontingenční matice ručně

Klasifikátor nádorů vytvořil na testovací sadě 12 predikcí. Skutečné třídy
a predikce jsou:

```
y_true = [1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0]
y_pred = [1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1]
```

1. Pro každý z 12 vzorků určete, zda jde o **TP, FP, TN, nebo FN**
   (pozitivní třída = maligní = `1`).
2. Sečtěte je a sestavte kontingenční matici `2 × 2` v orientaci
   `[[TN, FP], [FN, TP]]`.
3. Ověřte, že součet všech čtyř buněk je 12 a že součet pozitivních
   skutečností (`TP + FN`) odpovídá počtu jedniček v `y_true`.

---

## Příklad 2 — Metriky z kontingenční matice

Jiný model vykázal na testovací sadě tuto matici záměn (orientace
`[[TN, FP], [FN, TP]]`):

```
        pred 0   pred 1
skut 0 |  19       1   |
skut 1 |   6       9   |
```

Spočítejte ručně (jako zlomky i jako desetinná čísla):

1. **accuracy** = `(TP + TN) / N`,
2. **precision** = `TP / (TP + FP)`,
3. **recall / Se** = `TP / (TP + FN)`,
4. **specificita / Sp** = `TN / (TN + FP)`,
5. **F1** = harmonický průměr precision a recall,
   tj. `2 · precision · recall / (precision + recall)`.

Závěrečná otázka: která z metrik je v této úloze (nepřehlédnout maligní nádor)
klíčová a proč?

---

## Příklad 3 — Proč je accuracy zavádějící

Je dána nevyvážená testovací sada 100 pacientů: **90 benigních** (`0`) a
**10 maligních** (`1`). Triviální (líný) klasifikátor bez fáze učení přiřazuje
všem pacientům většinovou třídu, tedy `0` (benigní).

1. Sestavte kontingenční matici tohoto klasifikátoru
   (orientace `[[TN, FP], [FN, TP]]`).
2. Spočítejte **accuracy** a **recall / Se**.
3. U **precision** a **F1** se podívejte na jmenovatel — co se stane a jak to
   ošetřit?
4. Vysvětlete, proč je accuracy `0,90` v tomto případě zavádějící a jaká
   informace o modelu je pro klinické použití podstatná.

---

## Příklad 4 — Rozdělení na k částí (k-fold)

Je dán dataset s 10 vzorky, indexy `0, 1, 2, …, 9`, a `n_folds = 5`.
Jednotlivé části (foldy) se tvoří **bez zamíchání** (`shuffle = False`), tj.
jako po sobě jdoucí bloky stejné velikosti.

1. Vypište pro každý z 5 foldů množinu **testovacích** indexů a množinu
   **trénovacích** indexů.
2. Kolikrát se každý vzorek objeví v testovací sadě (napříč všemi foldy)?
   A kolikrát v trénovací?
3. Ověřte, že testovací množiny jsou disjunktní a dohromady pokrývají
   všech 10 indexů.

---

## Příklad 5 — kNN ručně

Je dáno 6 popsaných 2D bodů (příznaky `x₁, x₂`) a jejich třídy
(`0` = benigní, `1` = maligní):

```
A = (1, 1)   y = 0
B = (2, 2)   y = 0
C = (3, 3)   y = 0
D = (6, 5)   y = 1
E = (5, 6)   y = 1
F = (1, 4)   y = 0
```

Úkolem je klasifikovat nový bod `q = (4, 5)` metodou kNN s `k = 3`.

1. Spočítejte **kvadrát** eukleidovské vzdálenosti `q` ke každému z 6 bodů
   (`(x₁ − q₁)² + (x₂ − q₂)²`). Proč pro seřazení sousedů postačuje kvadrát
   vzdálenosti a není nutné odmocňovat?
2. Vyberte **3 nejbližší** body.
3. Většinovým hlasováním jejich tříd určete predikci pro `q`.
   Jak se hlasování řeší při rovnosti hlasů (podle pravidla v docstringu
   `predict`)?

---

## Příklad 6 — Bootstrapové převzorkování a out-of-bag

Dataset má 5 vzorků s indexy `0, 1, 2, 3, 4`. Jedno bootstrapové losování vybralo
z indexů `0`–`4` pětkrát **s opakováním** tuto trénovací multimnožinu:

```
train_idx = [3, 0, 3, 1, 3]
```

1. Vypište trénovací indexy (včetně opakování) a určete množinu **out-of-bag**
   (OOB), tj. indexy, které se v `train_idx` nevyskytují.
2. Jaký podíl vzorků je v tomto losování OOB? Porovnejte jej s asymptotickou
   hodnotou `(1 − 1/n)ⁿ → e⁻¹ ≈ 0,368`.
3. Pro `n = 5` spočítejte **přesně** pravděpodobnost, že daný vzorek zůstane OOB,
   tj. `(1 − 1/5)⁵`. O kolik se liší od `e⁻¹`?
4. Kolik různých vzorků trénovací množina obsahuje? Vysvětlete, proč
   bootstrapová trénovací množina nese méně informace, než odpovídá jejímu
   počtu prvků.

---

## Příklad 7 — k-fold se zamícháním

Uvažujte stejný dataset jako v Příkladu 4 (10 vzorků, indexy `0`–`9`,
`n_folds = 5`), tentokrát však se zamícháním (`shuffle = True`). Generátor
pseudonáhodných čísel vytvořil toto pořadí indexů:

```
perm = [7, 2, 9, 0, 4, 6, 1, 8, 3, 5]
```

1. Rozdělte zamíchané pořadí na 5 po sobě jdoucích bloků po dvou indexech.
   Vypište pro každý fold **testovací** a **trénovací** indexy (v původním
   číslování vzorků).
2. Ověřte, že testovací množiny jsou disjunktní a dohromady pokrývají všech
   10 indexů.
3. V čem se výsledek liší od Příkladu 4 (bez zamíchání) a proč se zamíchání
   před rozdělením obvykle doporučuje?

---

## Příklad 8 — Vliv volby `k` na predikci

Použijte body `A`–`F` a dotaz `q = (4, 5)` z Příkladu 5; kvadráty vzdáleností
již máte spočtené.

1. Určete predikci pro `q` metodou kNN s `k = 1`, `k = 3` a `k = 5`.
2. Při kterých hodnotách `k` se predikce shodují a při které se liší? Který
   trénovací bod způsobí změnu?
3. Ke kterému okraji kompromisu zkreslení–rozptyl (bias–variance) se blíží
   `k = 1` a ke kterému `k = 6` (počet všech bodů)? Která hodnota `k` by na
   takto malé sadě dávala smysl?

---

## Příklad 9 — Vliv volby metriky vzdálenosti

Uvažujte body a dotaz `q = (4, 5)` z Příkladu 5, nyní však s **manhattanskou**
vzdáleností `d = |x₁ − q₁| + |x₂ − q₂|`.

1. Spočítejte manhattanskou vzdálenost `q` ke každému ze 6 bodů.
2. Vyberte 3 nejbližší body a určete predikci pro `k = 3`.
3. Porovnejte pořadí sousedů s eukleidovským pořadím z Příkladu 5. Změnila se
   trojice nejbližších sousedů? Změnil se nejbližší soused? Změnila se predikce?
4. Jaký důsledek z toho plyne pro roli injektované metriky `Distance` ve třídě
   `KNNClassifier`?

---

## Příklad 10 — Únik dat při standardizaci

Uvažujte jediný příznak a jeho hodnoty rozdělené na trénovací a testovací část:

```
train = [2, 4, 4, 6]        test = [10, 16]
```

1. Spočítejte průměr a rozptyl (populační, dělíte `N`) **pouze z trénovací
   části** a proveďte z-transformaci obou testovacích hodnot těmito parametry
   (**správný postup**).
2. Spočítejte průměr **ze všech šesti hodnot** dohromady a určete, o kolik se
   posune vůči průměru z bodu 1.
3. Porovnejte vycentrovanou hodnotu `x − průměr` testovacího bodu `10` v obou
   postupech. Ve kterém případě se testovací bod jeví bližší středu trénovacích
   dat?
4. Vysvětlete, proč postup využívající průměr ze všech dat „nadhodnocuje"
   kvalitu modelu, přestože se testovací **popisky** nikdy nepoužijí.

---

## Příklad 11 — ROC křivka a AUC

Referenční kNN s `k = 4` vrátil pro 8 testovacích vzorků **skóre** rovné podílu
pozitivních (maligních) mezi 4 nejbližšími sousedy. Skutečné třídy jsou známé:

```
skóre  = [1,00; 0,75; 0,75; 0,50; 0,50; 0,25; 0,25; 0,00]
y_true = [   1;    1;    0;    1;    0;    0;    1;    0  ]
```

(Pozitivních je `P = 4`, negativních `N = 4`.)

1. Pro každý práh `τ ∈ {1,00; 0,75; 0,50; 0,25; 0,00}` označte vzorek za
   pozitivní právě tehdy, když `skóre ≥ τ`. Sestavte matici záměn a spočítejte
   `Se` a `1 − Sp` (tj. podíl falešně pozitivních, FPR).
2. Vyneste body `(1 − Sp, Se)` do grafu ROC (osa x = FPR, osa y = TPR) a spojte
   je úsečkami; přidejte krajní bod `(0, 0)`.
3. Spočítejte **AUC** lichoběžníkovým pravidlem.
4. Jak by vypadala ROC křivka a jaká by byla AUC pro klasifikátor hádající
   náhodně? A pro klasifikátor s dokonalým oddělením tříd?

---

## Příklad 12 — Agregace výsledků křížové validace

Pětinásobná křížová validace klasifikátoru vrátila tyto hodnoty senzitivity
po jednotlivých foldech:

```
Se = [0,80; 0,60; 1,00; 0,80; 0,80]
```

1. Spočítejte **průměr** a **výběrovou směrodatnou odchylku** (dělíte `n − 1`)
   těchto pěti hodnot.
2. Zapište výsledek ve tvaru `průměr ± směrodatná odchylka`.
3. Proč funkce `cross_validate` vrací průměr **i** směrodatnou odchylku, a ne
   pouze jediné číslo z jednoho rozdělení dat?

---

## Příklad 13 — Volba rozhodovacího prahu podle ceny chyby

Vyjděte ze skóre a skutečných tříd z Příkladu 11. Uvažujte dva rozhodovací
prahy: `τ = 0,50` a `τ = 0,25`.

1. Pro každý práh sestavte matici záměn `[[TN, FP], [FN, TP]]` a spočítejte
   `Se`, `Sp` a `precision`.
2. Náklady na chybnou klasifikaci jsou: **falešně negativní** (maligní nádor
   klasifikovaný jako benigní) `= 10`, **falešně pozitivní** `= 1`. Spočítejte
   celkový náklad pro každý práh.
3. Který práh je podle tohoto kritéria vhodnější? Jak se závěr změní, jsou-li
   oba náklady stejné (`FN = 1`, `FP = 1`)?

---

> **Řešení nejsou součástí repozitáře.** Ověřte výsledky s vyučujícím nebo
> přepočtem v numpy / scikit-learn (`sklearn.metrics.confusion_matrix`,
> `accuracy_score`, `precision_score`, `recall_score`, `f1_score`,
> `roc_curve`, `roc_auc_score`; `sklearn.model_selection.KFold`;
> `sklearn.neighbors.KNeighborsClassifier`). Shoda ručního výpočtu s pozdější
> implementací je zároveň kontrolou správnosti obojího.
