# Cvičení 6: kNN a validace modelu — vyhodnocení klasifikace

Šesté praktické cvičení předmětu **Umělá inteligence v medicíně** otevírá blok
**učení s učitelem**, který navazuje na pět cvičení bez učitele (Cvičení 01-05).
Vzorovou datovou sadou zůstává **Breast Cancer Wisconsin** — 569 nádorů
popsaných 30 číselnými příznaky a binární diagnózou (**1 = maligní,
0 = benigní**).

Klasifikátor **k nejbližších sousedů (kNN)** je zvolen záměrně jako nejjednodušší
možná varianta: ve fázi učení pouze uchová trénovací data, takže těžiště cvičení
může spočívat na metodické stránce modelování a vyhodnocení. Cvičení pokrývá
rozdělení dat a důvody, proč se testovací množina drží stranou; **únik dat**
(data leakage); **nevyváženost tříd** a důvody, proč je přesnost (accuracy)
v takovém případě zavádějící; **kontingenční tabulku** jako základ, z něhož se
odvozují všechny metriky; **robustní validaci** (k-fold, bootstrap) jako
alternativu jediného rozdělení; a **přeučení při nízkém `k`**. Interpretace
výsledků je předmětem diskuse na cvičení; repozitář poskytuje kostru kódu,
číselné výstupy a grafy.

Cvičení opět využívá třídu `Distance` z Cvičení 01 (v Cvičení 05 nebyla
potřeba): kNN volá `distance.calculate` při řazení sousedů. Metoda `predict`
na nových datech má zde poprvé plný význam — kNN je **induktivní** (na rozdíl
od transduktivních metod z Cvičení 04).

---

## Obsah

1. [Cíle cvičení](#cíle-cvičení)
2. [Struktura repozitáře](#struktura-repozitáře)
3. [Instalace a spuštění](#instalace-a-spuštění)
4. [Teoretický základ](#teoretický-základ)
5. [Konfigurace projektu](#konfigurace-projektu)
6. [Pokyny k vypracování](#pokyny-k-vypracování)
7. [Lokální testování](#lokální-testování)
8. [Doplňkové (papírové) příklady](#doplňkové-papírové-příklady)
9. [Odevzdání](#odevzdání)

---

## Cíle cvičení

Po dokončení tohoto cvičení student:

1. **Implementuje klasifikátor kNN od základů** — pro každý dotazovaný vzorek
   spočítá vzdálenost ke všem trénovacím bodům přes injektovanou metriku
   `Distance`, vybere `k` nejbližších a vrátí většinový hlas. Chápe, že trénink
   kNN spočívá pouze v uložení dat a že celý model je tvořen trénovací množinou
   — na rozdíl od několika polí natrénované PCA z Cvičení 05. Natrénovaný model
   umí uložit do souboru `.npz` a zpět načíst (metody `save` a `load`); u kNN to
   znamená uložit celou trénovací množinu, u PCA stačila hrstka polí.
2. **Chápe přechod k učení s učitelem** — nyní máme popisky `y` a učíme se je
   předpovídat. `predict` na nová data je smysluplný (kNN je induktivní), na
   rozdíl od transduktivních metod z Cvičení 04.
3. **Zachází s daty správně** — implementuje z-skórovou standardizaci ve dvou
   fázích (`standardize_fit` na trénovacích datech, `standardize_apply` na
   train i test), ví, proč se testovací množina drží stranou, umí pojmenovat
   **únik dat** (standardizace před rozdělením vs. po něm, ukázáno jako dvě
   čísla vedle sebe) a rozumí tomu, proč **nevyváženost tříd** činí přesnost
   (accuracy) zavádějící metrikou.
4. **Odvozuje vyhodnocení z kontingenční tabulky** — sestaví matici záměn a
   z ní odvodí přesnost, preciznost, **senzitivitu (Se = recall = TPR)**,
   **specificitu (Sp = TNR)** a F1. Ví, která metrika kdy rozhoduje — v
   klinickém kontextu senzitivita (nepropásnout maligní nádor) obvykle
   předčí preciznost. Rozumí ROC/AUC jako pohledu přes práh rozhodování.
5. **Implementuje robustní validaci jako návrhový vzor Strategy** — `Validator`
   (ABC) se třemi zaměnnými potomky `HoldOut` / `KFold` / `Bootstrap`, které
   všechny produkují dvojice `(train_idx, test_idx)`. `train_test_split` je
   jen nejjednodušší převzorkovací strategie; k-fold a bootstrap jsou její
   rovnocenné alternativy.
6. **Pozoruje přeučení na konkrétním případu** — `k = 1` je učebnicový příklad
   přeučení (vysoká přesnost na tréninku, nízká na testu); křivka skóre v
   závislosti na `k` propojuje toto pozorování zpět s bodem 3.
7. **Propojí Cvičení 05 a 06** — znovu použije výběr příznaků a ukáže, jak
   volba trénovacích příznaků mění přesnost kNN *i* dobu predikce.
8. **Pracuje s typovanou konfigurací** — čte hyperparametry z `config.yaml`
   přes dataclassy (`cfg.knn.k` místo `cfg["knn"]["k"]`), stejný vzor jako
   v Cvičení 03–05.

---

## Struktura repozitáře

```
cviceni-06-template/
├── cviceni_06.py            # Hlavní pipeline — spusťte pro průběžné ověření (PŘEDVYPLNĚNO)
├── config.yaml              # Konfigurace experimentu (YAML)
├── priklady_06.md           # Papírové (teoretické) příklady — BEZ řešení v repozitáři
├── requirements.txt         # Python závislosti (zamčené verze)
├── .gitignore
├── src/
│   ├── __init__.py          # Re-exporty balíčku (neupravujte)
│   ├── distance.py          # Distance (ABC) + Euclidean/Manhattan/Cosine — ÚKOL: zkopírujte z Cvičení 01
│   ├── knn.py               # KNNClassifier — ÚKOL: predict + save + load; __init__ a fit() předvyplněny
│   ├── metrics.py           # confusion_matrix + metriky z ní ODVOZENÉ — ÚKOL: 6 funkcí
│   └── validation.py        # Validator (ABC) + HoldOut (vzor) + KFold/Bootstrap — ÚKOL: 2 metody split(); cross_validate předvyplněno
├── dataio/
│   ├── __init__.py          # Re-exporty balíčku (neupravujte)
│   ├── loader.py            # load_breast_cancer_data() — načtení dat (předvyplněno)
│   ├── preprocessing.py     # ÚKOL: standardize_fit/apply + subsample_imbalance; demonstrate_leakage předvyplněno
│   ├── config_manager.py    # Dataclassy + load_config + validate_config (předvyplněno)
│   └── plotting.py          # Bohaté vizualizace: matice záměn, ROC, křivka přeučení, rozptyl CV, kompromis příznaků, rozhodovací hranice (předvyplněno)
├── models/                  # Sem pipeline uloží natrénovaný kNN model (models/knn_model.npz); negitováno
│   └── .gitkeep
├── graphs/                  # Výstupní složka pro grafy (generuje se automaticky)
│   └── .gitkeep
└── test_cviceni_06.py       # Automatické testy (pytest) — obsahuje DummyDistance
```

> **Poznámka k souborům `__init__.py`:** Každá složka s Python kódem (`src/`,
> `dataio/`) obsahuje `__init__.py`, který ji označuje jako balíček a definuje
> veřejné API. Díky tomu lze psát `from src import KNNClassifier` místo
> `from src.knn import KNNClassifier`. **Tyto soubory neupravujte.**

> **Balíček `dataio/` je z větší části předvyplněn** — načítání dat,
> konfigurace i vykreslování. Výjimkou je `dataio/preprocessing.py`, kde tři
> funkce (`standardize_fit`, `standardize_apply`, `subsample_imbalance`) tvoří
> samostatný úkol (Blok IV); `demonstrate_leakage` v témže souboru zůstává
> předvyplněná. Jinde v `dataio/` se žádný `NotImplementedError` nevyskytuje.

> **Data se nenačítají ze souboru** — `load_breast_cancer_data()` je bere přímo
> ze `scikit-learn`, repozitář proto žádnou složku `data/` neobsahuje. Jediný
> artefakt, který běh vytvoří vedle grafů, je natrénovaný model
> `models/knn_model.npz` (metody `KNNClassifier.save` / `load`). Složky `graphs/`
> i `models/` zůstávají ve verzování prázdné (přes `.gitkeep`), jejich obsah je
> v `.gitignore`.

> **Opětovné zařazení `src/distance.py`.** Na rozdíl od Cvičení 05 se zde znovu
> kopíruje třída `Distance` z Cvičení 01 — kNN hledá sousedy voláním
> `calculate()`. Bez funkční implementace se hlavní část cvičení nespustí.
> V testech je proto opět `DummyDistance`.

---

## Instalace a spuštění

### 1. Vytvoření virtuálního prostředí

```bash
python -m venv .venv
```

Aktivace (Windows):
```bash
.venv\Scripts\activate
```

Aktivace (Linux / macOS):
```bash
source .venv/bin/activate
```

### 2. Instalace závislostí

```bash
pip install -r requirements.txt
```

Cvičení používá `numpy`, `scipy`, `scikit-learn` (dataset, referenční kNN,
pomocné výpočty ROC/CV), `matplotlib` (grafy), `pyyaml` (konfigurace) a
`pytest` (testy). Verze jsou v `requirements.txt` zamčené.

### 3. Spuštění

```bash
python cviceni_06.py
```

Pipeline načte konfiguraci a data a projde bloky cvičení: (1) přechod k učení
s učitelem a kNN **+ uložení a znovunačtení natrénovaného modelu** (`save` /
`load` do `.npz`), (2) správné zacházení s daty (únik dat, nevyváženost),
(3) metriky z kontingenční tabulky a **ROC + AUC**, (4) robustní validace,
křivka přeučení a propojení s Cvičením 05. Každá fáze je obalena
samostatným blokem `try / except NotImplementedError`, takže **nedokončený
úkol jednu fázi přeskočí, ale nezablokuje ostatní** — cílem je poskytnout co
nejvíce zpětné vazby. Nedokončená fáze vypíše hlášku začínající
`[NENI HOTOVO] Úkol: …` a pipeline pokračuje. Ve stavu kostry tedy
`python cviceni_06.py` **nikdy neskončí nezpracovaným tracebackem**.

> **Jediná výjimka — načtení konfigurace.** `load_config()` volá
> `validate_config()`; kdyby v `config.yaml` byla nesmyslná hodnota, pipeline se
> korektně ukončí hláškou `[CHYBA KONFIGURACE]` hned na začátku. Obě funkce jsou
> předvyplněné, takže při nezměněné konfiguraci tento stav nenastane.

> **Jen jeden druh `NotImplementedError`.** V Cvičení 04 existovaly dva —
> studentský úkol (`Úkol: …`) a trvalé architektonické omezení (`predict()` u
> transduktivních metod). **V tomto cvičení je druh jediný:** každá zpráva
> začíná `Úkol:` a každou je potřeba doplnit. kNN je induktivní, takže
> `predict()` na nová data je dobře definovaný — neexistuje tedy důvod
> k trvalému omezení.

Jednotlivé metody lze mezitím ověřovat přes `pytest`, viz
[Lokální testování](#lokální-testování).

---

## Teoretický základ

Cvičení má dvě vrstvy: **metodickou** (metodika modelování a vyhodnocení) a
**algoritmickou** (funkce kNN). Metodická vrstva je stěžejní.

### 1. Učení s učitelem a klasifikátor kNN

Prvních pět cvičení pracovalo **bez učitele** — hledalo strukturu v datech bez
znalosti cílových hodnot. Cvičení 05 poprvé využilo cílovou proměnnou `y`
(filtrační test srovnával diagnostické skupiny), ale ještě neklasifikovalo.
Nyní se `y` učíme **předpovídat**.

**k nejbližších sousedů (kNN)** je *líný* (lazy), *instanční* klasifikátor:

- **`fit(X, y)`** pouze uloží trénovací matici a popisky. Žádný model se
  nepočítá. Naučeným modelem kNN je celá trénovací množina.
- **`predict(X)`** pro každý dotazovaný řádek $\mathbf{x}$:
  1. spočítá vzdálenost $d(\mathbf{x}, \mathbf{x}_i)$ ke **každému** trénovacímu
     bodu přes injektovanou metriku (`self.distance.calculate`),
  2. vybere `k` bodů s nejmenší vzdáleností,
  3. vrátí **většinový hlas** jejich popisků.

$$\hat{y}(\mathbf{x}) = \operatorname*{arg\,max}_{c \in \{0, 1\}}
\sum_{i \in N_k(\mathbf{x})} \mathbb{1}[\,y_i = c\,]$$

kde $N_k(\mathbf{x})$ je množina indexů `k` nejbližších sousedů. **Shody hlasů**
(ties) je nutné rozhodnout deterministicky — např. ve prospěch třídy s nižším
číslem, nebo podle nejbližšího souseda. Docstring `predict` uvádí zvolené pravidlo.

> **Injektovaná metrika (Dependency Injection).** `KNNClassifier.__init__`
> dostane instanci `Distance` a uloží ji. Stejný vzor jako v Cvičení 03/04:
> algoritmus nezná konkrétní vzorec vzdálenosti, jen rozhraní `calculate`.
> Výměna eukleidovské metriky za manhattanskou je pak záměna jednoho argumentu,
> ne úprava kódu kNN.

> **Induktivní vs. transduktivní — návrat `predict`.**
>
> | | DBSCAN / spektrální (Cv. 04) | PCA (Cv. 05) | **kNN (Cv. 06)** |
> |:---|:---|:---|:---|
> | Charakter | transduktivní | induktivní | **induktivní** |
> | `predict` na nová data | ne (nedefinováno) | ano (`transform`) | **ano** |
> | Co se naučí | pouze rozdělení trénovacích dat | průměr + komponenty | **celou trénovací množinu** |
>
> kNN je induktivní: `predict` na libovolná nová data je dobře definován.
> Proto zde na rozdíl od Cvičení 04 neexistuje trvalý `NotImplementedError`.

**Vliv `k`.** Malé `k` dává členitou, lokálně citlivou rozhodovací hranici
(`k = 1` kopíruje každý trénovací bod včetně šumu). Velké `k` hranici vyhlazuje
a v limitě `k = n` predikuje vždy majoritní třídu. Toto je **kompromis
zkreslení a rozptylu** (bias–variance) a přímo souvisí s přeučením (odd. 5).

### 2. Rozdělení dat a únik informace

#### 2.1 Proč se testovací množina drží stranou

Model se hodnotí podle toho, jak zobecňuje na **dosud neviděná** data. Kdybychom
měřili přesnost na týchž datech, která model použil při tréninku, `k = 1` kNN by
dal 100 % — a přitom by neposkytl žádnou informaci o chování na nových
pacientech. Data proto dělíme:

$$X = X_{\text{train}} \;\dot\cup\; X_{\text{test}}, \qquad
\text{model vidí } X_{\text{test}} \text{ až při závěrečném vyhodnocení.}$$

Testovací množina se použije jedinkrát, a to při závěrečném vyhodnocení.

#### 2.2 Únik dat (data leakage)

**Únik dat** nastane, když se do trénování dostane jakákoli informace z testovací
množiny. Nejčastější a obtížně odhalitelná forma je **předzpracování před
rozdělením**:

| Pořadí | Co se stane | Důsledek |
|:---|:---|:---|
| **Špatně** | standardizace **celého `X`** → rozdělení → trénink kNN | průměr a směrodatná odchylka byly spočítány i z testovacích řádků — model „viděl" statistiky testu |
| **Správně** | rozdělení → `standardize_fit` **jen na train** → `standardize_apply` na train i test → trénink kNN | testovací řádky se na parametrech škálování nepodílely |

Pipeline spustí **obě** pořadí a vytiskne obě přesnosti vedle sebe. Špatné
pořadí obvykle skóre mírně **nadhodnotí**, což je podstata této ukázky. Únik se
v pipeline neopravuje potichu; je prezentován jako dvojice čísel.

> Orchestrace `demonstrate_leakage` je **předvyplněná** — únik dat se
> demonstruje spuštěním, ne programováním. Standardizační funkce, které volá
> (`standardize_fit`, `standardize_apply`), si však implementujete sami
> (**Blok IV** v Pokynech k vypracování); bez nich demonstrace nedoběhne.

#### 2.3 Nevyváženost tříd

Datová sada Breast Cancer je mírně nevyvážená (~63 % benigních, ~37 % maligních).
Pro názornou demonstraci pipeline **podvzorkuje maligní třídu na ~10 %**
(`subsample_imbalance` — rovněž studentský úkol, viz Blok IV; řízeno
`config.yaml`). Na takto nevyvážených datech **přesnost klame**:

Triviální klasifikátor „vždy benigní" má na rozdělení 90/10 přesnost **90 %** —
a přitom **nezachytí ani jeden maligní nádor** (senzitivita 0 %). Přesnost zde
nemá vypovídací hodnotu; rozhoduje senzitivita a specificita (odd. 3). Pipeline
ukáže, že kNN na nevyvážených datech udrží vysokou přesnost, zatímco jeho
senzitivita výrazně klesá.

### 3. Vyhodnocení z kontingenční tabulky

#### 3.1 Matice záměn (confusion matrix)

Kořenový objekt veškerého vyhodnocení. Pro binární úlohu s **pozitivní třídou =
maligní (1)** je to tabulka $2 \times 2$:

$$\text{cm} =
\begin{pmatrix} \text{TN} & \text{FP} \\ \text{FN} & \text{TP} \end{pmatrix}
= \begin{pmatrix}
\text{správně benigní} & \text{benigní označený jako maligní} \\
\text{maligní označený jako benigní} & \text{správně maligní}
\end{pmatrix}$$

Řádek = **skutečnost**, sloupec = **predikce**. Tato orientace se shoduje s
`sklearn.metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])`. **Všechny
následující metriky berou tuto matici, ne surové vektory `y_true` / `y_pred`** —
jde o záměr: princip „všechny metriky se odvozují z kontingenční tabulky" je
promítnut přímo do struktury kódu.

#### 3.2 Metriky odvozené z matice

| Metrika | Vzorec | Otázka, na kterou odpovídá |
|:---|:---|:---|
| **Přesnost** (accuracy) | $\dfrac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$ | Jaký podíl všech predikcí je správně? |
| **Preciznost** (precision) | $\dfrac{\text{TP}}{\text{TP} + \text{FP}}$ | Když model řekne „maligní", jak často má pravdu? |
| **Senzitivita** (Se = recall = TPR) | $\dfrac{\text{TP}}{\text{TP} + \text{FN}}$ | Jaký podíl skutečně maligních model zachytí? |
| **Specificita** (Sp = TNR) | $\dfrac{\text{TN}}{\text{TN} + \text{FP}}$ | Jaký podíl skutečně benigních model správně propustí? |
| **F1** | $2 \cdot \dfrac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}$ | Harmonický průměr preciznosti a senzitivity. |

> **Terminologie — strojové učení vs. medicína.** Táž veličina má dvě jména
> podle oboru. Explicitní uvedení těchto ekvivalencí je součástí učiva:
> $$\text{Se} = \text{recall} = \text{senzitivita} = \text{TPR (true positive rate)}$$
> $$\text{Sp} = \text{specificita} = \text{TNR (true negative rate)}$$

> **Ošetřete dělení nulou.** Když je jmenovatel nula (např. model nepredikoval
> žádnou pozitivní třídu → `TP + FP = 0`), vraťte `0.0`, ne `NaN`. Každý
> docstring tento hraniční případ zmiňuje.

#### 3.3 Která metrika kdy

V medicíně nejsou chyby symetrické. **Falešně negativní** (FN — maligní nádor
označený jako benigní) znamená propásnutou diagnózu; **falešně pozitivní** (FP)
znamená zbytečné vyšetření navíc. Proto **senzitivita obvykle předčí
preciznost** — důsledky propásnuté diagnózy jsou závažnější než důsledky
falešně pozitivního nálezu. Volba prahu rozhodování představuje posun v rámci
tohoto kompromisu.

#### 3.4 ROC a AUC

kNN může místo tvrdého popisku vrátit **skóre** — např. podíl pozitivních mezi
`k` sousedy. Posouváním prahu $\tau \in [0, 1]$ nad tímto skóre dostáváme pro
každý práh jednu dvojici (1 − Sp, Se), tj. jeden bod **ROC křivky** (Receiver
Operating Characteristic). **AUC** (plocha pod křivkou) shrnuje kvalitu přes
všechny prahy: 1,0 = dokonalé oddělení, 0,5 = náhodné hádání.

> V tomto cvičení běží ROC na **referenčním `sklearn.KNeighborsClassifier`**
> (rámec „knihovna vs. vlastní implementace"). Jedinou studentskou metodou, která
> pracuje se sousedy, zůstává `predict`; logika hledání sousedů se kvůli
> `predict_proba` neduplikuje.

### 4. Robustní validace jako návrhový vzor Strategy

Jediné rozdělení `train_test_split` dá **jedno** číslo přesnosti — a to číslo je
náhodná veličina závislá na tom, které řádky byly zařazeny do testovací množiny.
Robustnější je data rozdělit **vícekrát** a posoudit rozptyl odhadu.

Klíčové pozorování (a návrhové východisko tohoto cvičení): **`train_test_split`
je nejjednodušší převzorkovací strategie; k-fold a bootstrap jsou její
rovnocenné, zaměnné alternativy.** Všechny tři produkují dvojice
`(train_idx, test_idx)`. To je návrhový vzor **Strategy**: abstraktní třída
`Validator` s metodou `split`, tři zaměnné potomky, a funkce `cross_validate`,
která nezávisí na tom, kterou strategii obdrží.

| Strategie | `split` vydá | Princip |
|:---|:---|:---|
| **`HoldOut`** (předvyplněno jako vzor) | právě 1 dvojici | jedno zamíchané rozdělení podle `test_size` |
| **`KFold`** (ÚKOL) | `n_folds` dvojic | indexy se rozdělí do `n_folds` přihrádek; každá je jednou testem, zbytek tréninkem — **každý vzorek je testován právě jednou** |
| **`Bootstrap`** (ÚKOL) | `n_bootstrap` dvojic | trénink = `n` indexů losovaných **s opakováním**; test = **out-of-bag** (nevylosované) indexy, průměrně ~36,8 % dat |

> **Proč ~36,8 %.** Pravděpodobnost, že konkrétní vzorek *není* vybrán v jednom
> ze `n` losování s opakováním, je $(1 - 1/n)^n \to e^{-1} \approx 0{,}368$.

`HoldOut` je předvyplněný **vzorový příklad** (paralela k `RandomUniformInit`
z Cvičení 03) — konkrétně ilustruje rozhraní, které mají `KFold` a `Bootstrap`
naplnit.

**`cross_validate(model, validator, X, y, metric_fns)`** (předvyplněno) pro
každé rozdělení z `validator.split(X, y)` znovu natrénuje `model`, predikuje
test, sestaví matici záměn, spočítá každou metriku z `metric_fns` a agreguje
průměr a směrodatnou odchylku napříč rozděleními (plus per-fold skóre pro graf).
Tím propojí studentův kNN + validátor + metriky do jednoho výpočtu.

### 5. Přeučení při nízkém `k`

`k = 1` je **učebnicový příklad přeučení**: na trénovacích datech je přesnost
100 % (nejbližší soused každého trénovacího bodu je on sám), na testovací
množině však skóre výrazně klesá — model se přizpůsobil šumu. S rostoucím `k` se
rozhodovací hranice vyhlazuje: trénovací přesnost klesá, testovací nejprve
stoupá (méně přeučení) a pak zase klesá (přílišné vyhlazení, *underfitting*).
Optimální `k` leží uvnitř tohoto kompromisu.

Pipeline projde `cfg.knn.k_values`, pro každé `k` změří **trénovací i testovací**
přesnost a vykreslí je do jednoho grafu (`plot_overfitting_curve`). Odstup obou
křivek při nízkém `k` představuje graficky patrné přeučení. Tím se výklad vrací
k odd. 2 — korektní oddělení testovacích dat je jedinou cestou, jak přeučení
vůbec zjistit.

### 6. Propojení s Cvičením 05 — příznaky vs. čas

Cvičení opět využívá výběr příznaků: pipeline projde
`cfg.feature_selection.n_features_grid`, pro každý počet příznaků (seřazených
jednoduchým univariátním kritériem) natrénuje kNN a změří **přesnost** i **dobu
predikce**. Méně příznaků zpravidla znamená rychlejší predikci (kNN počítá
vzdálenosti ve všech dimenzích) za cenu možného poklesu přesnosti. Graf
`plot_feature_tradeoff` tento kompromis znázorňuje a propojuje obě cvičení.

### 7. Rozdělení implementace mezi studenta a knihovny

| Komponenta | Kdo počítá | Poznámka |
|:---|:---|:---|
| `predict` — vzdálenosti ke všem trénovacím bodům, výběr `k` nejbližších, většinový hlas | **student** (`src/knn.py`) | jádro kNN |
| `save` / `load` natrénovaného modelu (`.npz`) | **student** (`src/knn.py`) | téma „model = uložené parametry"; u kNN je jím **celá trénovací množina** |
| `confusion_matrix` + `accuracy` / `precision` / `recall_sensitivity` / `specificity` / `f1_score` | **student** (`src/metrics.py`) | vše se odvozuje z kontingenční tabulky |
| `KFold.split`, `Bootstrap.split` | **student** (`src/validation.py`) | převzorkovací strategie (návrhový vzor Strategy) |
| `standardize_fit` / `standardize_apply`, `subsample_imbalance` | **student** (`dataio/preprocessing.py`) | obrana proti úniku dat, demonstrace nevyváženosti |
| `EuclideanDistance` / `ManhattanDistance` / `CosineCoeficient` | **student** — zkopírovat z Cvičení 01 | kNN volá `distance.calculate` |
| ROC křivka + **AUC** (`roc_curve`, `roc_auc_score`), referenční `KNeighborsClassifier` (`predict_proba`) | `scikit-learn` | spojité skóre pro ROC — vlastní `predict` záměrně nemá `predict_proba` |
| `cross_validate` (orchestrace splitů a agregace), `demonstrate_leakage` | **předvyplněno** | spojuje studentův kód (kNN + validátor + metriky) do jednoho výpočtu |
| Načtení dat, konfigurace (`config.yaml` → dataclassy), všechny grafy | **předvyplněno** (`dataio/`) | není předmětem cvičení |

---

## Konfigurace projektu

### Soubor `config.yaml`

```yaml
data:
  imbalance_ratio: 0.1       # cílový podíl minoritní třídy pro demonstraci nevyváženosti
  random_state: 42

knn:
  k: 5
  k_values: [1, 3, 5, 7, 15, 31]   # přehled pro křivku přeučení

validation:
  test_size: 0.2
  n_folds: 5
  n_bootstrap: 100

feature_selection:           # experiment navazující na Cvičení 05
  n_features_grid: [2, 5, 10, 20, 30]
```

Změna `k`, počtu foldů, poměru nevyváženosti nebo mřížky příznaků představuje
pouhou úpravu konfigurace, nikoli kódu.

### Typovaná konfigurace (dataclassy)

Konfigurace se načítá funkcí `load_config()` a vrací jako instance dataclassy
`ExperimentConfig`. Přístup k hodnotám je přes **atributy**, ne slovníkové klíče:

```
# Místo:   cfg["knn"]["k"]     ← runtime chyba při překlepu
# Správně: cfg.knn.k            ← editor odhalí překlep okamžitě
```

Struktura dataclassů zrcadlí sekce YAML:

```
ExperimentConfig
    ├── data: DataConfig
    │       ├── imbalance_ratio: float
    │       └── random_state: int
    ├── knn: KNNConfig
    │       ├── k: int
    │       └── k_values: list[int]
    ├── validation: ValidationConfig
    │       ├── test_size: float
    │       ├── n_folds: int
    │       └── n_bootstrap: int
    └── feature_selection: FeatureSelectionConfig
            └── n_features_grid: list[int]
```

`load_config()` volá `validate_config()`, která ověří rozsahy hodnot
(`0 < imbalance_ratio < 0.5`, `k >= 1`, `n_folds >= 2`, `0 < test_size < 1`,
položky `n_features_grid` v `1..30`, …) a při neplatné konfiguraci vyvolá
srozumitelný `ValueError`. Obě funkce jsou **předvyplněné**.

---

## Pokyny k vypracování

Otevřete soubory popsané níže a nahraďte všechny výskyty
`raise NotImplementedError("Úkol: …")` funkčním kódem. Bloky implementujte v
uvedeném pořadí — pipeline i testy na něm závisí.

Komentáře ve tvaru `# assert  Ověřte, že …` jsou **nápovědy pro validaci
vstupů**. Napište odpovídající příkazy `assert` na daná místa — chrání kód před
obtížně dohledatelnými chybami při neplatném vstupu. (Nikdy nenechávejte aktivní
`assert` uvnitř nedokončené kostry.)

---

### Předpoklad: třídy vzdálenosti z Cvičení 01 — `src/distance.py`

kNN volá `self.distance.calculate(x, y)` pro seřazení sousedů — bez funkční
implementace se hlavní část cvičení nespustí. **Zkopírujte** vlastní implementaci
z Cvičení 01. Kostra tříd je připravena — stačí doplnit `is_metric` a
`calculate` v `EuclideanDistance`, `ManhattanDistance` a `CosineCoeficient`.
Metoda `create_distance_matrix` v abstraktní třídě `Distance` je předvyplněná
(kNN ji přímo nepoužívá — volá `calculate` po dvojicích — zůstává kvůli
návaznosti na předchozí cvičení).

Připomenutí vzorců:

- **Eukleidovská:** $d = \sqrt{\sum_j (a_j - b_j)^2}$
- **Manhattanská:** $d = \sum_j |a_j - b_j|$
- **Kosinová:** $d = 1 - \dfrac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \cdot \|\mathbf{b}\|}$ — ošetřete nulovou normu

---

### Blok I: Klasifikátor kNN — `src/knn.py`

Třída `KNNClassifier` je samostatná (nedědí z ničeho), ve stylu scikit-learn.
**Předvyplněny** jsou `__init__` (uloží `k`, injektuje `distance`, nastaví
`self.X_train_ = None`, `self.y_train_ = None`) a `fit` (zapamatuje si trénovací
data a vrátí `self` — kNN je líný). Implementujete metody `predict`, `save`
a `load`.

#### `predict(X)`

```
# Ověřte (assert), že model je nafitován (self.X_train_ není None)
# a že X.shape[1] == self.X_train_.shape[1].
#
# Pro každý řádek dotazu x v X:
#   1. spočítejte vzdálenost self.distance.calculate(x, x_train_i) ke KAŽDÉMU
#      trénovacímu bodu
#   2. najděte indexy k nejmenších vzdáleností (self.k)
#   3. z popisků těchto k sousedů vezměte většinový hlas
#
# Vraťte np.ndarray tvaru (n_dotazů,) s predikovanými popisky.
# Shody hlasů (ties) rozhodněte deterministicky — viz docstring.
```

> **Časté chyby:** porovnávání vzdáleností napříč různým počtem příznaků;
> `k` větší než počet trénovacích bodů; nedeterministické rozhodování shod
> (výsledek se pak mezi běhy liší).

#### `save(path)`

Uloží natrénovaný model do souboru `.npz` (formát NumPy). U kNN je
natrénovaným modelem celá trénovací množina, takže se ukládají pole
`x_train_` a `y_train_` spolu s hyperparametrem `k` — na rozdíl od PCA
z Cvičení 05, kde stačilo uložit několik malých polí (průměr, vlastní
čísla, vlastní vektory).

```
# Ověřte (assert), že model je nafitován (self.x_train_ není None).
#
# Uložte přes np.savez(path, …) tři pole:
#   x_train = self.x_train_
#   y_train = self.y_train_
#   k       = np.asarray(self.k)
```

#### `load(path)`

Načte model dříve uložený metodou `save` a naplní jím atributy `self.x_train_`,
`self.y_train_` a `self.k`. Vrací `self`, aby šlo řetězit `.load(path).predict(X)`.

```
# Ověřte (assert), že soubor existuje a je platný .npz uložený metodou save.
#
# Načtěte přes np.load(path) pole x_train, y_train a k a přiřaďte je do
# self.x_train_, self.y_train_ a self.k (k převeďte na int).
```

> Dvojice `save` / `load` je zařazena kvůli návaznosti na Cvičení 05. Pipeline
> ji volá ve fázi „Model jako uložené parametry" (natrénuje kNN → `save` do
> `models/knn_model.npz` → `load` → ověří, že znovunačtený model dává stejné
> predikce); pokrývá ji i `TestKNNPersistence`. Metrika `Distance` se do
> znovunačteného modelu injektuje až při vytvoření instance, do souboru se
> neukládá.

---

### Blok II: Metriky z kontingenční tabulky — `src/metrics.py`

#### `confusion_matrix(y_true, y_pred)`

```
# Ověřte (assert): stejná délka, binární hodnoty ({0, 1}).
#
# Sestavte tabulku 2×2 [[TN, FP], [FN, TP]] — řádek = skutečnost, sloupec =
# predikce, index 1 = pozitivní = maligní.
# Výsledek se musí shodovat s
#   sklearn.metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]).
```

#### `accuracy(cm)`, `precision(cm)`, `recall_sensitivity(cm)`, `specificity(cm)`, `f1_score(cm)`

Každá funkce bere **matici záměn `cm`** (ne surové popisky) a vrací `float`
podle vzorců z [teorie, odd. 3.2](#32-metriky-odvozené-z-matice). Ošetřete
**dělení nulou** (vraťte `0.0`). Nepočítejte metriky znovu z `y_true` /
`y_pred` — to je smysl celého bloku: všechno plyne z kontingenční tabulky.

> **Časté chyby:** prohození orientace matice — `[[TN, FP], [FN, TP]]`, řádek =
> skutečnost, sloupec = predikce (shoduje se s `confusion_matrix(y_true, y_pred,
> labels=[0, 1])`); počítání metrik znovu ze syrových `y_true` / `y_pred` místo
> z `cm`; návrat `NaN` místo `0.0` při nulovém jmenovateli; záměna `precision`
> (dělí `TP + FP`) a `recall` (dělí `TP + FN`); u `specificity` čtení `FP`
> z `cm[1, 0]` místo `cm[0, 1]`.

---

### Blok III: Robustní validace — `src/validation.py`

**Předvyplněny** jsou `Validator` (ABC s abstraktní `split`), `HoldOut`
(vzorový příklad — jedno zamíchané rozdělení) a `cross_validate` (orchestrace).
Implementujete metody `split` u dvou potomků.

#### `KFold.split(X, y)`

```
# __init__ dostane n_folds, shuffle=True, random_state=None (předvyplněno).
#
# Rozdělte indexy 0..n-1 do n_folds přihrádek (při shuffle nejprve zamíchejte
# generátorem se random_state). Pro každou přihrádku:
#   test_idx  = tato přihrádka
#   train_idx = všechny ostatní indexy
#   yield (train_idx, test_idx)
#
# Každý vzorek je testován právě jednou; přihrádky se v testu nepřekrývají
# a dohromady pokryjí všechny indexy.
```

> **Časté chyby:** `np.array_split(indices, n_folds)` vyřeší nerovnoměrné dělení
> za vás — nepočítejte velikost foldu ručně (off-by-one, ztracený „zbytkový"
> vzorek); `train_idx` je doplněk testovací přihrádky přes **všechny** ostatní
> foldy, ne jen sousední; míchání provádějte přes
> `np.random.default_rng(random_state)`, ne `np.random.shuffle` bez seedu
> (jinak nejde výsledek reprodukovat).

#### `Bootstrap.split(X, y)`

```
# __init__ dostane n_bootstrap, random_state=None (předvyplněno).
#
# n_bootstrap-krát:
#   train_idx = n indexů losovaných S OPAKOVÁNÍM z 0..n-1
#   test_idx  = out-of-bag = indexy, které se v train_idx neobjevily
#   yield (train_idx, test_idx)
#
# Průměrně je ~36,8 % vzorků out-of-bag (viz teorie, odd. 4).
```

> **Časté chyby:** `test_idx` je doplněk **unikátních** trénovacích indexů
> (`np.setdiff1d(np.arange(n), np.unique(train_idx))`), ne rozdíl vůči celému
> poli `train_idx`; trénovací pole musí mít délku `n` a **smí** obsahovat
> duplicity (to je podstata bootstrapu, ne chyba); vyšlo-li OOB prázdné
> (vylosovaly se všechny indexy), iteraci přeskočte — nevracejte prázdný test.

---

### Blok IV: Předzpracování dat — `dataio/preprocessing.py`

**Předvyplněna** je funkce `demonstrate_leakage` (spustí obě pořadí
standardizace a vrátí obě přesnosti vedle sebe). Implementujete tři funkce,
které používá zbytek pipeline.

#### `standardize_fit(x_train)`

```
# Ověřte (assert), že x_train má dva rozměry.
#
# Vraťte {"mean": …, "std": …} — průměr a populační směrodatnou odchylku
# (ddof=0) po sloupcích (axis=0), spočtené POUZE z x_train.
```

#### `standardize_apply(x, params)`

```
# Ověřte (assert), že params má klíče "mean" a "std" a že sedí počet příznaků.
#
# Vraťte (x − mean) / std jako float64. Nulovou směrodatnou odchylku
# (konstantní příznak) nejdřív nahraďte 1.0, aby nevznikly inf/nan.
```

> **Časté chyby:** **populační** směrodatná odchylka `ddof=0` (ne výběrová
> `ddof=1`) — jinak se rozejdete se `StandardScaler`; parametry se fitují **jen
> z `x_train`**, nikdy z celého `X` (to je přesně únik dat z oddílu 2.2);
> nulovou odchylku nahrazujte `1.0` až v `standardize_apply` (v `params["std"]`
> nech původní nuly), ať `fit` a `apply` zůstanou oddělené.

#### `subsample_imbalance(x, y, minority_ratio, random_state)`

```
# Ověřte (assert), že 0 < minority_ratio < 0.5 a že y obsahuje obě třídy.
#
# Podle vzorce v docstringu (sekce „Detail výpočtu") uberte z početnější
# strany tolik řádků, aby minoritní třída tvořila zhruba minority_ratio
# celku. Výběr přes np.random.default_rng(random_state) bez opakování,
# výsledné pořadí zamíchejte. Vraťte (x[keep], y[keep]).
```

> **Proč jsou tyto funkce úkolem.** Rozdělení standardizace na `fit` (parametry
> jen z trénovacích dat) a `apply` (na train i test) je jádro obrany proti úniku
> dat z oddílu 2.2 — stojí za to napsat si ho ručně. `subsample_imbalance` je
> nástroj k demonstraci z oddílu 2.3. Správnost `standardize_*` si ověříte proti
> `sklearn.preprocessing.StandardScaler`.

---

## Lokální testování

Spusťte automatické testy příkazem:

```bash
python -m pytest test_cviceni_06.py -v
```

| Třída testů | Co ověřuje |
|:---|:---|
| `TestKNN` | `predict` se shoduje se `sklearn.neighbors.KNeighborsClassifier` na dobře oddělené množině (trénovací i nové body); **remíza hlasů** → nižší třída; validační `assert` v `predict` (bez `fit`, neshoda počtu příznaků). Používá `DummyDistance`. |
| `TestKNNPersistence` | round-trip `save` → `load` (přes `tmp_path`): znovunačtený model dává stejné predikce a nese stejná naučená pole `x_train_` / `y_train_` / `k` (`k` se bere ze souboru, ne z konstruktoru). Obdoba `TestPCAPersistence` z Cvičení 05. |
| `TestMetrics` | `confusion_matrix` se shoduje se `sklearn.metrics.confusion_matrix` (stejná orientace); `accuracy` / `precision` / `recall_sensitivity` / `f1_score` se shodují se scikit-learn; `specificity` ověřena ručně. |
| `TestValidation` | `KFold`: testovací přihrádky se nepřekrývají a pokryjí všechny indexy (struktura jako `sklearn.model_selection.KFold`); `Bootstrap`: trénink má opakování, out-of-bag test je doplněk vylosovaných; `cross_validate` vrací rozumné agregáty na jednoduchém modelu. |
| `TestPreprocessing` | `standardize_fit` + `standardize_apply` se shodují se `sklearn.preprocessing.StandardScaler` (včetně ošetření konstantního sloupce); `subsample_imbalance` dodrží cílový podíl tříd a vrací neprázdný výstup s oběma třídami. |

Testy používají malá syntetická data a vlastní pomocnou třídu `DummyDistance`
(eukleidovská implementace přímo v testovacím souboru), takže testy kNN běží
nezávisle na tom, zda už máte hotový `src/distance.py`.

Dokud nejsou příslušné metody hotové, testy, které je volají, se hlásí jako
**`xfail`** (očekávané selhání na `NotImplementedError`) a celá sada skončí s
návratovým kódem 0 — jde o záměrné chování, nikoli o chybu. Jakmile metodu
doplníte, stejný test začne procházet (`xpass` → `pass`).

Průběžně ověřujte i celou pipeline:

```bash
python cviceni_06.py
```

Kroky s neimplementovanými metodami se přeskočí s hláškou `[NENI HOTOVO] Úkol: …`;
ostatní proběhnou normálně a uloží grafy do `graphs/`.

---

## Doplňkové (papírové) příklady

Soubor `priklady_06.md` obsahuje třináct číselných příkladů k ručnímu výpočtu:

1. **Matice záměn ručně** — z krátkých `y_true` / `y_pred` sestavit tabulku
   TP/FP/TN/FN (pozitivní = maligní).
2. **Metriky z tabulky** — z dané matice záměn spočítat přesnost, preciznost,
   senzitivitu/Se, specificitu/Sp a F1.
3. **Proč je accuracy zavádějící** — nevyvážená matice záměn (~90/10, líný
   majoritní klasifikátor); spočítat přesnost (vysoká) vs. senzitivitu (nízká)
   a interpretovat.
4. **k-fold rozdělení** — 10 indexů a `n_folds = 5` bez zamíchání; vypsat
   train/test indexy pro každý fold.
5. **kNN ručně** — několik označených 2D bodů a dotaz, `k = 3`; spočítat
   vzdálenosti, vybrat 3 nejbližší, hlasovat.
6. **Bootstrap a out-of-bag** — z jednoho losování s opakováním určit OOB
   množinu a podíl; porovnat s `e⁻¹`.
7. **k-fold se zamícháním** — z dané permutace indexů sestavit foldy; srovnat
   s příkladem 4.
8. **Vliv `k` na predikci** — tatáž data jako v příkladu 5 pro `k = 1, 3, 5`;
   propojení s kompromisem zkreslení–rozptyl.
9. **Vliv metriky vzdálenosti** — manhattanská vs. eukleidovská metrika na
   datech z příkladu 5; role injektované `Distance`.
10. **Únik dat při standardizaci** — standardizace z trénovacích parametrů vs.
    ze všech dat; číselné srovnání posunu.
11. **ROC a AUC ručně** — ze skóre a tříd sestavit ROC body přes prahy a
    spočítat AUC lichoběžníkovým pravidlem.
12. **Agregace křížové validace** — průměr a směrodatná odchylka senzitivity
    přes foldy; proč `cross_validate` vrací obojí.
13. **Volba prahu podle ceny chyby** — dva prahy nad skóre z příkladu 11,
    porovnání podle nákladové funkce (FN vs. FP).

Příklady 1–2, 5, 8, 9 a 13 odpovídají krokům programovaným v `src/metrics.py`
a `src/knn.py`; příklady 4, 6 a 7 upevňují logiku `KFold.split` a
`Bootstrap.split`; příklad 12 se váže k `cross_validate`; příklady 3, 10 a 11
jsou pojmové a doplňují metodický blok. **Řešení nejsou součástí repozitáře** —
výsledky si ověřte u vyučujícího nebo výpočtem v `numpy` / `scikit-learn`.

---

## Odevzdání

Úloha se odevzdává prostřednictvím systému **GitHub Classroom**. Po dokončení
implementace proveďte:

```bash
git add src/distance.py src/knn.py src/metrics.py src/validation.py dataio/preprocessing.py
git commit -m "Implementace cvičení 6"
git push
```

Po přijetí příkazu `push` se automaticky spustí testovací skripty, které ověří
správnost výpočtů. Výsledek bude zobrazen přímo v rozhraní GitHub u vašeho
repozitáře formou zelené fajfky (úspěch) nebo červeného křížku (neúspěch).

> **Soubory, které se neodevzdávají:** `src/__init__.py`, `dataio/__init__.py`,
> `dataio/loader.py`, `dataio/config_manager.py`, `dataio/plotting.py`,
> `cviceni_06.py` a `test_cviceni_06.py` jsou předvyplněny nebo se nemají
> měnit. Systém hodnotí soubory `src/distance.py`, `src/knn.py`,
> `src/metrics.py`, `src/validation.py` a `dataio/preprocessing.py`.
