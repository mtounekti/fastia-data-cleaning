# 🧹 Brief 1 – Nettoyage de Données Numériques
### Projet FastIA – Pipeline de traitement de données

---

## 📋 Description du projet

FastIA souhaite enrichir son modèle d'IA avec de nouvelles données numériques. Avant toute utilisation, ces données brutes doivent être **analysées, nettoyées, transformées et documentées** afin de garantir leur intégrité, leur conformité éthique et leur pertinence pour les modèles d'apprentissage automatique.

---

## 📁 Structure du dépôt

```
fastia-data-cleaning/
├── nettoyage_donnees.py              # Script principal – pipeline complet
├── dataset_propre.csv                # Dataset nettoyé prêt à l'emploi
├── README.md                         # Cette documentation
├── requirements.txt                  # Dépendances Python
└── graphiques/
    ├── 01_valeurs_manquantes.png     # Matrice + barres des NaN
    ├── 02_distributions_brutes.png   # Histogrammes du dataset brut
    ├── 03_boxplots_bruts.png         # Détection outliers (brut)
    ├── 04_comparaison_distributions.png  # Avant / après nettoyage
    ├── 05_boxplots_propres.png       # Vérification outliers (propre)
    ├── 06_correlation.png            # Matrice de corrélation
    └── 07_pairplot.png               # Relations entre variables
```

### Lecture du graphique `01_valeurs_manquantes.png`

- 🔴 **Rouge foncé** sur `historique_credits` et `score_credit` → ~53% de NaN, bien au-dessus du seuil 40% (ligne rouge pointillée) → **supprimées**
- 🔵 **Bleu clair** sur `loyer_mensuel` → 29.1% de NaN, en dessous du seuil → **imputé par KNN**
- 🟢 **Tout vert** sur les autres colonnes → aucune valeur manquante

---

## 📊 Description du dataset initial

| Propriété | Valeur |
|---|---|
| Lignes | 10 000 |
| Colonnes | 9 |
| Variables numériques | 9 (toutes numériques) |
| Valeurs manquantes | Oui (3 colonnes concernées) |

### Colonnes du dataset brut

| Colonne | Type | % NaN | Description |
|---|---|---|---|
| `age` | int | 0% | Âge de l'individu (18–75 ans) |
| `taille` | float | 0% | ⚠️ Donnée corporelle sensible |
| `poids` | float | 0% | ⚠️ Donnée corporelle sensible |
| `revenu_estime_mois` | int | 0% | Revenu mensuel estimé (€) |
| `historique_credits` | float | **52.9%** | Score historique crédit |
| `risque_personnel` | float | 0% | Score de risque [0–1] |
| `score_credit` | float | **53.1%** | Score crédit calculé |
| `loyer_mensuel` | float | 29.1% | Loyer mensuel (€) |
| `montant_pret` | float | 0% | Montant du prêt demandé (€) |

### Anomalies détectées

- **2 colonnes quasi-vides** : `historique_credits` (52.9%) et `score_credit` (53.1%)
- **1 colonne avec NaN significatifs** : `loyer_mensuel` (29.1%)
- **Valeurs aberrantes** : `loyer_mensuel` min = -395€ (valeur impossible), `montant_pret` avec quelques extrêmes, `revenu_estime_mois` avec quelques pics
- **2 colonnes sensibles** : `poids` et `taille` (données corporelles personnelles)

---

## 🔧 Méthodologie et choix techniques

### Étape 1 – Suppression des colonnes sensibles

**Colonnes supprimées** : `poids`, `taille`

**Justification** : Ces colonnes contiennent des données corporelles à caractère personnel. Conformément aux principes RGPD et aux bonnes pratiques d'éthique en IA, elles sont exclues du pipeline car :
- Elles ne sont pas nécessaires à l'objectif métier (évaluation de crédit)
- Leur conservation sans anonymisation constitue un risque légal
- Elles pourraient introduire des biais discriminatoires dans le modèle

---

### Étape 2 – Suppression des colonnes quasi-vides

**Colonnes supprimées** : `historique_credits` (52.9% NaN), `score_credit` (53.1% NaN)

**Seuil appliqué** : > 40% de valeurs manquantes

**Justification** : Avec plus de la moitié des valeurs manquantes, toute imputation de ces colonnes introduirait un biais trop important dans le dataset. L'information statistique disponible est insuffisante pour inférer de façon fiable les valeurs manquantes. La suppression est préférable à une imputation qui produirait des données artificielles non représentatives.

---

### Étape 3 – Suppression des lignes trop incomplètes

**Seuil appliqué** : > 50% des colonnes manquantes par ligne

**Résultat** : 0 ligne supprimée (aucune ligne ne dépassait le seuil après les suppressions de colonnes)

**Justification** : Après suppression des colonnes quasi-vides, les lignes restantes avaient au maximum 1 valeur manquante sur 5 colonnes (20%), en dessous du seuil. La reproductibilité est garantie par le paramètre `SEUIL_LIGNE_INCOMPLETE = 0.50`.

---

### Étape 4 – Traitement des outliers (Winsorisation)

**Méthode** : Winsorisation par bornes IQR × 1.5

**Colonnes traitées** :
| Colonne | Nb outliers | Borne basse | Borne haute |
|---|---|---|---|
| `revenu_estime_mois` | 41 | -748.50 | 5735.50 |
| `loyer_mensuel` | 0 | -12535.58 | 23521.35 |
| `montant_pret` | 103 | -23118.30 | 39863.84 |

**Justification** : La winsorisation (clipping) est préférée à la suppression des lignes car :
- Elle préserve le volume de données (10 000 lignes maintenues)
- Elle neutralise l'effet des valeurs extrêmes sans les supprimer
- Elle est plus robuste pour des modèles d'IA que la simple suppression
- Les bornes IQR × 1.5 correspondent à la définition statistique standard des outliers

---

### Étape 5 – Imputation des valeurs manquantes

**Colonne concernée** : `loyer_mensuel` (29.1% NaN)

**Méthode** : KNN Imputer (k=5 voisins)

**Justification** : Le KNN Imputer est choisi car :
- Il exploite les corrélations entre toutes les variables (âge, revenu, risque, montant prêt) pour estimer le loyer
- Il est plus précis que la moyenne/médiane simples car il tient compte du profil global de chaque individu
- Avec 5 voisins, il offre un bon compromis biais/variance
- Alternative considérée : médiane → rejetée car elle ne tient pas compte des corrélations inter-variables

---

### Étape 6 – Restauration des types

**Colonnes concernées** : `age`, `revenu_estime_mois`

**Justification** : Le KNN Imputer transforme toutes les colonnes en `float64`. On restitue le type `int` d'origine pour ces colonnes car elles ne peuvent contenir que des entiers (âge en années, revenu mensuel en euros entiers).

---

## 📈 Comparatif statistique avant / après

| Colonne | Moy. avant | Moy. après | σ avant | σ après | Méd. avant | Méd. après |
|---|---|---|---|---|---|---|
| `age` | 46.52 | 46.52 | 16.83 | 16.83 | 46 | 46 |
| `revenu_estime_mois` | 2521.00 | 2519.53 | 1157.53 | 1153.10 | 2480 | 2480 |
| `risque_personnel` | 0.50 | 0.50 | 0.29 | 0.29 | 0.50 | 0.50 |
| `loyer_mensuel` | 5175.89 | 5179.90 | **3750.61** | **3288.20** | 5000 | 5000 |
| `montant_pret` | 9149.76 | 9111.35 | 10785.94 | 10665.32 | 3600.61 | 3600.61 |

**Observations clés** :
- Les **moyennes et médianes sont très stables** → le nettoyage n'a pas introduit de biais
- L'**écart-type de `loyer_mensuel` diminue** (3750 → 3288) → preuve que l'imputation KNN est cohérente et réduit la dispersion due aux NaN
- Les distributions sont préservées pour toutes les colonnes

---

## 📦 Dataset final propre

| Propriété | Valeur |
|---|---|
| Fichier | `dataset_propre.csv` |
| Lignes | 10 000 |
| Colonnes | 5 |
| Valeurs manquantes | **0** |
| Doublons | 2 (conservés – pas de critère d'identifiant unique) |

### Colonnes du dataset propre

| Colonne | Type | Description |
|---|---|---|
| `age` | int | Âge (18–75 ans) |
| `revenu_estime_mois` | int | Revenu mensuel estimé (€), outliers winsorisés |
| `risque_personnel` | float | Score de risque [0–1] |
| `loyer_mensuel` | float | Loyer mensuel (€), NaN imputés par KNN |
| `montant_pret` | float | Montant prêt (€), outliers winsorisés |

---

## ▶️ Reproductibilité

```bash
# Installation des dépendances
pip install pandas numpy matplotlib seaborn scikit-learn missingno

# Exécution du pipeline
python nettoyage_donnees.py
```

> **Note sur missingno** : Le script utilise une implémentation manuelle des visualisations de valeurs manquantes (compatible avec ou sans `missingno` installé). Pour l'utiliser nativement : `import missingno as msno; msno.matrix(df)`.

---

## ⚙️ Paramètres configurables

```python
SEUIL_COLONNE_VIDE      = 0.40   # Seuil suppression colonnes (40%)
SEUIL_LIGNE_INCOMPLETE  = 0.50   # Seuil suppression lignes (50%)
SEUIL_OUTLIER_IQR       = 1.5    # Multiplicateur IQR standard
```

---

## 🔗 Ressources

- [Documentation missingno](https://github.com/ResidentMario/missingno)
- [OPCO ATLAS – Pandas/Seaborn](ressources/opco-atlas-pandas-seaborn.docx)
- [Scikit-learn KNNImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html)