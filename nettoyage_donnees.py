# =============================================================================
# BRIEF 1 - NETTOYAGE DE DONNÉES NUMÉRIQUES
# Projet FastIA – Pipeline de traitement de données
# =============================================================================
# Ce script réalise l'ensemble du pipeline de traitement :
#   1. Analyse exploratoire (EDA)
#   2. Détection des anomalies (valeurs manquantes, outliers)
#   3. Nettoyage et transformation
#   4. Export du dataset propre
#   5. Rapport statistique avant/après
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.impute import KNNImputer
import warnings
import os

warnings.filterwarnings("ignore")

# ── Paramètres globaux ──────────────────────────────────────────────────────
FICHIER_SOURCE = "fichier-de-donnees-numeriques-69202f25dea8b267811864.csv"
FICHIER_PROPRE = "dataset_propre.csv"
SEUIL_COLONNE_VIDE = 0.40   # On supprime les colonnes avec > 40% de NaN
SEUIL_LIGNE_INCOMPLETE = 0.50  # On supprime les lignes avec > 50% de NaN
SEUIL_OUTLIER_IQR = 1.5        # Multiplicateur IQR standard

# Palette couleurs cohérente pour les graphiques
COULEUR_AVANT = "#E07B54"
COULEUR_APRES = "#5B8DB8"

os.makedirs("graphiques", exist_ok=True)

# =============================================================================
# SECTION 1 – CHARGEMENT ET PREMIER APERÇU
# =============================================================================

print("=" * 65)
print("  ÉTAPE 1 – CHARGEMENT DU DATASET")
print("=" * 65)

df_brut = pd.read_csv(FICHIER_SOURCE)

print(f"\n✔  Dataset chargé : {df_brut.shape[0]} lignes × {df_brut.shape[1]} colonnes")
print(f"\n── Aperçu des 5 premières lignes ──")
print(df_brut.head().to_string())

print(f"\n── Types de données ──")
print(df_brut.dtypes.to_string())

print(f"\n── Statistiques descriptives brutes ──")
print(df_brut.describe().round(2).to_string())

# =============================================================================
# SECTION 2 – ANALYSE EXPLORATOIRE (EDA)
# =============================================================================

print("\n" + "=" * 65)
print("  ÉTAPE 2 – ANALYSE EXPLORATOIRE")
print("=" * 65)

# ── 2.1 Valeurs manquantes ──────────────────────────────────────────────────

nb_manquants = df_brut.isnull().sum()
pct_manquants = (nb_manquants / len(df_brut) * 100).round(2)

rapport_manquants = pd.DataFrame({
    "Valeurs manquantes": nb_manquants,
    "Pourcentage (%)":    pct_manquants
}).sort_values("Pourcentage (%)", ascending=False)

print("\n── Rapport des valeurs manquantes ──")
print(rapport_manquants.to_string())

# ── 2.2 Colonnes sensibles ──────────────────────────────────────────────────
# Les colonnes 'poids' et 'taille' sont des données à caractère personnel
# (données corporelles) – elles doivent être signalées et exclues du modèle
# conformément aux bonnes pratiques RGPD / éthique IA.
COLONNES_SENSIBLES = ["poids", "taille"]
print(f"\n⚠  Colonnes sensibles identifiées (données corporelles) : {COLONNES_SENSIBLES}")
print("   → Ces colonnes seront supprimées avant tout traitement de modélisation.")

# ── 2.3 Visualisation des valeurs manquantes (style missingno) ─────────────
# Simulation du graphique missingno.matrix() sans la bibliothèque externe

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Analyse des valeurs manquantes", fontsize=14, fontweight="bold")

# Heatmap des NaN par colonne
ax1 = axes[0]
data_nan = df_brut.isnull().astype(int)
# Affichage d'un échantillon de 300 lignes pour la lisibilité
echantillon = data_nan.sample(300, random_state=42).reset_index(drop=True)
im = ax1.imshow(echantillon.T, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
ax1.set_yticks(range(len(df_brut.columns)))
ax1.set_yticklabels(df_brut.columns, fontsize=9)
ax1.set_xlabel("Échantillon de 300 lignes", fontsize=9)
ax1.set_title("Matrice des valeurs manquantes\n(vert=présent, rouge=absent)", fontsize=10)
plt.colorbar(im, ax=ax1, fraction=0.04)

# Barres du pourcentage manquant
ax2 = axes[1]
colors = [COULEUR_AVANT if p > 40 else "#AACCE0" for p in pct_manquants.values]
barres = ax2.barh(pct_manquants.index, pct_manquants.values, color=colors, edgecolor="white")
ax2.axvline(40, color="red", linestyle="--", linewidth=1.5, label="Seuil 40% (suppression)")
ax2.set_xlabel("% de valeurs manquantes", fontsize=9)
ax2.set_title("Taux de valeurs manquantes\npar colonne", fontsize=10)
ax2.legend(fontsize=8)
for barre, val in zip(barres, pct_manquants.values):
    if val > 0:
        ax2.text(val + 0.5, barre.get_y() + barre.get_height()/2,
                 f"{val:.1f}%", va="center", fontsize=8)

plt.tight_layout()
plt.savefig("graphiques/01_valeurs_manquantes.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✔  Graphique sauvegardé : graphiques/01_valeurs_manquantes.png")

# ── 2.4 Distribution des variables numériques ──────────────────────────────

colonnes_num = df_brut.select_dtypes(include=np.number).columns.tolist()

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
fig.suptitle("Distribution des variables – Dataset BRUT", fontsize=14, fontweight="bold")

for i, col in enumerate(colonnes_num):
    ax = axes[i]
    donnees_valides = df_brut[col].dropna()
    sns.histplot(donnees_valides, kde=True, ax=ax, color=COULEUR_AVANT, edgecolor="white")
    ax.set_title(col, fontsize=10, fontweight="bold")
    ax.set_xlabel("")
    ax.axvline(donnees_valides.mean(), color="navy", linestyle="--",
               linewidth=1.2, label=f"Moy. {donnees_valides.mean():.1f}")
    ax.axvline(donnees_valides.median(), color="green", linestyle=":",
               linewidth=1.2, label=f"Méd. {donnees_valides.median():.1f}")
    ax.legend(fontsize=7)

# Masquer les axes vides éventuels
for j in range(len(colonnes_num), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig("graphiques/02_distributions_brutes.png", dpi=150, bbox_inches="tight")
plt.close()
print("✔  Graphique sauvegardé : graphiques/02_distributions_brutes.png")

# ── 2.5 Détection des outliers avec boxplots ───────────────────────────────

fig, axes = plt.subplots(3, 3, figsize=(15, 10))
axes = axes.flatten()
fig.suptitle("Boxplots – Détection des outliers (dataset brut)", fontsize=14, fontweight="bold")

rapport_outliers = {}

for i, col in enumerate(colonnes_num):
    ax = axes[i]
    donnees_valides = df_brut[col].dropna()

    Q1  = donnees_valides.quantile(0.25)
    Q3  = donnees_valides.quantile(0.75)
    IQR = Q3 - Q1
    borne_basse = Q1 - SEUIL_OUTLIER_IQR * IQR
    borne_haute = Q3 + SEUIL_OUTLIER_IQR * IQR

    nb_outliers = ((donnees_valides < borne_basse) | (donnees_valides > borne_haute)).sum()
    pct_outliers = nb_outliers / len(donnees_valides) * 100
    rapport_outliers[col] = {"nb": nb_outliers, "pct": round(pct_outliers, 2),
                              "borne_basse": round(borne_basse, 2),
                              "borne_haute": round(borne_haute, 2)}

    sns.boxplot(y=donnees_valides, ax=ax, color=COULEUR_AVANT,
                flierprops=dict(marker="o", markerfacecolor="red", markersize=3, alpha=0.5))
    ax.set_title(f"{col}\n({nb_outliers} outliers, {pct_outliers:.1f}%)", fontsize=9)
    ax.set_ylabel("")

for j in range(len(colonnes_num), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig("graphiques/03_boxplots_bruts.png", dpi=150, bbox_inches="tight")
plt.close()
print("✔  Graphique sauvegardé : graphiques/03_boxplots_bruts.png")

print("\n── Rapport outliers (méthode IQR × 1.5) ──")
df_outliers = pd.DataFrame(rapport_outliers).T
print(df_outliers.to_string())

# =============================================================================
# SECTION 3 – NETTOYAGE ET TRANSFORMATION
# =============================================================================

print("\n" + "=" * 65)
print("  ÉTAPE 3 – NETTOYAGE ET TRANSFORMATION")
print("=" * 65)

# On travaille sur une copie pour préserver les données brutes
df = df_brut.copy()

# ── 3.1 Suppression des colonnes sensibles ─────────────────────────────────
# Décision : suppression de 'poids' et 'taille' (données corporelles)
# Ces données sont considérées comme sensibles selon le RGPD et les pratiques
# éthiques en IA. Elles ne sont pas nécessaires à l'objectif métier (crédit).

df = df.drop(columns=COLONNES_SENSIBLES)
print(f"\n[3.1] ✔  Colonnes sensibles supprimées : {COLONNES_SENSIBLES}")
print(f"       Dimensions restantes : {df.shape}")

# ── 3.2 Suppression des colonnes quasi-vides (> 40% NaN) ──────────────────
# Les colonnes 'historique_credits' (52.9%) et 'score_credit' (53.1%) sont
# trop incomplètes pour être imputées de façon fiable. Leur signal utile
# est limité et toute imputation introduirait un biais majeur.

colonnes_a_supprimer = [col for col in df.columns
                        if df[col].isnull().mean() > SEUIL_COLONNE_VIDE]
print(f"\n[3.2] Colonnes avec > {SEUIL_COLONNE_VIDE*100:.0f}% de NaN → supprimées :")
for col in colonnes_a_supprimer:
    print(f"       - {col} ({df[col].isnull().mean()*100:.1f}% manquants)")

df = df.drop(columns=colonnes_a_supprimer)
print(f"       Dimensions restantes : {df.shape}")

# ── 3.3 Suppression des lignes trop incomplètes (> 50% colonnes NaN) ──────
# Conformément aux modalités : on supprime les lignes très incomplètes.
# Seuil fixé à 50% des colonnes restantes manquantes.

nb_avant = len(df)
nb_colonnes = df.shape[1]
df = df[df.isnull().sum(axis=1) / nb_colonnes <= SEUIL_LIGNE_INCOMPLETE]
nb_supprime = nb_avant - len(df)
print(f"\n[3.3] ✔  Lignes trop incomplètes supprimées : {nb_supprime} "
      f"({nb_supprime/nb_avant*100:.2f}% du dataset)")
print(f"       Dimensions restantes : {df.shape}")

# ── 3.4 Traitement des outliers ────────────────────────────────────────────
# Stratégie retenue : winsorisation (clipping) sur les bornes IQR × 1.5
# Avantage : préserve les lignes tout en neutralisant les valeurs extrêmes.
# Colonnes concernées : revenu_estime_mois, loyer_mensuel, montant_pret
# Note : 'age' est borné [18-75] → pas d'outlier métier détecté.

colonnes_a_clipper = ["revenu_estime_mois", "loyer_mensuel", "montant_pret"]
print("\n[3.4] Traitement des outliers par winsorisation (IQR × 1.5) :")

for col in colonnes_a_clipper:
    if col not in df.columns:
        continue
    donnees = df[col].dropna()
    Q1  = donnees.quantile(0.25)
    Q3  = donnees.quantile(0.75)
    IQR = Q3 - Q1
    borne_basse = Q1 - SEUIL_OUTLIER_IQR * IQR
    borne_haute = Q3 + SEUIL_OUTLIER_IQR * IQR

    nb_avant_clip = ((df[col] < borne_basse) | (df[col] > borne_haute)).sum()
    df[col] = df[col].clip(lower=borne_basse, upper=borne_haute)
    print(f"       - {col} : {nb_avant_clip} valeurs winsorisées "
          f"[{borne_basse:.2f} ; {borne_haute:.2f}]")

# ── 3.5 Imputation des valeurs manquantes ──────────────────────────────────
# Colonne 'loyer_mensuel' : 29% de NaN → imputation par KNN (k=5)
# Le KNN Imputer utilise les voisins les plus proches selon toutes les autres
# variables pour estimer le loyer. C'est plus précis que la moyenne/médiane
# car il tient compte des corrélations entre variables.

print("\n[3.5] Imputation des valeurs manquantes restantes :")
print(f"       NaN avant imputation :")
print(df.isnull().sum()[df.isnull().sum() > 0].to_string())

# KNN Imputation sur l'ensemble du dataframe (k=5 voisins)
imputer = KNNImputer(n_neighbors=5)
colonnes_df = df.columns.tolist()
df_impute = pd.DataFrame(imputer.fit_transform(df), columns=colonnes_df)

print(f"\n       NaN après imputation KNN (k=5) :")
print(df_impute.isnull().sum().to_string())
print("       ✔  Aucune valeur manquante restante.")

df = df_impute.copy()

# ── 3.6 Arrondi des colonnes entières ─────────────────────────────────────
# 'age' et 'revenu_estime_mois' étaient des entiers dans les données brutes.
# Après le KNN, ils peuvent avoir des décimales → on les remet en entiers.

for col in ["age", "revenu_estime_mois"]:
    if col in df.columns:
        df[col] = df[col].round(0).astype(int)

print("\n[3.6] ✔  Colonnes entières restaurées (age, revenu_estime_mois)")

# ── 3.7 Vérification finale du dataset propre ─────────────────────────────

print("\n[3.7] ── Vérification finale ──")
print(f"  Dimensions : {df.shape[0]} lignes × {df.shape[1]} colonnes")
print(f"  Valeurs manquantes totales : {df.isnull().sum().sum()}")
print(f"  Doublons : {df.duplicated().sum()}")
print(f"  Colonnes : {list(df.columns)}")

# =============================================================================
# SECTION 4 – VISUALISATIONS APRÈS NETTOYAGE
# =============================================================================

print("\n" + "=" * 65)
print("  ÉTAPE 4 – VISUALISATIONS APRÈS NETTOYAGE")
print("=" * 65)

# ── 4.1 Distributions avant / après sur les colonnes communes ─────────────
colonnes_communes = [col for col in df.columns if col in df_brut.columns]

fig, axes = plt.subplots(len(colonnes_communes), 2,
                         figsize=(13, len(colonnes_communes) * 3))
fig.suptitle("Comparaison distributions – Avant vs Après nettoyage",
             fontsize=14, fontweight="bold")

for i, col in enumerate(colonnes_communes):
    # Avant
    ax_avant = axes[i][0]
    sns.histplot(df_brut[col].dropna(), kde=True, ax=ax_avant,
                 color=COULEUR_AVANT, edgecolor="white")
    ax_avant.set_title(f"{col} — AVANT", fontsize=10)
    ax_avant.set_xlabel("")

    # Après
    ax_apres = axes[i][1]
    sns.histplot(df[col], kde=True, ax=ax_apres,
                 color=COULEUR_APRES, edgecolor="white")
    ax_apres.set_title(f"{col} — APRÈS", fontsize=10)
    ax_apres.set_xlabel("")

plt.tight_layout()
plt.savefig("graphiques/04_comparaison_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("✔  Graphique sauvegardé : graphiques/04_comparaison_distributions.png")

# ── 4.2 Boxplots après nettoyage ──────────────────────────────────────────

fig, axes = plt.subplots(1, len(df.columns), figsize=(14, 5))
fig.suptitle("Boxplots après nettoyage – Vérification outliers",
             fontsize=13, fontweight="bold")

for i, col in enumerate(df.columns):
    sns.boxplot(y=df[col], ax=axes[i], color=COULEUR_APRES,
                flierprops=dict(marker="o", markerfacecolor="orange",
                                markersize=3, alpha=0.7))
    axes[i].set_title(col, fontsize=9, fontweight="bold")
    axes[i].set_ylabel("")

plt.tight_layout()
plt.savefig("graphiques/05_boxplots_propres.png", dpi=150, bbox_inches="tight")
plt.close()
print("✔  Graphique sauvegardé : graphiques/05_boxplots_propres.png")

# ── 4.3 Matrice de corrélation ────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 7))
corr_matrix = df.corr(numeric_only=True)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, vmin=-1, vmax=1,
            ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.8})
ax.set_title("Matrice de corrélation – Dataset propre", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("graphiques/06_correlation.png", dpi=150, bbox_inches="tight")
plt.close()
print("✔  Graphique sauvegardé : graphiques/06_correlation.png")

# ── 4.4 Pairplot (aperçu des relations entre variables) ───────────────────
# Sous-ensemble pour la lisibilité (4 colonnes clés)
colonnes_pairplot = ["age", "revenu_estime_mois", "loyer_mensuel", "montant_pret"]
colonnes_pairplot = [c for c in colonnes_pairplot if c in df.columns]

g = sns.pairplot(df[colonnes_pairplot], plot_kws={"alpha": 0.3, "color": COULEUR_APRES},
                 diag_kws={"color": COULEUR_APRES})
g.fig.suptitle("Pairplot – Variables financières clés (après nettoyage)",
               y=1.02, fontsize=12, fontweight="bold")
g.savefig("graphiques/07_pairplot.png", dpi=120, bbox_inches="tight")
plt.close()
print("✔  Graphique sauvegardé : graphiques/07_pairplot.png")

# =============================================================================
# SECTION 5 – ANALYSE STATISTIQUE AVANT / APRÈS
# =============================================================================

print("\n" + "=" * 65)
print("  ÉTAPE 5 – ANALYSE STATISTIQUE AVANT / APRÈS")
print("=" * 65)

stats_avant = df_brut[colonnes_communes].describe().round(3)
stats_apres = df[colonnes_communes].describe().round(3)

print("\n── Statistiques AVANT nettoyage ──")
print(stats_avant.to_string())
print("\n── Statistiques APRÈS nettoyage ──")
print(stats_apres.to_string())

# Tableau comparatif condensé
print("\n── Tableau comparatif (moyenne / écart-type / médiane) ──")
lignes = []
for col in colonnes_communes:
    lignes.append({
        "Colonne": col,
        "Moy. avant":  round(df_brut[col].mean(), 2),
        "Moy. après":  round(df[col].mean(), 2),
        "σ avant":     round(df_brut[col].std(), 2),
        "σ après":     round(df[col].std(), 2),
        "Méd. avant":  round(df_brut[col].median(), 2),
        "Méd. après":  round(df[col].median(), 2),
    })
df_comparatif = pd.DataFrame(lignes).set_index("Colonne")
print(df_comparatif.to_string())

# =============================================================================
# SECTION 6 – EXPORT DU DATASET PROPRE
# =============================================================================

print("\n" + "=" * 65)
print("  ÉTAPE 6 – EXPORT")
print("=" * 65)

df.to_csv(FICHIER_PROPRE, index=False)
print(f"\n✔  Dataset propre exporté : {FICHIER_PROPRE}")
print(f"   {df.shape[0]} lignes × {df.shape[1]} colonnes")
print(f"   Colonnes : {list(df.columns)}")

# ── Résumé final des opérations ───────────────────────────────────────────
print("\n" + "=" * 65)
print("  RÉSUMÉ DU PIPELINE")
print("=" * 65)
print(f"""
  Dataset initial   : {df_brut.shape[0]} lignes × {df_brut.shape[1]} colonnes
  Dataset final     : {df.shape[0]} lignes × {df.shape[1]} colonnes

  Opérations réalisées :
  ┌─────────────────────────────────────────────────────────────┐
  │ 1. Suppression colonnes sensibles : poids, taille           │
  │ 2. Suppression colonnes quasi-vides (>40% NaN) :            │
  │    historique_credits (52.9%), score_credit (53.1%)         │
  │ 3. Suppression lignes trop incomplètes (>50% NaN)           │
  │ 4. Winsorisation des outliers (IQR × 1.5) :                 │
  │    revenu_estime_mois, loyer_mensuel, montant_pret           │
  │ 5. Imputation KNN (k=5) sur loyer_mensuel                   │
  │ 6. Restauration types entiers (age, revenu_estime_mois)     │
  └─────────────────────────────────────────────────────────────┘
""")

print("  Pipeline terminé avec succès ✅")
print("  Graphiques disponibles dans le dossier : graphiques/")