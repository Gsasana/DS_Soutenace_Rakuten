import os
import warnings

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from math import sqrt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.preprocessing import StandardScaler
from statsmodels.api import OLS
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from tabulate import tabulate
from matplotlib.colors import ListedColormap


def analyse_univariee(df):
    """
    Effectue une analyse univariée sur toutes les variables d'un DataFrame.

    Pour les variables numériques, des statistiques descriptives, un histogramme et un boxplot sont générés.
    Pour les variables catégorielles, des statistiques descriptives, un tableau de fréquence et un barplot sont générés.

    Paramètres
    ----------
    df : DataFrame
        DataFrame pandas à analyser.
    """

    for col in df.columns:
        # Affichage du nom de la variable analysée
        print(f"Analyse univariée pour la variable '{col}':")

        # Vérifie si la colonne est numérique
        if pd.api.types.is_numeric_dtype(df[col]):
            # Si oui, calcul des statistiques descriptives, skewness et kurtosis
            analyse = df[col].describe().to_frame().transpose()
            analyse["skew"] = df[col].skew()
            analyse["kurtosis"] = df[col].kurt()

            # Affichage des statistiques descriptives sous forme de tableau
            print(tabulate(analyse, headers="keys", tablefmt="fancy_grid"))

            # Création de figures pour les graphiques
            plt.figure(figsize=(12, 4))

            # Histogramme avec densité de probabilité
            plt.subplot(1, 2, 1)
            sns.histplot(data=df, x=col, kde=True)
            plt.axvline(
                df[col].mean(),
                color="crimson",
                linestyle="dotted",
                label=f"Moyenne {col}",
            )
            plt.axvline(
                df[col].median(),
                color="black",
                linestyle="dashed",
                label=f"Médiane {col}",
            )
            plt.title(f"Histogramme de {col}")
            plt.legend()

            # Boxplot
            plt.subplot(1, 2, 2)
            sns.boxplot(x=col, data=df, linewidth=3, color="white", showmeans=True)
            plt.title(f"Boxplot de {col}")

            # Affichage des graphiques
            plt.tight_layout()
            plt.show()

        # Vérifie si la colonne est catégorielle
        elif pd.api.types.is_categorical_dtype(df[col]):
            # Si la colonne est catégorielle, calcul des statistiques de base
            analyse = df[col].describe().to_frame().transpose()

            # Affichage des statistiques descriptives sous forme de tableau
            print(tabulate(analyse, headers="keys", tablefmt="fancy_grid"))

            # Calcul des fréquences et pourcentages pour les variables catégorielles
            frequences = df[col].value_counts().to_frame().reset_index()
            frequences.columns = [col, "Count"]
            frequences["Percentage"] = (frequences["Count"] / len(df)) * 100

            # Affichage du tableau de fréquences sous forme de tableau
            print(tabulate(frequences, headers="keys", tablefmt="fancy_grid"))

            # Création de figure pour le barplot
            plt.figure(figsize=(10, 6))

            # Barplot
            sns.countplot(
                data=df, x=col, order=df[col].value_counts().index, palette="Set2"
            )
            plt.title(f"Répartition des catégories de {col}")
            plt.xlabel(col)
            plt.ylabel("Count")

            # Affichage du graphique
            plt.xticks(rotation=45)
            plt.show()

        else:
            # Si la colonne n'est ni numérique ni catégorielle, passer
            print(
                f"La variable '{col}' n'est ni numérique ni catégorielle et n'a pas été analysée."
            )

        # Insértion d'une ligne vide pour séparer les résultats de chaque variable
        print("\n")


def analyse_univariee_par_categorie(df, col_categorie):
    """
    Effectue une analyse univariée sur toutes les variables d'un DataFrame,
    en séparant les données par la catégorie spécifiée.

    Pour les variables numériques, des statistiques descriptives, un histogramme et un boxplot sont générés.
    Pour les variables catégorielles, des statistiques descriptives, un tableau de fréquence et un barplot sont générés.

    Paramètres
    ----------
    df : DataFrame
        DataFrame pandas à analyser.
    col_categorie : str
        Nom de la colonne catégorielle utilisée pour séparer les données.
    """

    # Vérifie que la colonne catégorielle existe dans le DataFrame
    if col_categorie not in df.columns:
        print(f"La colonne '{col_categorie}' n'existe pas dans le DataFrame.")
        return

    # Récupère les catégories uniques
    categories = df[col_categorie].unique()

    # Parcourt chaque catégorie
    for categorie in categories:
        print(f"\n\n=== Analyse pour la catégorie '{categorie}' ===\n")

        # Filtre le DataFrame pour la catégorie actuelle
        df_categorie = df[df[col_categorie] == categorie]

        # Effectue l'analyse univariée sur le sous-DataFrame
        for col in df_categorie.columns:
            # On ignore la colonne catégorielle utilisée pour la séparation
            if col == col_categorie:
                continue

            # Affichage du nom de la variable analysée
            print(f"Analyse univariée pour la variable '{col}':")

            # Vérifie si la colonne est numérique
            if pd.api.types.is_numeric_dtype(df_categorie[col]):
                # Si oui, calcul des statistiques descriptives, skewness et kurtosis
                analyse = df_categorie[col].describe().to_frame().transpose()
                analyse["skew"] = df_categorie[col].skew()
                analyse["kurtosis"] = df_categorie[col].kurt()

                # Affichage des statistiques descriptives sous forme de tableau
                print(tabulate(analyse, headers="keys", tablefmt="fancy_grid"))

                # Création de figures pour les graphiques
                plt.figure(figsize=(12, 4))

                # Histogramme avec densité de probabilité
                plt.subplot(1, 2, 1)
                sns.histplot(data=df_categorie, x=col, kde=True)
                plt.axvline(
                    df_categorie[col].mean(),
                    color="crimson",
                    linestyle="dotted",
                    label=f"Moyenne {col}",
                )
                plt.axvline(
                    df_categorie[col].median(),
                    color="black",
                    linestyle="dashed",
                    label=f"Médiane {col}",
                )
                plt.title(f"Histogramme de {col} pour la catégorie '{categorie}'")
                plt.legend()

                # Boxplot
                plt.subplot(1, 2, 2)
                sns.boxplot(
                    x=col, data=df_categorie, linewidth=3, color="white", showmeans=True
                )
                plt.title(f"Boxplot de {col} pour la catégorie '{categorie}'")

                # Affichage des graphiques
                plt.tight_layout()
                plt.show()

            # Vérifie si la colonne est catégorielle
            elif (
                pd.api.types.is_categorical_dtype(df_categorie[col])
                or df_categorie[col].dtype == object
            ):
                # Si la colonne est catégorielle, calcul des statistiques de base
                analyse = df_categorie[col].describe().to_frame().transpose()

                # Affichage des statistiques descriptives sous forme de tableau
                print(tabulate(analyse, headers="keys", tablefmt="fancy_grid"))

                # Calcul des fréquences et pourcentages pour les variables catégorielles
                frequences = df_categorie[col].value_counts().to_frame().reset_index()
                frequences.columns = [col, "Count"]
                frequences["Percentage"] = (
                    frequences["Count"] / len(df_categorie)
                ) * 100

                # Affichage du tableau de fréquences sous forme de tableau
                print(tabulate(frequences, headers="keys", tablefmt="fancy_grid"))

                # Création de figure pour le barplot
                plt.figure(figsize=(10, 6))

                # Barplot
                sns.countplot(
                    data=df_categorie,
                    x=col,
                    order=df_categorie[col].value_counts().index,
                    palette="Set2",
                )
                plt.title(
                    f"Répartition des catégories de {col} pour la catégorie '{categorie}'"
                )
                plt.xlabel(col)
                plt.ylabel("Count")

                # Affichage du graphique
                plt.xticks(rotation=45)
                plt.show()

            else:
                # Si la colonne n'est ni numérique ni catégorielle, passer
                print(
                    f"La variable '{col}' n'est ni numérique ni catégorielle et n'a pas été analysée."
                )

            # Insertion d'une ligne vide pour séparer les résultats de chaque variable
            print("\n")


def generation_boxplots(df, x_var):
    """
    Cette fonction génère des boxplots pour chaque colonne numérique
    dans le DataFrame (sauf pour celles contenant le mot "cluster"),
    avec la variable `x` spécifiée sur l'axe des x. Les boxplots sont
    disposés en subplots avec deux plots par ligne.

    Arguments:
    df -- DataFrame contenant les données à tracer
    x_var -- Nom de la colonne à utiliser pour l'axe des x
    """
    num_plots = sum("cluster" not in col for col in df.columns)
    num_rows = (num_plots + 1) // 2

    fig, axs = plt.subplots(num_rows, 2, figsize=(20, 6 * num_rows))

    # Aplatir la liste de Axes pour une itération facile
    axs = axs.flatten()

    # Index pour le subplot actuel
    ax_idx = 0

    for col in df.columns:
        if "cluster" in col:  # Pas de boxplot pour les colonnes contenant 'cluster'
            continue

        sns.boxplot(x=x_var, y=col, data=df, ax=axs[ax_idx])

        # La moyenne totale pour la variable courante est calculée
        moyenne_totale = df[col].mean()

        # Une ligne horizontale pointillée rouge est ajoutée pour la moyenne totale
        axs[ax_idx].axhline(y=moyenne_totale, color="r", linestyle="--")

        axs[ax_idx].set_title(f"Boxplot de {col} pour chaque cluster")

        ax_idx += 1

    # Supprimer les subplots inutilisés
    for i in range(ax_idx, len(axs)):
        fig.delaxes(axs[i])

    plt.tight_layout()
    plt.show()


# Créer une colormap discrète personnalisée
def discrete_cmap(n, base_cmap=None):
    """Crée une colormap discrète avec n couleurs."""
    if base_cmap is None:
        base_cmap = plt.cm.get_cmap("Dark2")
    colors = base_cmap(np.linspace(0, 1, n))
    cmap = ListedColormap(colors)
    return cmap


def detecter_valeurs_aberrantes(df):
    """
    Détecte les valeurs aberrantes dans toutes les colonnes numériques d'un DataFrame en utilisant la méthode de l'IQR.

    Paramètres
    ----------
    df : DataFrame
        DataFrame à analyser.
    """
    for col in df.columns:
        # Vérifier si la colonne est numérique
        if np.issubdtype(df[col].dtype, np.number):
            # Calculer l'IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            # Définir les limites
            limite_inf = Q1 - 1.5 * IQR
            limite_sup = Q3 + 1.5 * IQR

            # Identifier les valeurs aberrantes
            valeurs_aberrantes = df[(df[col] < limite_inf) | (df[col] > limite_sup)][
                col
            ]

            # Afficher le tableau des valeurs aberrantes
            if not valeurs_aberrantes.empty:
                print(f"Valeurs aberrantes pour la variable '{col}' :")
                print(
                    tabulate(
                        valeurs_aberrantes.to_frame(),
                        headers="keys",
                        tablefmt="fancy_grid",
                    )
                )
                print("\n")
            else:
                print(f"Aucune valeur aberrante détectée pour la variable '{col}'.\n")
        else:
            print(f"La variable '{col}' n'est pas numérique, elle est ignorée.\n")


def tracer_skus_pour_categorie(
    df,
    categorie,
    nombre_skus=3,
    skus_selectionnes=None,
    moyenne_mobile_heures=300,
    echantillon_reel=10,
):
    """
    Trace les fluctuations de prix pour les SKUs dans une catégorie donnée, avec ajout de la moyenne mobile et d'une ligne horizontale pour la moyenne des prix.
    Paramètres:
        df (pd.DataFrame): Le DataFrame contenant les données des SKUs.
        categorie (str): La catégorie de produit.
        nombre_skus (int): Nombre de SKUs à tracer (utilisé si skus_selectionnes est None).
        skus_selectionnes (list): Liste des SKUs sélectionnés manuellement à tracer.
        moyenne_mobile_heures (int): Le nombre d'observations (heures) pour calculer la moyenne mobile.
        echantillon_reel (int): Intervalle d'échantillonnage pour la trace réelle.
    """
    if skus_selectionnes is None:
        # Sélectionner les SKUs aléatoirement si non sélectionnés manuellement
        skus = (
            df[df["Categorie"] == categorie]["SKU"]
            .drop_duplicates()
            .sample(nombre_skus)
        )
    else:
        # Utiliser les SKUs sélectionnés manuellement
        skus = skus_selectionnes

    plt.figure(figsize=(14, 10))

    for sku in skus:
        sku_data = df[df["SKU"] == sku].sort_values("Timestamp")

        # Trace réelle (échantillonnée)
        sampled_data = sku_data.iloc[::echantillon_reel]
        sns.lineplot(
            data=sampled_data,
            x="Timestamp",
            y="Prix",
            label=f"{sku} Prix Réel",
            marker="o",
            linestyle="--",
            markersize=4,
            linewidth=0.8,
        )

        # Calcul de la moyenne mobile
        sku_data = sku_data.set_index("Timestamp")
        sku_data[f"Moyenne Mobile {moyenne_mobile_heures} Heures"] = (
            sku_data["Prix"].rolling(window=moyenne_mobile_heures, min_periods=1).mean()
        )

        # Trace de la moyenne mobile
        sns.lineplot(
            data=sku_data.reset_index(),  # Réinitialiser l'index pour faciliter le tracé
            x="Timestamp",
            y=f"Moyenne Mobile {moyenne_mobile_heures} Heures",
            label=f"{sku} Moyenne Mobile",
            linewidth=1,
        )

    plt.xlabel("Timestamp")
    plt.ylabel("Prix")
    plt.title(f"Évolution du prix pour les SKUs dans la catégorie {categorie}")
    plt.legend()
    plt.show()


# def tracer_skus_pour_categorie(
#     df,
#     categorie,
#     nombre_skus=3,
#     skus_selectionnes=None,
#     moyenne_mobile_heures=2160,
#     echantillon_reel=10,
# ):
#     """
#     Trace les fluctuations de prix pour les SKUs dans une catégorie donnée, avec ajout de la moyenne mobile et d'une ligne horizontale pour la moyenne des prix.
#     Paramètres:
#         df (pd.DataFrame): Le DataFrame contenant les données des SKUs.
#         categorie (str): La catégorie de produit.
#         nombre_skus (int): Nombre de SKUs à tracer (utilisé si skus_selectionnes est None).
#         skus_selectionnes (list): Liste des SKUs sélectionnés manuellement à tracer.
#         moyenne_mobile_heures (int): Le nombre d'observations (heures) pour calculer la moyenne mobile.
#         echantillon_reel (int): Intervalle d'échantillonnage pour le tracé réelle.
#     """
#     if skus_selectionnes is None:
#         # Sélectionner les SKUs aléatoirement si non sélectionnés manuellement
#         skus = (
#             df[df["Categorie"] == categorie]["SKU"]
#             .drop_duplicates()
#             .sample(nombre_skus)
#         )
#     else:
#         # Utiliser les SKUs sélectionnés manuellement
#         skus = skus_selectionnes

#     plt.figure(figsize=(14, 10))

#     for sku in skus:
#         sku_data = df[df["SKU"] == sku].sort_values("Timestamp")

#         # Tracé (échantillonnée)
#         sampled_data = sku_data.iloc[::echantillon_reel]
#         sns.lineplot(
#             data=sampled_data,
#             x="Timestamp",
#             y="Prix",
#             label=f"{sku} Prix",
#             marker="o",
#             linestyle="--",
#             markersize=4,
#             linewidth=0.8,
#         )

#         # Calcul de la moyenne mobile
#         sku_data = sku_data.set_index("Timestamp")
#         sku_data[f"Moyenne Mobile {moyenne_mobile_heures} Heures"] = (
#             sku_data["Prix"].rolling(window=moyenne_mobile_heures, min_periods=1).mean()
#         )
#         sku_data.reset_index(inplace=True)  # Revenir à l'index par défaut pour tracer

#         # Trace de la moyenne mobile
#         sns.lineplot(
#             data=sku_data,
#             x="Timestamp",
#             y=f"Moyenne Mobile {moyenne_mobile_heures} Heures",
#             label=f"{sku} Moyenne Mobile",
#             linewidth=2,
#         )

#     plt.xlabel("Timestamp")
#     plt.ylabel("Prix")
#     plt.title(f"Évolution du prix pour les SKUs dans la catégorie {categorie}")
#     plt.legend()
#     plt.show()


# def tracer_skus_pour_categorie(
#     df, categorie, nombre_skus=3, skus_selectionnes=None, moyenne_mobile_mois=3
# ):
#     """
#     Trace les fluctuations de prix pour les SKUs dans une catégorie donnée, avec ajout de la moyenne mobile et d'une ligne horizontale pour la moyenne des prix.

#     Paramètres:
#         df (pd.DataFrame): Le DataFrame contenant les données des SKUs.
#         categorie (str): La catégorie de produit.
#         nombre_skus (int): Nombre de SKUs à tracer (utilisé si skus_selectionnes est None).
#         skus_selectionnes (list): Liste des SKUs sélectionnés manuellement à tracer.
#         moyenne_mobile_mois (int): Le nombre de mois pour calculer la moyenne mobile.
#     """
#     if skus_selectionnes is None:
#         # Sélectionner les SKUs aléatoirement si non sélectionnés manuellement
#         skus = (
#             df[df["Categorie"] == categorie]["SKU"]
#             .drop_duplicates()
#             .sample(nombre_skus)
#         )
#     else:
#         # Utiliser les SKUs sélectionnés manuellement
#         skus = skus_selectionnes

#     plt.figure(figsize=(14, 10))

#     for sku in skus:
#         sku_data = df[df["SKU"] == sku]
#         sns.lineplot(data=sku_data, x="Date", y="Prix", label=f"{sku} Prix", marker="o")

#         # Calcul de la moyenne mobile
#         sku_data = sku_data.set_index("Date")
#         sku_data[f"Moyenne Mobile {moyenne_mobile_mois} Mois"] = (
#             sku_data["Prix"].rolling(window=moyenne_mobile_mois).mean()
#         )
#         sns.lineplot(
#             data=sku_data,
#             x=sku_data.index,
#             y=f"Moyenne Mobile {moyenne_mobile_mois} Mois",
#             label=f"{sku} Moyenne Mobile",
#         )

#         # # Calcul de la moyenne des prix
#         # moyenne_prix = sku_data["Prix"].mean()
#         # plt.axhline(
#         #     y=moyenne_prix,
#         #     color="r",
#         #     linestyle="--",
#         #     label=f"{sku} Moyenne: {moyenne_prix:.2f}",
#         # )

#     plt.xlabel("Date")
#     plt.ylabel("Prix")
#     plt.title(f"Évolution du prix pour les SKUs dans la catégorie {categorie}")
#     plt.legend()
#     plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import shapiro, kstest, normaltest


class RegressionDiagnostics:
    def __init__(self, model, X, y):
        """
        Initialise la classe avec le modèle de régression, les variables indépendantes (X) et la variable dépendante (y).
        """
        self.model = model
        self.X = X
        self.y = y
        self.residuals = y - model.predict(X)

        # Identifier les variables encodées
        self.encoded_vars = [col for col in X.columns if "_" in col]
        self.X_no_encoded = X.drop(columns=self.encoded_vars)

    def linearity_test(self):
        """
        Teste la linéarité entre les variables indépendantes et la variable dépendante.
        """
        plt.figure(figsize=(10, 6))
        for i, col in enumerate(self.X_no_encoded.columns):
            plt.subplot(len(self.X_no_encoded.columns) // 2 + 1, 2, i + 1)
            sns.scatterplot(x=self.X_no_encoded[col], y=self.residuals)
            plt.xlabel(col)
            plt.ylabel("Résidus")
            plt.title(f"Linéarité de {col}")
        plt.tight_layout()
        plt.show()

    def independence_test(self):
        """
        Teste l'indépendance des erreurs (résidus) en traçant les résidus en fonction du temps ou de l'ordre des observations.
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=np.arange(len(self.residuals)), y=self.residuals)
        plt.xlabel("Index des observations")
        plt.ylabel("Résidus")
        plt.title("Indépendance des erreurs")
        plt.show()

    def homoscedasticity_test(self):
        """
        Teste l'homoscédasticité des résidus.
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.model.predict(self.X), y=self.residuals)
        plt.xlabel("Valeurs prédites")
        plt.ylabel("Résidus")
        plt.title("Homoscedasticité")
        plt.show()

    def normality_test(self):
        """
        Teste la normalité des résidus en utilisant un Q-Q plot et des tests statistiques.
        """
        # Q-Q plot
        plt.figure(figsize=(10, 6))
        sm.qqplot(self.residuals, line="s")
        plt.title("Normalité des résidus (Q-Q plot)")
        plt.show()

        # Shapiro-Wilk test
        stat, p_value = shapiro(self.residuals)
        print(f"Shapiro-Wilk test: p-value={p_value}")

        # Kolmogorov-Smirnov test
        stat, p_value = kstest(self.residuals, "norm")
        print(f"Kolmogorov-Smirnov test: p-value={p_value}")

        # D'Agostino's K-squared test
        stat, p_value = normaltest(self.residuals)
        print(f"D'Agostino's K-squared test: p-value={p_value}")

    def multicollinearity_test(self):
        """
        Teste la multicolinéarité des variables indépendantes en calculant le VIF (Variance Inflation Factor).
        """
        vif_data = pd.DataFrame()
        vif_data["feature"] = self.X_no_encoded.columns
        vif_data["VIF"] = [
            variance_inflation_factor(self.X_no_encoded.values, i)
            for i in range(len(self.X_no_encoded.columns))
        ]
        print(vif_data)

    def run_all_diagnostics(self):
        """
        Exécute tous les tests de diagnostic.
        """
        print("Test de linéarité")
        self.linearity_test()
        print("Test d'indépendance des erreurs")
        self.independence_test()
        print("Test d'homoscédasticité")
        self.homoscedasticity_test()
        print("Test de normalité des erreurs")
        self.normality_test()
        print("Test de multicolinéarité")
        self.multicollinearity_test()
