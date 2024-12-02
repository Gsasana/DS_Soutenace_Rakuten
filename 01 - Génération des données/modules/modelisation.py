# Module : modelisation.py

import logging
from collections import defaultdict
from typing import Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from jax import random
from scipy.stats import truncnorm
from tqdm import tqdm


class ParametresSynthese:
    """Pour stocker les paramètres de configuration de la synthèse des données."""

    def __init__(self):
        # Paramètres généraux
        self.seed = 17
        self.random_seed_range = 10000

        # Paramètres des catégories et produits
        self.parametres_prix_par_categorie = {
            "Electronique": {
                "prix_moyen": 450,
                "prix_ecart_type": 50,
                "plancher_pct": 0.4,
                "delta": 0.01,
                "gamma": 0.0015,
                "demande_lambda": 3,
                "remise_prob": 0.05,
            },
            "Livres": {
                "prix_moyen": 25,
                "prix_ecart_type": 5,
                "plancher_pct": 0.2,
                "delta": 0.005,
                "gamma": 0.0005,
                "demande_lambda": 7,
                "remise_prob": 0.15,
            },
            "Vetements": {
                "prix_moyen": 50,
                "prix_ecart_type": 10,
                "plancher_pct": 0.5,
                "delta": 0.02,
                "gamma": 0.001,
                "demande_lambda": 5,
                "remise_prob": 0.1,
            },
        }
        self.n_categories = len(self.parametres_prix_par_categorie)
        self.n_skus = 60  # Nombre total de produits (SKUs)
        self.n_periodes = 600  # Nombre de jours pour lesquels générer des données

        # Coefficients pour les modèles
        self.beta_prix = -abs(np.random.normal(0.05, 0.005))
        self.beta_qualite = abs(np.random.normal(0.05, 0.005))
        self.beta_promo = abs(np.random.normal(0.03, 0.002))
        self.erreur_std = 1

        # Paramètres pour les remises
        self.remise_valeur = 0.1

        # Prix minimum
        self.prix_minimum = 0.01


class SyntheseDonnees:
    """Classe pour générer des données synthétiques basées sur les paramètres donnés."""

    def __init__(self, params: ParametresSynthese):
        """Initialise le générateur de données avec les paramètres spécifiés."""
        self.params = params
        np.random.seed(self.params.seed)
        self.random_seed_generator = np.random.default_rng(self.params.seed)
        self.random_seed = random.PRNGKey(self.params.seed)

        # Génération des produits et des catégories
        self.produits_df = self.generer_produits()
        self.dates_df = self.generer_dates()

    def _get_random_seed(self) -> jnp.ndarray:
        """Génère une clé aléatoire pour JAX."""
        seed = self.random_seed_generator.integers(
            low=0, high=self.params.random_seed_range
        )
        seed = random.PRNGKey(seed)
        return seed

    def generer_produits(self) -> pd.DataFrame:
        """Génère un DataFrame contenant les produits avec leurs attributs."""
        skus = []
        categories = []
        prix_initials = []
        qualites = []
        plancher_pourcentages = []

        categories_list = list(self.params.parametres_prix_par_categorie.keys())
        total_categories = len(categories_list)

        # Répartition dynamique des SKUs entre les catégories
        base_skus_par_categorie = self.params.n_skus // total_categories
        skus_reste = self.params.n_skus % total_categories
        skus_par_categorie = [base_skus_par_categorie] * total_categories
        for i in range(skus_reste):
            skus_par_categorie[i] += 1  # Distribution des SKUs restantes

        for idx, categorie_nom in enumerate(categories_list):
            param_categorie = self.params.parametres_prix_par_categorie[categorie_nom]
            for sku_id in range(1, skus_par_categorie[idx] + 1):
                sku = f"SKU{idx + 1}_{sku_id}"
                skus.append(sku)
                categories.append(categorie_nom)
                # Génération du prix initial avec une distribution normale tronquée
                prix_moyen = param_categorie["prix_moyen"]
                prix_ecart_type = param_categorie["prix_ecart_type"]
                a, b = (0 - prix_moyen) / prix_ecart_type, np.inf
                prix_initial = truncnorm.rvs(
                    a, b, loc=prix_moyen, scale=prix_ecart_type
                )
                prix_initials.append(prix_initial)
                qualite = np.random.uniform(0, 1)
                qualites.append(qualite)
                plancher_pourcentages.append(param_categorie["plancher_pct"])

        produits_df = pd.DataFrame(
            {
                "SKU": skus,
                "Categorie": categories,
                "PrixInitial": prix_initials,
                "Qualite": qualites,
                "PlancherPourcentage": plancher_pourcentages,
            }
        )

        return produits_df

    def generer_dates(self) -> pd.DataFrame:
        """Génère une série temporelle avec 3 timestamps par jour."""
        dates = pd.date_range(
            start="2023-01-01", periods=self.params.n_periodes, freq="D"
        )
        timestamps = []
        intervals_per_day = 3  # Nombre d'observations par jour
        hours_between = (
            24 // intervals_per_day
        )  # Espacement horaire entre les observations

        for date in dates:
            for interval in range(intervals_per_day):
                timestamps.append(date + pd.Timedelta(hours=interval * hours_between))

        timestamps_df = pd.DataFrame({"Timestamp": timestamps})
        return timestamps_df

    def ajuster_prix(self) -> pd.DataFrame:
        """Génère les prix ajustés des produits en fonction du temps, avec 3 observations par jour."""
        prix_liste = []

        for _, produit in self.produits_df.iterrows():
            param_categorie = self.params.parametres_prix_par_categorie[
                produit["Categorie"]
            ]
            gamma = param_categorie["gamma"]
            prix_plancher = produit["PrixInitial"] * param_categorie["plancher_pct"]
            remise_prob = param_categorie.get("remise_prob", self.params.remise_valeur)

            for _, timestamp_row in self.dates_df.iterrows():
                timestamp = timestamp_row["Timestamp"]
                age_produit = (
                    timestamp.normalize() - pd.to_datetime("2023-01-01")
                ).days

                # Dépréciation du prix en fonction de l'âge du produit
                depreciation = np.exp(-gamma * age_produit)
                prix = produit["PrixInitial"] * depreciation

                # Application de la remise
                remise = np.random.binomial(1, remise_prob) * self.params.remise_valeur
                prix_apres_remise = prix * (1 - remise)

                # Ajout d'une variation intra-journalière (bruit gaussien léger)
                variation_intra_jour = np.random.normal(0, self.params.erreur_std / 2)
                prix_final = prix_apres_remise + variation_intra_jour

                # Application du prix plancher spécifique à la catégorie
                prix_final = max(prix_final, prix_plancher)

                # Promotion binaire
                promotion = 1 if remise > 0 else 0

                prix_liste.append(
                    {
                        "SKU": produit["SKU"],
                        "Categorie": produit["Categorie"],
                        "Timestamp": timestamp,
                        "Date": timestamp.normalize(),
                        "Prix": prix_final,
                        "PrixInitial": produit["PrixInitial"],
                        "Remise": remise,
                        "ErreurAleatoire": variation_intra_jour,
                        "Promotion": promotion,
                        "Qualite": produit["Qualite"],
                        "AgeProduitEnJours": age_produit,
                        "PlancherPourcentage": param_categorie["plancher_pct"],
                        "PrixPlancher": prix_plancher,
                    }
                )

        prix_df = pd.DataFrame(prix_liste)
        return prix_df

    def calculer_utilite(self, prix_df: pd.DataFrame) -> pd.DataFrame:
        """Calcule l'utilité du produit pour chaque observation."""
        # Fusion avec les paramètres de dépréciation par catégorie
        prix_df["Delta"] = prix_df["Categorie"].map(
            lambda x: self.params.parametres_prix_par_categorie[x]["delta"]
        )
        # Erreur aléatoire sur l'utilité
        erreur_utilite = np.random.normal(0, self.params.erreur_std, size=len(prix_df))
        prix_df["UtiliteProduit"] = (
            self.params.beta_prix * prix_df["Prix"]
            + self.params.beta_qualite * prix_df["Qualite"]
            + self.params.beta_promo * prix_df["Promotion"]
            - prix_df["Delta"] * prix_df["AgeProduitEnJours"]
            + erreur_utilite
        )
        return prix_df

    def calculer_probabilite_achat(self, prix_df: pd.DataFrame) -> pd.DataFrame:
        """Calcule la probabilité d'achat pour chaque observation."""
        prix_df["ProbabiliteAchat"] = 1 / (1 + np.exp(-prix_df["UtiliteProduit"]))
        return prix_df

    def simuler_quantite_vendue(self, prix_df: pd.DataFrame) -> pd.DataFrame:
        """Simule la quantité vendue pour chaque observation."""
        prix_df["DemandeLambda"] = prix_df["Categorie"].map(
            lambda x: self.params.parametres_prix_par_categorie[x]["demande_lambda"]
        )
        prix_df["QuantiteVendue"] = np.random.poisson(
            lam=prix_df["ProbabiliteAchat"] * prix_df["DemandeLambda"]
        )
        return prix_df

    def calculer_elasticite(self, prix_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule une élasticité prix réaliste pour chaque observation.

        L'élasticité est définie comme une mesure de la sensibilité de la demande
        aux variations de prix, avec des valeurs contraintes entre 0 et 1.

        Paramètres
        ----------
        prix_df : pd.DataFrame
            DataFrame contenant les colonnes nécessaires pour le calcul :
            - 'ProbabiliteAchat' : Probabilité d'achat.
            - 'Prix' : Prix du produit.

        Retourne
        -------
        pd.DataFrame
            DataFrame avec une colonne ajoutée 'ElasticitePrix'.
        """
        if "ProbabiliteAchat" not in prix_df.columns or "Prix" not in prix_df.columns:
            raise ValueError(
                "Les colonnes 'ProbabiliteAchat' et 'Prix' doivent être présentes dans le DataFrame."
            )

        # Calcul de l'élasticité simplifiée avec contrainte [0, 1]
        prix_df["ElasticitePrix"] = abs(self.params.beta_prix) * (
            1 - prix_df["ProbabiliteAchat"]
        )

        # Contraindre les valeurs entre 0 et 1
        prix_df["ElasticitePrix"] = prix_df["ElasticitePrix"].clip(lower=0, upper=1)

        return prix_df

    def generer_donnees(self) -> pd.DataFrame:
        """Génère l'ensemble des données simulées."""
        prix_df = self.ajuster_prix()
        prix_df = self.calculer_utilite(prix_df)
        prix_df = self.calculer_probabilite_achat(prix_df)
        prix_df = self.simuler_quantite_vendue(prix_df)
        prix_df = self.calculer_elasticite(prix_df)
        prix_df["DateLancement"] = pd.to_datetime("2023-01-01")
        colonnes = [
            "Categorie",
            "SKU",
            "PrixInitial",
            "DateLancement",
            "AgeProduitEnJours",
            "Date",
            "Timestamp",
            "Prix",
            "PrixPlancher",
            "PlancherPourcentage",
            "Promotion",
            "QuantiteVendue",
            "ProbabiliteAchat",
            "UtiliteProduit",
            "ElasticitePrix",
            "Remise",
            "ErreurAleatoire",
            "Qualite",
        ]
        donnees_df = prix_df[colonnes]
        return donnees_df

    def analyser_donnees(self, donnees_df: pd.DataFrame):
        """Analyse statistique des données générées (optionnel)."""
        # Par exemple, afficher les statistiques descriptives par catégorie
        print(donnees_df.groupby("Categorie")["Prix"].describe())

        # Visualiser la distribution des prix
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.histplot(data=donnees_df, x="Prix", hue="Categorie", kde=True)
        plt.title("Distribution des Prix par Catégorie")
        plt.show()

        # Visualiser la relation entre le prix et la quantité vendue
        sns.scatterplot(
            data=donnees_df,
            x="Prix",
            y="QuantiteVendue",
            hue="Categorie",
            alpha=0.7,
        )
        plt.title("Relation entre le Prix et la Quantité Vendue")
        plt.show()
