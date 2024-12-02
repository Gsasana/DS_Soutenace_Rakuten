from modules.modelisation import *

# Exemple d'utilisation
if __name__ == "__main__":
    params = ParametresSynthese()
    synthese = SyntheseDonnees(params)
    donnees_df = synthese.generer_donnees()
    donnees_df.to_csv("donnees_synthetiques.csv", index=False)
    print(donnees_df.head())

    # Optionnel : Analyse des données générées
    # synthese.analyser_donnees(donnees_df)
