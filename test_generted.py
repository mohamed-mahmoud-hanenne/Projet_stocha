import pandas as pd
import numpy as np

# Fonction de logique floue pour calculer Historique_Credit
def calcul_historique_credit(nbr_payes, nbr_total):
    if nbr_total == 0:
        return 0  # Si aucun crédit souscrit, considérer comme mauvais historique
    ratio = nbr_payes / nbr_total

    # Définition des ensembles flous
    if ratio <= 0.4:
        return 0  # Mauvais historique
    elif 0.4 < ratio <= 0.7:
        return np.random.choice([0, 1], p=[0.6, 0.4])  # Transition (tendance vers mauvais)
    else:
        return 1  # Bon historique

# Génération de données simulées
np.random.seed(42)
n_samples = 1000

data = {
    "ID_Client": range(1, n_samples + 1),
    "Revenu_Mensuel": np.random.randint(50000, 500000, n_samples),
    "Montant_Credit": np.random.randint(100000, 2000000, n_samples),
    "Duree_Credit": np.random.randint(6, 60, n_samples),
    "Age": np.random.randint(18, 65, n_samples),
    "Statut_Emploi": np.random.choice(["CDI", "CDD", "Indépendant", "Sans emploi"], n_samples),
    "Nbr_Total_Credits": np.random.randint(1, 10, n_samples),
}

# Création du DataFrame
df = pd.DataFrame(data)

# Ajout des nouvelles colonnes
df["Nbr_Credits_Payes"] = np.random.randint(0, df["Nbr_Total_Credits"] + 1, n_samples)
df["Nbr_Credits_NonPayes"] = df["Nbr_Total_Credits"] - df["Nbr_Credits_Payes"]

# Application de la logique floue pour Historique_Credit
df["Historique_Credit"] = df.apply(lambda row: calcul_historique_credit(row["Nbr_Credits_Payes"], row["Nbr_Total_Credits"]), axis=1)

# Définition du défaut de paiement (0 = pas de défaut, 1 = défaut)
df["Defaut_Paiement"] = np.where(df["Historique_Credit"] == 0, 1, 0)

# Sauvegarde en fichier CSV
file_path = "microfinance_data_fuzzy_with_history.csv"
df.to_csv(file_path, index=False)

print(f"Fichier généré : {file_path}")
