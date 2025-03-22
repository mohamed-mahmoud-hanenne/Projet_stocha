import pandas as pd
import numpy as np

# Fonction de logique floue pour calculer Historique_Credit
def calcul_historique_credit_fuzzy(nbr_payes, nbr_total):
    if nbr_total == 0:
        return 0  # Pas d'historique si aucun crédit souscrit

    ratio = nbr_payes / nbr_total

    if ratio <= 0.4:
        return 0  # Mauvais historique
    elif ratio >= 0.8:
        return 1  # Bon historique
    else:
        # Interpolation linéaire entre 0.4 et 0.8
        return (ratio - 0.4) / (0.8 - 0.4)  # Retourne une valeur entre 0 et 1 pour historique moyen

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
df["Historique_Credit"] = df.apply(lambda row: calcul_historique_credit_fuzzy(row["Nbr_Credits_Payes"], row["Nbr_Total_Credits"]), axis=1)

# Définition des pondérations relatives
poids = {
    "Historique_Credit": 0.40,
    "Statut_Emploi": 0.20,
    "Revenu_Mensuel": 0.15,
    "Montant_Credit": 0.10,
    "Duree_Credit": 0.10,
    "Age": 0.05
}

# Normalisation des valeurs pour les pondérations
df["Ponderation_Historique_Credit"] = df["Historique_Credit"] * poids["Historique_Credit"]
df["Ponderation_Statut_Emploi"] = df["Statut_Emploi"].apply(lambda x: 1 if x == "CDI" else (0.7 if x == "CDD" else (0.5 if x == "Indépendant" else 0))) * poids["Statut_Emploi"]
df["Ponderation_Revenu"] = (df["Revenu_Mensuel"] - df["Revenu_Mensuel"].min()) / (df["Revenu_Mensuel"].max() - df["Revenu_Mensuel"].min()) * poids["Revenu_Mensuel"]
df["Ponderation_Montant_Credit"] = (1 - (df["Montant_Credit"] - df["Montant_Credit"].min()) / (df["Montant_Credit"].max() - df["Montant_Credit"].min())) * poids["Montant_Credit"]
df["Ponderation_Duree_Credit"] = (df["Duree_Credit"] - df["Duree_Credit"].min()) / (df["Duree_Credit"].max() - df["Duree_Credit"].min()) * poids["Duree_Credit"]
df["Ponderation_Age"] = (df["Age"] - df["Age"].min()) / (df["Age"].max() - df["Age"].min()) * poids["Age"]

# Calcul du Score Global
df["Score_Global"] = df[[
    "Ponderation_Historique_Credit",
    "Ponderation_Statut_Emploi",
    "Ponderation_Revenu",
    "Ponderation_Montant_Credit",
    "Ponderation_Duree_Credit",
    "Ponderation_Age"
]].sum(axis=1)

# Définition de la décision finale
df["Decision_Credit"] = df["Score_Global"].apply(lambda x: "Accepté" if x >= 0.5 else "Refusé")

# Sauvegarde en fichier CSV
file_path = "microfinance_data_fuzzy_with_weights_test.csv"
df.to_csv(file_path, index=False)

print(f"Fichier généré : {file_path}")
