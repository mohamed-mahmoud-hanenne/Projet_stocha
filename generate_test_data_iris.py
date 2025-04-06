import pandas as pd
import numpy as np

# Paramètres pour générer des données réalistes (basées sur les statistiques de l'ensemble Iris)
species_params = {
    "setosa": {"petal_length": (1.0, 2.0), "petal_width": (0.1, 0.6)},
    "versicolor": {"petal_length": (3.0, 5.0), "petal_width": (1.0, 1.8)},
    "virginica": {"petal_length": (4.5, 7.0), "petal_width": (1.5, 2.5)}
}

# Nombre d'échantillons par espèce (peut être modifié)
n_samples = 20

# Génération des données
data = []
for species, params in species_params.items():
    petal_lengths = np.random.uniform(*params["petal_length"], n_samples)
    petal_widths = np.random.uniform(*params["petal_width"], n_samples)
    for pl, pw in zip(petal_lengths, petal_widths):
        data.append([pl, pw, species])

# Création du DataFrame
columns = ["petal_length", "petal_width", "species"]
df = pd.DataFrame(data, columns=columns)

# Mélanger les données (optionnel)
df = df.sample(frac=1).reset_index(drop=True)

# Sauvegarde en CSV
df.to_csv("iris_test.csv", index=False)
print("Fichier 'iris_test.csv' généré avec succès!")
print(f"Dimensions: {df.shape[0]} lignes, {df.shape[1]} colonnes")

# Aperçu des données générées
print("\nAperçu des données générées:")
print(df.head())