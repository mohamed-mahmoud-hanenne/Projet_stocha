import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Chargement des données
df = pd.read_csv("microfinance_data_fuzzy_with_history.csv")

# Encodage de la variable catégorielle "Statut_Emploi"
df["Statut_Emploi"] = df["Statut_Emploi"].astype("category").cat.codes

# Normalisation des données continues **(Inclure Statut_Emploi maintenant)**
scaler = MinMaxScaler()
df[["Revenu_Mensuel", "Montant_Credit", "Duree_Credit", "Age", "Statut_Emploi",
    "Nbr_Total_Credits", "Nbr_Credits_Payes", "Nbr_Credits_NonPayes"]] = scaler.fit_transform(
    df[["Revenu_Mensuel", "Montant_Credit", "Duree_Credit", "Age", "Statut_Emploi",
        "Nbr_Total_Credits", "Nbr_Credits_Payes", "Nbr_Credits_NonPayes"]])

# Sauvegarde du scaler pour l'utiliser dans l'API Flask
import joblib
joblib.dump(scaler, "scaler.pkl")

# Définition des paramètres du Q-learning
num_states = df.shape[1] - 2  # Toutes les colonnes sauf Defaut_Paiement et ID_Client
num_actions = 2  # Accepter (1) ou refuser (0) le crédit
alpha = 0.1  # Taux d'apprentissage
gamma = 0.9  # Facteur de réduction
epsilon = 0.1  # Facteur d'exploration

# Initialisation de la table Q
q_table = np.zeros((len(df), num_actions))

# Entraînement du modèle Q-learning
num_episodes = 2000  # Augmenté pour un meilleur apprentissage

data_values = df.drop(columns=["ID_Client", "Defaut_Paiement"]).values
target_values = df["Defaut_Paiement"].values

for episode in range(num_episodes):
    indices = np.random.permutation(len(df))  # Mélange des indices
    for i in indices:
        state = data_values[i]
        action = np.argmax(q_table[i]) if np.random.rand() > epsilon else np.random.choice(num_actions)

        # Déterminer la récompense
        if action == 1 and target_values[i] == 0:
            reward = 10  # Bonne décision d'accorder le crédit
        elif action == 1 and target_values[i] == 1:
            reward = -10  # Mauvaise décision
        elif action == 0:
            reward = -1  # Petite pénalité pour refus

        # Mise à jour de la table Q
        next_index = np.random.choice(len(df))  # Sélection d'un état suivant aléatoire
        next_max = np.max(q_table[next_index])
        q_table[i, action] += alpha * (reward + gamma * next_max - q_table[i, action])

# Sauvegarde du modèle Q-learning
np.save("q_table_trained.npy", q_table)
print("Modèle Q-learning entraîné et sauvegardé !")
