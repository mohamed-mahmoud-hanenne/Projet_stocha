import pandas as pd
import numpy as np
import pickle

# Chargement des données
file_path = "microfinance_data_fuzzy_with_weights_test.csv"
df = pd.read_csv(file_path)

# Définition des états et actions
states = df[['Age', 'Revenu_Mensuel', 'Montant_Credit', 'Duree_Credit', 'Historique_Credit']].values
actions = ["Accepté", "Refusé"]

# Initialisation de la table Q
Q_table = np.zeros((len(states), len(actions)))

# Paramètres du Q-learning
alpha = 0.1  # Taux d'apprentissage
gamma = 0.9  # Facteur de récompense
epsilon = 0.1  # Probabilité d'exploration

# Fonction de récompense
def reward_function(decision, score_global):
    if decision == "Accepté" and score_global >= 0.5:
        return 1  # Bonne décision
    elif decision == "Refusé" and score_global < 0.5:
        return 1  # Bonne décision
    else:
        return -1  # Mauvaise décision

# Entraînement du modèle
num_episodes = 1000
for episode in range(num_episodes):
    for i, row in df.iterrows():
        state_idx = i  # Index de l'état dans la table
        if np.random.rand() < epsilon:
            action_idx = np.random.choice(len(actions))  # Exploration
        else:
            action_idx = np.argmax(Q_table[state_idx])  # Exploitation
        
        action = actions[action_idx]
        reward = reward_function(action, row['Score_Global'])
        
        # Mise à jour de la table Q
        Q_table[state_idx, action_idx] = (1 - alpha) * Q_table[state_idx, action_idx] + \
                                         alpha * (reward + gamma * np.max(Q_table[state_idx]))

# Sauvegarde du modèle entraîné
with open("qlearning_model.pkl", "wb") as f:
    pickle.dump(Q_table, f)

print("Modèle Q-learning entraîné et sauvegardé.")
