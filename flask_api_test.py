from flask import Flask, request, jsonify
import numpy as np
import pickle
import pandas as pd
from flask_cors import CORS


# Charger le modèle Q-learning
with open("qlearning_model.pkl", "rb") as f:
    Q_table = pickle.load(f)

# Charger les données de référence
file_path = "microfinance_data_fuzzy_with_weights_test.csv"
df = pd.read_csv(file_path)

# Définir l'application Flask
app = Flask(__name__)
CORS(app)  # Activer CORS

# Définition des actions possibles
actions = ["Accepté", "Refusé"]

# Fonction pour calculer l'historique de crédit
def calcul_historique_credit_fuzzy(nbr_payes, nbr_total):
    if nbr_total == 0:
        return 0
    ratio = nbr_payes / nbr_total
    if ratio <= 0.4:
        return 0
    elif ratio >= 0.8:
        return 1
    else:
        return (ratio - 0.4) / (0.8 - 0.4)

# Route API pour prédire la décision de crédit
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    
    # Extraction des données client
    age = data["Age"]
    revenu = data["Revenu_Mensuel"]
    montant_credit = data["Montant_Credit"]
    duree_credit = data["Duree_Credit"]
    statut_emploi = data["Statut_Emploi"]
    nbr_total_credits = data["Nbr_Total_Credits"]
    nbr_credits_payes = data["Nbr_Credits_Payes"]
    nbr_credits_nonpayes = data["Nbr_Credits_NonPayes"]
    
    # Calcul de l'historique de crédit
    historique_credit = calcul_historique_credit_fuzzy(nbr_credits_payes, nbr_total_credits)
    
    # Normalisation des valeurs
    poids = {
        "Historique_Credit": 0.40,
        "Statut_Emploi": 0.20,
        "Revenu_Mensuel": 0.15,
        "Montant_Credit": 0.10,
        "Duree_Credit": 0.10,
        "Age": 0.05
    }
    
    ponderation_historique = historique_credit * poids["Historique_Credit"]
    ponderation_statut = 1 if statut_emploi == "CDI" else (0.7 if statut_emploi == "CDD" else (0.5 if statut_emploi == "Indépendant" else 0))
    ponderation_statut *= poids["Statut_Emploi"]
    ponderation_revenu = (revenu - df["Revenu_Mensuel"].min()) / (df["Revenu_Mensuel"].max() - df["Revenu_Mensuel"].min()) * poids["Revenu_Mensuel"]
    ponderation_montant = (1 - (montant_credit - df["Montant_Credit"].min()) / (df["Montant_Credit"].max() - df["Montant_Credit"].min())) * poids["Montant_Credit"]
    ponderation_duree = (duree_credit - df["Duree_Credit"].min()) / (df["Duree_Credit"].max() - df["Duree_Credit"].min()) * poids["Duree_Credit"]
    ponderation_age = (age - df["Age"].min()) / (df["Age"].max() - df["Age"].min()) * poids["Age"]
    
    # Calcul du score global
    score_global = sum([ponderation_historique, ponderation_statut, ponderation_revenu,
                        ponderation_montant, ponderation_duree, ponderation_age])
    
    # # Prédiction avec Q-learning
    # state = np.array([age, revenu, montant_credit, duree_credit, historique_credit])
    # state_idx = np.argmin(np.sum((df[['Age', 'Revenu_Mensuel', 'Montant_Credit', 'Duree_Credit', 'Historique_Credit']].values - state) ** 2, axis=1))
    # action_idx = np.argmax(Q_table[state_idx])
    # decision = actions[action_idx]
    

     # Prise de décision basée sur le score global
    if score_global < 0.5:
        decision = "Refusé"
    else:
        decision = "Accepté"

        
    return jsonify({
        "Historique_Credit": historique_credit,
        "Score_Global": score_global,
        "Decision_Credit": decision
    })

if __name__ == "__main__":
    app.run(debug=True)
