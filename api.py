from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

app = Flask(__name__)
CORS(app)  # Activer CORS

# Charger le modèle Q-learning entraîné
q_table = np.load("q_table_trained.npy")

# Charger le scaler sauvegardé
scaler = joblib.load("scaler.pkl")

# Charger les données historiques
df = pd.read_csv("microfinance_data_fuzzy_with_history.csv")

# Normalisation des données historiques
scaler = MinMaxScaler()
df["Statut_Emploi"] = df["Statut_Emploi"].astype("category").cat.codes
df[["Revenu_Mensuel", "Montant_Credit", "Duree_Credit", "Age",
    "Statut_Emploi", "Nbr_Total_Credits", "Nbr_Credits_Payes", "Nbr_Credits_NonPayes"]] = scaler.fit_transform(
    df[["Revenu_Mensuel", "Montant_Credit", "Duree_Credit", "Age",
        "Statut_Emploi", "Nbr_Total_Credits", "Nbr_Credits_Payes", "Nbr_Credits_NonPayes"]])


# Fonction de logique floue pour calculer une pondération entre 0 et 1
def calcul_ponderation_fuzzy(valeur, seuil_bas, seuil_haut):
    if valeur <= seuil_bas:
        return 0  # Pondération faible
    elif valeur >= seuil_haut:
        return 1  # Pondération élevée
    else:
        return (valeur - seuil_bas) / (seuil_haut - seuil_bas)  # Interpolation linéaire


def calcul_ponderations(data):
    """
    Cette fonction attribue une pondération floue pour chaque paramètre du client.
    Plus la valeur d’un paramètre est élevée dans une catégorie donnée, plus sa pondération est forte.
    """
    pondérations = {}

    # Revenu Mensuel (plus c’est élevé, mieux c’est)
    if data["Revenu_Mensuel"] < 100000:
        pondérations["Revenu_Mensuel"] = 0.2
    elif data["Revenu_Mensuel"] < 300000:
        pondérations["Revenu_Mensuel"] = 0.5
    else:
        pondérations["Revenu_Mensuel"] = 0.8

    # Montant Crédit (plus c’est élevé, plus le risque est grand)
    if data["Montant_Credit"] < 500000:
        pondérations["Montant_Credit"] = 0.8
    elif data["Montant_Credit"] < 1000000:
        pondérations["Montant_Credit"] = 0.5
    else:
        pondérations["Montant_Credit"] = 0.2

    # Âge (les jeunes ont plus de risques)
    if data["Age"] < 25:
        pondérations["Age"] = 0.3
    elif data["Age"] < 50:
        pondérations["Age"] = 0.7
    else:
        pondérations["Age"] = 0.5

    # Statut Emploi
    emploi_scores = {
        "CDI": 0.9,
        "CDD": 0.6,
        "Indépendant": 0.5,
        "Chômeur": 0.2
    }
    pondérations["Statut_Emploi"] = emploi_scores.get(data["Statut_Emploi"], 0.2)

    # Historique de crédit (logique floue déjà définie)
    pondérations["Historique_Credit"] = calcul_historique_credit_fuzzy(
        data["Nbr_Credits_Payes"], data["Nbr_Total_Credits"]
    )

    # Nombre total de crédits (plus on a de crédits en cours, plus c’est risqué)
    if data["Nbr_Total_Credits"] < 3:
        pondérations["Nbr_Total_Credits"] = 0.8
    elif data["Nbr_Total_Credits"] < 6:
        pondérations["Nbr_Total_Credits"] = 0.5
    else:
        pondérations["Nbr_Total_Credits"] = 0.2

    return pondérations


# Fonction pour calculer l'historique de crédit avec la logique floue
def calcul_historique_credit_fuzzy(nbr_payes, nbr_total):
    if nbr_total == 0:
        return 0  # Pas d'historique

    ratio = nbr_payes / nbr_total

    if ratio <= 0.4:
        return 0  # Mauvais historique
    elif ratio >= 0.8:
        return 1  # Bon historique
    else:
        return (ratio - 0.4) / (0.8 - 0.4)  # Interpolation linéaire


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Vérification des données entrantes
        expected_keys = ["Revenu_Mensuel", "Montant_Credit", "Duree_Credit", "Age", 
                         "Statut_Emploi", "Nbr_Total_Credits", "Nbr_Credits_Payes", "Nbr_Credits_NonPayes"]
        for key in expected_keys:
            if key not in data:
                return jsonify({"error": f"Donnée manquante : {key}"}), 400

        # Convertir le statut d'emploi en numérique
        employment_mapping = {"CDI": 0, "CDD": 1, "Indépendant": 2, "Chômeur": 3}
        data["Statut_Emploi"] = employment_mapping.get(data["Statut_Emploi"], 3)

        # Normalisation des données
        client_data = np.array([
            data["Revenu_Mensuel"], data["Montant_Credit"], data["Duree_Credit"],
            data["Age"], data["Statut_Emploi"], data["Nbr_Total_Credits"],
            data["Nbr_Credits_Payes"], data["Nbr_Credits_NonPayes"]
        ]).reshape(1, -1)

        client_data = scaler.transform(client_data)

        # Calcul des pondérations
        pondérations = calcul_ponderations(data)

        # Calcul du score global en pondérant chaque paramètre
        score_global = sum(client_data[0][i] * pondérations[key] for i, key in enumerate(pondérations))

        # Définition du seuil de défaut de paiement basé sur le score global
        defaut_paiement = 1 if score_global < 0.5 else 0

        # Sélection de l'action optimale via Q-learning
        action = np.argmax(q_table.mean(axis=0))

        # Décision finale basée sur le modèle et le score global
        decision = "Accordé" if action == 1 and defaut_paiement == 0 else "Refusé"

        return jsonify({
            "Ponderations": pondérations,
            "Score_Global": score_global,
            "Historique_Credit": pondérations["Historique_Credit"],
            "Defaut_Paiement": defaut_paiement,
            "Decision_Credit": decision
        })
    
    except Exception as e:
        return jsonify({"error": "Erreur interne du serveur", "details": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
