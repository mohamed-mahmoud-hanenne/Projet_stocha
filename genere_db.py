import pandas as pd
import numpy as np
from skfuzzy import control as ctrl
import skfuzzy as fuzz

np.random.seed(42)
data_size = 1000  # Nombre de clients

# Génération des données
ids = np.arange(1, data_size + 1)
age = np.random.randint(18, 65, size=data_size)
sexe = np.random.choice(["Homme", "Femme"], size=data_size)
revenu_mensuel = np.random.uniform(5000, 100000, size=data_size)
montant_credit = np.random.uniform(10000, 500000, size=data_size)
duree_credit = np.random.randint(6, 60, size=data_size)  # Durée en mois
taux_interet = np.random.uniform(2, 12, size=data_size)  # Taux en pourcentage
type_emploi = np.random.choice(["Salarié", "Indépendant", "Sans emploi"], size=data_size)
region = np.random.choice(["Nouakchott", "Nouadhibou", "Atar", "Rosso", "Kiffa"], size=data_size)
nombre_total_credits = np.random.randint(1, 10, size=data_size)
total_credits_payes = np.random.randint(0, nombre_total_credits + 1, size=data_size)
total_credits_non_payes = nombre_total_credits - total_credits_payes

# Définition du système de logique floue
credit_payes = ctrl.Antecedent(np.arange(0, 11, 1), 'credit_payes')
credit_non_payes = ctrl.Antecedent(np.arange(0, 11, 1), 'credit_non_payes')
historique_credit = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'historique_credit')

credit_payes.automf(3)
credit_non_payes.automf(3)
historique_credit['mauvais'] = fuzz.trimf(historique_credit.universe, [0, 0, 0.5])
historique_credit['bon'] = fuzz.trimf(historique_credit.universe, [0.5, 1, 1])

rule1 = ctrl.Rule(credit_payes['poor'] & credit_non_payes['good'], historique_credit['mauvais'])
rule2 = ctrl.Rule(credit_payes['good'] & credit_non_payes['poor'], historique_credit['bon'])
rule3 = ctrl.Rule(credit_payes['average'] & credit_non_payes['average'], historique_credit['mauvais'])
rule4 = ctrl.Rule(credit_payes['good'] & credit_non_payes['average'], historique_credit['bon'])

historique_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
historique_simulation = ctrl.ControlSystemSimulation(historique_ctrl)

# Initialisation des listes
historique_credit_values = []
endettement = []
defaut_paiement = []

for i in range(data_size):
    historique_simulation.input['credit_payes'] = total_credits_payes[i]
    historique_simulation.input['credit_non_payes'] = total_credits_non_payes[i]

    print(f"Avant le calcul - Crédit payés: {total_credits_payes[i]}, Crédit non payés: {total_credits_non_payes[i]}")
    historique_simulation.compute()
    print(f"Après le calcul - Sortie: {historique_simulation.output}")
    
    hist_credit = 1 if historique_simulation.output.get('historique_credit', 0) > 0.5 else 0
    historique_credit_values.append(hist_credit)
    
    # Calcul de l'endettement basé sur la logique floue
    endettement_client = (montant_credit[i] / revenu_mensuel[i]) * (duree_credit[i] / 60)
    endettement.append(min(endettement_client, 1))
    
    # Déterminer le défaut de paiement basé sur la logique floue
    defaut = 1 if hist_credit == 0 and endettement_client > 0.5 else 0
    defaut_paiement.append(defaut)

# Création du DataFrame
df_microfinance = pd.DataFrame({
    "ID_Client": ids,
    "Age": age,
    "Sexe": sexe,
    "Revenu_Mensuel": revenu_mensuel,
    "Montant_Credit": montant_credit,
    "Duree_Credit": duree_credit,
    "Taux_Interet": taux_interet,
    "Type_Emploi": type_emploi,
    "Region": region,
    "Nombre_Total_Credits": nombre_total_credits,
    "Total_Credits_Payes": total_credits_payes,
    "Total_Credit_Non_Payes": total_credits_non_payes,
    "Historique_Credit": historique_credit_values,
    "Endettement": endettement,
    "Defaut_Paiement": defaut_paiement
})

# Sauvegarde en fichier CSV
file_path = "microfinance_mauritanie_logique_test.csv"
df_microfinance.to_csv(file_path, index=False)

print(f"✅ Le fichier {file_path} a été généré avec succès avec des données logiques et cohérentes.")
