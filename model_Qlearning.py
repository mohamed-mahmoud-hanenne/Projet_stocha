import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    df = pd.get_dummies(df, columns=['Sexe', 'Type_Emploi', 'Region'], drop_first=True)
    
    X = df.drop(columns=['Defaut_Paiement'])
    y = df['Defaut_Paiement']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, X.columns.tolist()

def balance_data(X, y):
    smote = SMOTE(sampling_strategy=0.75, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    return model

def predict_client(model, scaler, feature_names):
    print("Saisissez les informations du client :")
    age = int(input("Âge: "))
    revenu_mensuel = float(input("Revenu Mensuel: "))
    montant_credit = float(input("Montant Crédit: "))
    duree_credit = int(input("Durée Crédit (mois): "))
    historique_credit = int(input("Historique Crédit (0: Mauvais, 1: Bon): "))
    nombre_credits_en_cours = int(input("Nombre de crédits en cours: "))
    taux_interet = float(input("Taux d'intérêt: "))
    type_emploi = input("Type d'emploi (Salarié, Indépendant, Sans emploi): ")
    sexe = input("Sexe (Homme/Femme): ")
    region = input("Région (Nouakchott, Nouadhibou, Atar, Rosso, Kiffa): ")
    endettement = float(input("Endettement (0-1): "))
    
    # Mapping des valeurs pour correspondre aux colonnes générées
    features = {
        'Age': age, 'Revenu_Mensuel': revenu_mensuel, 'Montant_Credit': montant_credit,
        'Duree_Credit': duree_credit, 'Historique_Credit': historique_credit,
        'Nombre_Credits_En_Cours': nombre_credits_en_cours, 'Taux_Interet': taux_interet,
        'Endettement': endettement, 'Sexe_Homme': 1 if sexe == 'Homme' else 0,
        'Type_Emploi_Indépendant': 1 if type_emploi == 'Indépendant' else 0,
        'Type_Emploi_Sans emploi': 1 if type_emploi == 'Sans emploi' else 0,
        'Region_Nouakchott': 1 if region == 'Nouakchott' else 0,
        'Region_Nouadhibou': 1 if region == 'Nouadhibou' else 0,
        'Region_Atar': 1 if region == 'Atar' else 0,
        'Region_Rosso': 1 if region == 'Rosso' else 0,
        'Region_Kiffa': 1 if region == 'Kiffa' else 0
    }
    
    # Assurer que les features sont bien alignées
    client_data = [features.get(col, 0) for col in feature_names]
    client_data_scaled = scaler.transform([client_data])
    prediction = model.predict(client_data_scaled)[0]
    
    if prediction == 1:
        print("Décision: Refus du crédit (Défaut de paiement probable)")
    else:
        print("Décision: Crédit accordé (Faible risque de défaut)")

# Chargement et traitement des données
df = load_data("microfinance_mauritanie_logique.csv")
X, y, scaler, feature_names = preprocess_data(df)
X_balanced, y_balanced = balance_data(X, y)

# Entraînement du modèle
model = train_model(X_balanced, y_balanced)

# Prédiction pour un client
predict_client(model, scaler, feature_names)
