import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns



# 1. Importation et préparation
iris = pd.read_csv("iris_test.csv")
X = iris[["petal_length", "petal_width"]]
y = iris["species"]


# 2 Statistique descriptive
print("\=== Statistique descriptive===")
print(iris.describe())
print("\nTableau de frequence")
print(y.value_counts())


# 3 Visualisation pour les variables importantes
plt.figure(figsize=(10,5))
sns.boxplot(x = 'species', y = 'petal_length', data=iris)
plt.title("Boite à moustache de la longuer de petal par espéce")
plt.show()


# 4 Séparation des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5 Model knn
k = 3 
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)

# 6 Prédiction et évaluation 
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n Precision du modél {accuracy:.2%}")