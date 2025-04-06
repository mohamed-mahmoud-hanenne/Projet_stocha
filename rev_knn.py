import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns


# 1. Importation et préparation des données
iris = pd.read_csv("iris_test")
X = iris[["petal_length, petal_width"]]
y = iris["species"]


# 2. Statistique descriptives  
print("\n=== Statistique descriptive")
print(iris.describe())
print("\Tableau de fréquence")
print(y.value_counts)


# 3. Visualisation 