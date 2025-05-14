import os
import pandas
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '../data/data.csv')

# Charger le fichier CSV (remplace 'breast_cancer.csv' par le chemin de ton fichier)
data = pandas.read_csv(data_path, header=None)

# Afficher les 5 premières lignes pour vérifier
print(data.head())

# Afficher les informations sur les colonnes et les types de données
print(data.info())

plt.hist(data[3], bins=20, color='skyblue', edgecolor='black')
plt.title("Distribution de la taille (colonne 2)")
plt.xlabel("Valeur")
plt.ylabel("Fréquence")
plt.show()

# Enlever l'affichage, puis coder le modele de maniere propre:
# POO, puis utiliser un main et une structure propre, doc de fonctions etc...