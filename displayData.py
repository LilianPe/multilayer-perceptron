import pandas

# Charger le fichier CSV (remplace 'breast_cancer.csv' par le chemin de ton fichier)
data = pandas.read_csv('data.csv')

# Afficher les 5 premières lignes pour vérifier
print(data.head())

# Afficher les informations sur les colonnes et les types de données
print(data.info())