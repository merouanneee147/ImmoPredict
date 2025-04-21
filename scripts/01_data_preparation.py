# 01_data_preparation.py
# Script pour le chargement et la préparation des données

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
import joblib

# Configuration pour de meilleurs graphiques
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Définir le chemin du projet
project_root = r'C:\Users\Marouane\ImmoPredict'
raw_data_path = os.path.join(project_root, 'data', 'raw')
processed_data_path = os.path.join(project_root, 'data', 'processed')

# Créer les dossiers s'ils n'existent pas
os.makedirs(raw_data_path, exist_ok=True)
os.makedirs(processed_data_path, exist_ok=True)
os.makedirs(os.path.join(project_root, 'models'), exist_ok=True)

# Chargement des données
print("Chargement des données...")
features_path = os.path.join(processed_data_path, 'dvf_features.csv')
df = pd.read_csv(features_path)
print(f"Dimensions initiales: {df.shape}")

# Afficher les premières lignes
print("\nAperçu des données:")
print(df.head())

# Informations sur les types de données
print("\nInformations sur les types de données:")
df.info()

# Statistiques descriptives
print("\nStatistiques descriptives:")
print(df.describe())

# Sélectionner uniquement les colonnes numériques
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
df = df[numeric_cols]
print(f"\nDimensions après sélection des colonnes numériques: {df.shape}")

# Vérification des valeurs manquantes
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100
missing_df = pd.DataFrame({'Nombre': missing_values, 'Pourcentage (%)': missing_percent})
print("\nValeurs manquantes par colonne:")
print(missing_df[missing_df['Nombre'] > 0].sort_values('Pourcentage (%)', ascending=False))

# Visualisation des valeurs manquantes
plt.figure(figsize=(12, 8))
plt.title('Pourcentage de valeurs manquantes par colonne')
missing_df = missing_df[missing_df['Nombre'] > 0].sort_values('Pourcentage (%)', ascending=False)
if not missing_df.empty:
    sns.barplot(x=missing_df.index, y='Pourcentage (%)', data=missing_df)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(processed_data_path, 'missing_values.png'))
    plt.show()
else:
    print("Aucune valeur manquante trouvée.")

# Définition de la variable cible
target = 'Valeur fonciere'

# Vérifier si la variable cible existe
if target not in df.columns:
    print(f"ERREUR: La colonne cible '{target}' n'existe pas dans le dataset!")
    print(f"Colonnes disponibles: {df.columns.tolist()}")
else:
    # Vérifier si la variable cible a des valeurs manquantes
    if df[target].isnull().sum() > 0:
        print(f"ATTENTION: La variable cible a {df[target].isnull().sum()} valeurs manquantes.")
        print("Suppression des lignes avec des valeurs manquantes dans la variable cible...")
        df = df.dropna(subset=[target])
        print(f"Dimensions après suppression: {df.shape}")
    
    # Sélection des features (toutes les colonnes sauf la cible)
    features = [col for col in df.columns if col != target]
    
    # Définition des features et de la cible
    X = df[features]
    y = df[target]
    
    print(f"\nNombre de features: {len(features)}")
    
    # Division en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Taille de l'ensemble d'entraînement: {X_train.shape}")
    print(f"Taille de l'ensemble de test: {X_test.shape}")
    
    # Sauvegarde des données préparées
    prepared_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'features': features,
        'target': target
    }
    
    prepared_data_path = os.path.join(processed_data_path, 'prepared_data.pkl')
    joblib.dump(prepared_data, prepared_data_path)
    print(f"\nDonnées préparées sauvegardées dans: {prepared_data_path}")
    
    # Visualisation de la distribution de la variable cible
    plt.figure(figsize=(12, 6))
    sns.histplot(df[target], kde=True, bins=50)
    plt.title(f'Distribution de {target}')
    plt.xlabel(target)
    plt.ylabel('Fréquence')
    plt.tight_layout()
    plt.savefig(os.path.join(processed_data_path, 'target_distribution.png'))
    plt.show()
    
    # Matrice de corrélation
    plt.figure(figsize=(14, 12))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', 
                linewidths=0.5, vmin=-1, vmax=1)
    plt.title('Matrice de corrélation')
    plt.tight_layout()
    plt.savefig(os.path.join(processed_data_path, 'correlation_matrix.png'))
    plt.show()
    
    # Top 10 des corrélations avec la variable cible
    correlations = df.corr()[target].sort_values(ascending=False)
    print("\nTop 10 des features les plus corrélées avec la variable cible:")
    print(correlations.head(11))  # 11 car la première est la cible elle-même
    
    # Visualisation des top corrélations
    plt.figure(figsize=(12, 8))
    top_corr = correlations[1:11]  # Exclure la cible elle-même
    sns.barplot(x=top_corr.values, y=top_corr.index)
    plt.title(f'Top 10 des features les plus corrélées avec {target}')
    plt.xlabel('Coefficient de corrélation')
    plt.tight_layout()
    plt.savefig(os.path.join(processed_data_path, 'top_correlations.png'))
    plt.show()

print("\nPréparation des données terminée!")