# -*- coding: utf-8 -*-
"""
Interface de prédiction pour le modèle immobilier
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
import json
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Configuration de base
project_root = r'C:\Users\Marouane\ImmoPredict'
models_dir = os.path.join(project_root, 'models')
data_dir = os.path.join(project_root, 'data')
features_path = os.path.join(data_dir, 'processed', 'dvf_features.csv')

# Vérifier si les répertoires existent
os.makedirs(models_dir, exist_ok=True)

# Fonction pour charger les modèles
def load_models():
    """
    Charge l'imputer, le scaler et le modèle de prédiction.
    """
    try:
        imputer_path = os.path.join(models_dir, 'imputer.pkl')
        scaler_path = os.path.join(models_dir, 'scaler.pkl')
        model_path = os.path.join(models_dir, 'best_model.pkl')
        
        imputer = joblib.load(imputer_path)
        scaler = joblib.load(scaler_path)
        model = joblib.load(model_path)
        
        print("Modèles chargés avec succès.")
        return imputer, scaler, model
    
    except Exception as e:
        print(f"Erreur lors du chargement des modèles: {e}")
        return None, None, None

# Fonction pour obtenir les noms des colonnes exactes utilisées lors de l'entraînement
def get_feature_columns():
    """
    Récupère les noms exacts des colonnes utilisées pour l'entraînement.
    """
    try:
        # Charger l'imputer pour obtenir les noms exacts des colonnes
        imputer_path = os.path.join(models_dir, 'imputer.pkl')
        imputer = joblib.load(imputer_path)
        
        # Si l'imputer a un attribut feature_names_in_, l'utiliser
        if hasattr(imputer, 'feature_names_in_'):
            return imputer.feature_names_in_.tolist()
        
        # Sinon, essayer de charger à partir d'un fichier de configuration
        feature_names_path = os.path.join(models_dir, 'feature_names.json')
        if os.path.exists(feature_names_path):
            with open(feature_names_path, 'r') as f:
                feature_names = json.load(f)
            return feature_names['feature_names']
        
        # En dernier recours, récupérer à partir du fichier CSV original
        df_original = pd.read_csv(features_path, low_memory=False)
        numeric_cols = df_original.select_dtypes(include=['number']).columns.tolist()
        
        # Exclure la variable cible
        if 'Valeur fonciere' in numeric_cols:
            numeric_cols.remove('Valeur fonciere')
        
        return numeric_cols
    
    except Exception as e:
        print(f"Erreur lors de la récupération des noms de colonnes: {e}")
        return []

# Fonction de prédiction
def predict_property_price(property_data):
    """
    Prédit le prix d'une propriété à partir des caractéristiques fournies.
    """
    try:
        # Charger les modèles
        imputer, scaler, model = load_models()
        if imputer is None or scaler is None or model is None:
            return None
        
        # Obtenir les noms exacts des colonnes utilisées lors de l'entraînement
        feature_columns = get_feature_columns()
        if not feature_columns:
            print("Impossible de déterminer les caractéristiques du modèle.")
            return None
        
        # Créer un DataFrame avec toutes les colonnes nécessaires, initialisées à 0
        property_df = pd.DataFrame(0, index=[0], columns=feature_columns)
        
        # Mettre à jour uniquement les colonnes fournies dans property_data
        for key, value in property_data.items():
            if key in property_df.columns:
                property_df.loc[0, key] = value
        
        # S'assurer que toutes les colonnes nécessaires sont présentes
        # Notamment '2eme lot' qui semble causer des problèmes
        if '2eme lot' not in property_df.columns and '2eme lot' in feature_columns:
            property_df['2eme lot'] = 0
        
        # Réorganiser les colonnes pour qu'elles correspondent exactement à l'ordre utilisé lors de l'entraînement
        property_df = property_df[feature_columns]
        
        # Appliquer l'imputation et la normalisation
        property_imputed = imputer.transform(property_df)
        property_scaled = scaler.transform(property_imputed)
        
        # Prédire le prix
        predicted_price = model.predict(property_scaled)[0]
        
        return predicted_price
    
    except Exception as e:
        print(f"Erreur lors de la prédiction: {e}")
        import traceback
        traceback.print_exc()
        return None

# Fonction pour tester l'impact des features
def test_feature_impact():
    """
    Teste l'impact de chaque feature sur la prédiction.
    """
    base_property = {
        'surface': 100,
        'chambres': 2,
        'salles_bain': 1,
        'etage': 1,
        'distance_centre': 5
    }
    
    print("\nTest de l'impact des features sur la prédiction:")
    print(f"Propriété de base: {base_property}")
    base_price = predict_property_price(base_property)
    
    if base_price is None:
        print("Impossible de calculer le prix de base.")
        return
    
    print(f"Prix de base: {base_price:,.2f} €")
    
    # Tester l'impact de chaque feature
    for feature in base_property:
        test_property = base_property.copy()
        
        # Augmenter la valeur de 50%
        test_property[feature] *= 1.5
        
        test_price = predict_property_price(test_property)
        
        if test_price is not None:
            impact = ((test_price - base_price) / base_price) * 100
            print(f"Impact de +50% de {feature}: {impact:.2f}% ({test_price:,.2f} €)")
        else:
            print(f"Impossible de calculer l'impact de {feature}.")

# Fonction pour la prédiction interactive
def interactive_prediction():
    """
    Permet à l'utilisateur de saisir les caractéristiques d'une propriété et prédit son prix.
    """
    print("\n" + "=" * 50)
    print("PRÉDICTEUR DE PRIX IMMOBILIER")
    print("=" * 50)
    
    # Charger les plages de valeurs typiques à partir des données d'entraînement
    try:
        df_stats = pd.read_csv(features_path, low_memory=False)
        
        # Afficher les plages pour les principales caractéristiques
        print("Plages de valeurs typiques:")
        for feature in ['surface', 'chambres', 'salles_bain', 'etage', 'distance_centre']:
            if feature in df_stats.columns:
                min_val = df_stats[feature].min()
                max_val = df_stats[feature].max()
                median_val = df_stats[feature].median()
                print(f"  {feature}: {min_val:.1f} à {max_val:.1f} (médiane: {median_val:.1f})")
    except Exception as e:
        print("Impossible de déterminer les caractéristiques exactes du modèle.")
    
    # Saisie des caractéristiques
    property_data = {}
    
    try:
        property_data['surface'] = float(input("Surface (m²): "))
        property_data['chambres'] = int(input("Nombre de chambres: "))
        property_data['salles_bain'] = int(input("Nombre de salles de bain: "))
        property_data['etage'] = int(input("Étage (0 pour RDC): "))
        property_data['distance_centre'] = float(input("Distance du centre-ville (km): "))
        
    except ValueError:
        print("Erreur: Veuillez entrer des valeurs numériques valides.")
        return
    
    # Prédiction
    start_time = time.time()
    predicted_price = predict_property_price(property_data)
    prediction_time = time.time() - start_time
    
    if predicted_price is not None:
        print("\nRÉSULTAT:")
        print(f"Prix estimé: {predicted_price:,.2f} €")
        
        # Ajouter un intervalle de confiance (estimation simple)
        print(f"Intervalle de confiance: {predicted_price*0.85:,.2f} € à {predicted_price*1.15:,.2f} €")
        print(f"Temps de prédiction: {prediction_time:.2f} secondes")
        print("=" * 50)
    else:
        print("\nLa prédiction a échoué. Veuillez vérifier les logs pour plus de détails.")

# Fonction pour la prédiction par lot
def batch_prediction():
    """
    Permet à l'utilisateur de prédire les prix pour un fichier CSV contenant plusieurs propriétés.
    """
    print("\n" + "=" * 50)
    print("PRÉDICTION PAR LOT")
    print("=" * 50)
    
    # Demander le chemin du fichier
    file_path = input("Chemin du fichier CSV: ")
    
    if not os.path.exists(file_path):
        print(f"Erreur: Le fichier {file_path} n'existe pas.")
        return
    
    try:
        # Charger le fichier
        df = pd.read_csv(file_path, low_memory=False)
        print(f"Fichier chargé avec succès. {len(df)} propriétés trouvées.")
        
        # Vérifier les colonnes requises
        required_columns = ['surface', 'chambres', 'salles_bain', 'etage', 'distance_centre']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Erreur: Colonnes manquantes dans le fichier: {missing_columns}")
            return
        
        # Prédire les prix
        print("Prédiction en cours...")
        start_time = time.time()
        
        predictions = []
        for i, row in df.iterrows():
            property_data = {
                'surface': row['surface'],
                'chambres': row['chambres'],
                'salles_bain': row['salles_bain'],
                'etage': row['etage'],
                'distance_centre': row['distance_centre']
            }
            
            predicted_price = predict_property_price(property_data)
            predictions.append(predicted_price)
            
            # Afficher la progression
            if (i + 1) % 10 == 0 or i == len(df) - 1:
                print(f"Progression: {i+1}/{len(df)} ({(i+1)/len(df)*100:.1f}%)")
        
        # Ajouter les prédictions au DataFrame
        df['prix_predit'] = predictions
        
        # Sauvegarder les résultats
        output_path = os.path.splitext(file_path)[0] + "_predictions.csv"
        df.to_csv(output_path, index=False)
        
        prediction_time = time.time() - start_time
        print(f"\nPrédictions terminées en {prediction_time:.2f} secondes.")
        print(f"Résultats sauvegardés dans: {output_path}")
        
        # Afficher quelques statistiques
        print("\nStatistiques des prédictions:")
        print(f"Prix moyen: {df['prix_predit'].mean():,.2f} €")
        print(f"Prix médian: {df['prix_predit'].median():,.2f} €")
        print(f"Prix minimum: {df['prix_predit'].min():,.2f} €")
        print(f"Prix maximum: {df['prix_predit'].max():,.2f} €")
        
    except Exception as e:
        print(f"Erreur lors de la prédiction par lot: {e}")
        import traceback
        traceback.print_exc()

# Fonction pour l'exemple de prédiction
def example_prediction():
    """
    Affiche un exemple de prédiction avec des valeurs prédéfinies.
    """
    # Créer un exemple de propriété
    example_property = {
        'surface': 120,
        'chambres': 3,
        'salles_bain': 2,
        'etage': 2,
        'distance_centre': 5
    }
    
    print("\nExemple de propriété:")
    for key, value in example_property.items():
        print(f"{key}: {value}")
    
    predicted_price = predict_property_price(example_property)
    
    if predicted_price is not None:
        print("\nRÉSULTAT:")
        print(f"Prix estimé: {predicted_price:,.2f} €")
    else:
        print("\nLa prédiction a échoué. Veuillez vérifier les logs pour plus de détails.")

# Fonction principale
def main():
    """
    Fonction principale qui affiche le menu et gère les choix de l'utilisateur.
    """
    while True:
        print("\n" + "=" * 50)
        print("MENU PRINCIPAL - PRÉDICTEUR IMMOBILIER")
        print("=" * 50)
        print("1. Prédiction interactive")
        print("2. Prédiction par lot (fichier CSV)")
        print("3. Exemple de prédiction")
        print("4. Tester l'impact des features")
        print("5. Quitter")
        
        choice = input("\nVotre choix (1-5): ")
        
        if choice == '1':
            interactive_prediction()
        elif choice == '2':
            batch_prediction()
        elif choice == '3':
            example_prediction()
        elif choice == '4':
            test_feature_impact()
        elif choice == '5':
            print("\nMerci d'avoir utilisé le prédicteur immobilier. Au revoir!")
            sys.exit(0)
        else:
            print("Choix invalide. Veuillez réessayer.")

# Point d'entrée du programme
if __name__ == "__main__":
    # Vérifier si les modèles existent
    imputer_path = os.path.join(models_dir, 'imputer.pkl')
    model_path = os.path.join(models_dir, 'best_model.pkl')
    
    if not os.path.exists(imputer_path) or not os.path.exists(model_path):
        print("ERREUR: Les modèles nécessaires n'ont pas été trouvés.")
        print(f"Veuillez vous assurer que les fichiers suivants existent:")
        print(f"- {imputer_path}")
        print(f"- {model_path}")
        sys.exit(1)
    

    
    # Lancer l'application
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgramme interrompu par l'utilisateur. Au revoir!")
        sys.exit(0)
    except Exception as e:
        print(f"\nUne erreur inattendue s'est produite: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    