# 02_model_training.py
# Script pour l'entraînement des modèles de base

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Configuration pour de meilleurs graphiques
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Définir le chemin du projet
project_root = r'C:\Users\Marouane\ImmoPredict'
processed_data_path = os.path.join(project_root, 'data', 'processed')
models_path = os.path.join(project_root, 'models')

# Créer les dossiers s'ils n'existent pas
os.makedirs(models_path, exist_ok=True)

# Chargement des données préparées
print("Chargement des données préparées...")
prepared_data_path = os.path.join(processed_data_path, 'prepared_data.pkl')

try:
    prepared_data = joblib.load(prepared_data_path)
    X_train = prepared_data['X_train']
    X_test = prepared_data['X_test']
    y_train = prepared_data['y_train']
    y_test = prepared_data['y_test']
    features = prepared_data['features']
    target = prepared_data['target']
    
    print(f"Données chargées avec succès!")
    print(f"Taille de l'ensemble d'entraînement: {X_train.shape}")
    print(f"Taille de l'ensemble de test: {X_test.shape}")
    print(f"Nombre de features: {len(features)}")

except FileNotFoundError:
    print(f"ERREUR: Fichier {prepared_data_path} non trouvé!")
    print("Veuillez d'abord exécuter le script 01_data_preparation.py")
    exit(1)

# Imputation des valeurs manquantes
print("\nImputation des valeurs manquantes...")
start_time = time.time()
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)
print(f"Temps d'imputation: {time.time() - start_time:.2f} secondes")

# Sauvegarde de l'imputer
imputer_path = os.path.join(models_path, 'imputer.pkl')
joblib.dump(imputer, imputer_path)
print(f"Imputer sauvegardé dans: {imputer_path}")

# Normalisation des données
print("Normalisation des données...")
start_time = time.time()
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)
print(f"Temps de normalisation: {time.time() - start_time:.2f} secondes")

# Sauvegarde du scaler
scaler_path = os.path.join(models_path, 'scaler.pkl')
joblib.dump(scaler, scaler_path)
print(f"Scaler sauvegardé dans: {scaler_path}")

# Entraînement de différents modèles
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'Random Forest': RandomForestRegressor(
        n_estimators=50,  # Réduit pour accélérer l'entraînement
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        n_jobs=-1,
        random_state=42
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=50,  # Réduit pour accélérer l'entraînement
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
}

results = {}

for name, model in models.items():
    print(f"\nEntraînement du modèle: {name}")
    start_time = time.time()
    
    # Entraînement
    model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    
    # Prédiction
    y_pred = model.predict(X_test_scaled)
    
    # Évaluation
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'Temps d\'entraînement (s)': training_time
    }
    
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  R²: {r2:.4f}")
    print(f"  Temps d'entraînement: {training_time:.2f} secondes ({training_time/60:.2f} minutes)")
    
    # Sauvegarde du modèle
    model_path = os.path.join(models_path, f"{name.lower().replace(' ', '_')}.pkl")
    joblib.dump(model, model_path)
    print(f"  Modèle sauvegardé dans: {model_path}")

# Comparaison des modèles
results_df = pd.DataFrame(results).T
print("\nComparaison des modèles:")
print(results_df)

# Sauvegarde des résultats
results_path = os.path.join(models_path, 'model_results.csv')
results_df.to_csv(results_path)
print(f"Résultats sauvegardés dans: {results_path}")

# Visualisation des performances (RMSE)
plt.figure(figsize=(12, 6))
sns.barplot(x=results_df.index, y=results_df['RMSE'])
plt.title('Comparaison des modèles - RMSE (erreur)')
plt.xlabel('Modèle')
plt.ylabel('RMSE (€)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(models_path, 'model_comparison_rmse.png'))
plt.show()

# Visualisation des performances (R²)
plt.figure(figsize=(12, 6))
sns.barplot(x=results_df.index, y=results_df['R²'])
plt.title('Comparaison des modèles - R² (qualité de l\'ajustement)')
plt.xlabel('Modèle')
plt.ylabel('R²')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(models_path, 'model_comparison_r2.png'))
plt.show()

# Visualisation du temps d'entraînement
plt.figure(figsize=(12, 6))
sns.barplot(x=results_df.index, y=results_df['Temps d\'entraînement (s)'])
plt.title('Comparaison des modèles - Temps d\'entraînement')
plt.xlabel('Modèle')
plt.ylabel('Temps (secondes)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(models_path, 'model_comparison_time.png'))
plt.show()

# Sélection du meilleur modèle
best_model_name = results_df['R²'].idxmax()
best_model = models[best_model_name]

print(f"\nMeilleur modèle: {best_model_name}")
print(f"R²: {results_df.loc[best_model_name, 'R²']:.4f}")
print(f"RMSE: {results_df.loc[best_model_name, 'RMSE']:.2f}")

# Sauvegarde du meilleur modèle
best_model_path = os.path.join(models_path, 'best_model.pkl')
joblib.dump(best_model, best_model_path)
print(f"Meilleur modèle sauvegardé dans: {best_model_path}")

# Analyse de l'importance des features (pour les modèles basés sur les arbres)
if hasattr(best_model, 'feature_importances_'):
    # Vérification des longueurs
    print(f"\nNombre de features dans la liste 'features': {len(features)}")
    print(f"Nombre de features dans 'feature_importances_': {len(best_model.feature_importances_)}")
    
    # Si les longueurs sont différentes, utilisez uniquement les features utilisées par le modèle
    if len(features) != len(best_model.feature_importances_):
        print("ATTENTION: Différence de longueur détectée!")
        
        # Solution 1: Utiliser les noms de colonnes du modèle si disponible
        if hasattr(best_model, 'feature_names_in_'):
            print("Utilisation des noms de features du modèle...")
            model_features = best_model.feature_names_in_
            feature_importance = pd.DataFrame({
                'Feature': model_features,
                'Importance': best_model.feature_importances_
            })
        # Solution 2: Créer des noms génériques
        else:
            print("Création de noms de features génériques...")
            feature_importance = pd.DataFrame({
                'Feature': [f'Feature_{i}' for i in range(len(best_model.feature_importances_))],
                'Importance': best_model.feature_importances_
            })
    else:
        # Si les longueurs correspondent, procéder normalement
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Importance': best_model.feature_importances_
        })
    
    # Tri par importance décroissante
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    # Visualisation
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
    plt.title(f'Top 15 des variables les plus importantes ({best_model_name})')
    plt.tight_layout()
    plt.savefig(os.path.join(models_path, 'feature_importance.png'))
    plt.show()
    
    # Sauvegarde de l'importance des features
    importance_path = os.path.join(models_path, 'feature_importance.csv')
    feature_importance.to_csv(importance_path, index=False)
    print(f"Importance des features sauvegardée dans: {importance_path}")

print("\nEntraînement des modèles terminé!")