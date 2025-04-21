# 03_hyperparameter_tuning.py
# Script pour l'optimisation des hyperparamètres du meilleur modèle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
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

# Chargement des données préparées
print("Chargement des données préparées...")
prepared_data_path = os.path.join(processed_data_path, 'prepared_data.pkl')
imputer_path = os.path.join(models_path, 'imputer.pkl')
scaler_path = os.path.join(models_path, 'scaler.pkl')
best_model_path = os.path.join(models_path, 'best_model.pkl')

try:
    prepared_data = joblib.load(prepared_data_path)
    X_train = prepared_data['X_train']
    X_test = prepared_data['X_test']
    y_train = prepared_data['y_train']
    y_test = prepared_data['y_test']
    
    imputer = joblib.load(imputer_path)
    scaler = joblib.load(scaler_path)
    best_model = joblib.load(best_model_path)
    
    # Appliquer l'imputation et la normalisation
    X_train_imputed = imputer.transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    X_train_scaled = scaler.transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    print(f"Données et modèles chargés avec succès!")
    print(f"Type du meilleur modèle: {type(best_model).__name__}")

except FileNotFoundError as e:
    print(f"ERREUR: Fichier non trouvé - {e}")
    print("Veuillez d'abord exécuter les scripts 01_data_preparation.py et 02_model_training.py")
    exit(1)

# Définition des grilles de paramètres selon le type de modèle
model_type = type(best_model).__name__

if model_type == 'RandomForestRegressor':
    print("\nOptimisation des hyperparamètres pour Random Forest...")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
elif model_type == 'GradientBoostingRegressor':
    print("\nOptimisation des hyperparamètres pour Gradient Boosting...")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'min_samples_split': [2, 5, 10],
        'subsample': [0.8, 0.9, 1.0]
    }
    
elif model_type in ['Ridge', 'Lasso']:
    print(f"\nOptimisation des hyperparamètres pour {model_type}...")
    param_grid = {
        'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    }
    
elif model_type == 'LinearRegression':
    print("\nLinearRegression n'a pas d'hyperparamètres significatifs à optimiser.")
    print("Passage à l'évaluation du modèle...")
    
    # Évaluation du modèle
    y_pred = best_model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")
    
    # Sauvegarde du modèle final (même que le meilleur modèle)
    optimized_model_path = os.path.join(models_path, 'optimized_model.pkl')
    joblib.dump(best_model, optimized_model_path)
    print(f"Modèle sauvegardé dans: {optimized_model_path}")
    
    # Sortie du script
    exit(0)
    
else:
    print(f"Type de modèle non reconnu: {model_type}")
    print("Passage à l'évaluation du modèle sans optimisation...")
    
    # Évaluation du modèle
    y_pred = best_model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")
    
    # Sauvegarde du modèle final (même que le meilleur modèle)
    optimized_model_path = os.path.join(models_path, 'optimized_model.pkl')
    joblib.dump(best_model, optimized_model_path)
    print(f"Modèle sauvegardé dans: {optimized_model_path}")
    
    # Sortie du script
    exit(0)

# Optimisation des hyperparamètres avec GridSearchCV
print("\nDémarrage de la recherche par grille...")
print(f"Nombre de combinaisons à tester: {np.prod([len(v) for v in param_grid.values()])}")
print("Cette opération peut prendre beaucoup de temps...")

start_time = time.time()

# Utiliser RandomizedSearchCV si le nombre de combinaisons est trop grand
if np.prod([len(v) for v in param_grid.values()]) > 100:
    print("Nombre de combinaisons élevé, utilisation de RandomizedSearchCV...")
    search = RandomizedSearchCV(
        estimator=best_model,
        param_distributions=param_grid,
        n_iter=20,  # Nombre d'itérations
        cv=5,
        n_jobs=-1,
        scoring='neg_mean_squared_error',
        verbose=1,
        random_state=42
    )
else:
    search = GridSearchCV(
        estimator=best_model,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='neg_mean_squared_error',
        verbose=1
    )

# Entraînement
search.fit(X_train_scaled, y_train)

# Temps d'exécution
tuning_time = time.time() - start_time
print(f"\nTemps d'optimisation: {tuning_time:.2f} secondes ({tuning_time/60:.2f} minutes)")

# Afficher les meilleurs paramètres
print(f"\nMeilleurs paramètres: {search.best_params_}")
print(f"Meilleur score CV: {np.sqrt(-search.best_score_):.2f} (RMSE)")

# Évaluation sur l'ensemble de test
best_model_optimized = search.best_estimator_
y_pred_optimized = best_model_optimized.predict(X_test_scaled)
rmse_optimized = np.sqrt(mean_squared_error(y_test, y_pred_optimized))
r2_optimized = r2_score(y_test, y_pred_optimized)

print(f"\nPerformance sur l'ensemble de test:")
print(f"RMSE optimisé: {rmse_optimized:.2f}")
print(f"R² optimisé: {r2_optimized:.4f}")

# Sauvegarde du modèle optimisé
optimized_model_path = os.path.join(models_path, 'optimized_model.pkl')
joblib.dump(best_model_optimized, optimized_model_path)
print(f"\nModèle optimisé sauvegardé dans: {optimized_model_path}")

# Sauvegarde des résultats de la recherche
cv_results = pd.DataFrame(search.cv_results_)
cv_results_path = os.path.join(models_path, 'hyperparameter_tuning_results.csv')
cv_results.to_csv(cv_results_path, index=False)
print(f"Résultats de l'optimisation sauvegardés dans: {cv_results_path}")

# Visualisation de l'évolution des scores pendant la recherche
plt.figure(figsize=(12, 6))
plt.plot(np.sqrt(-cv_results['mean_test_score']), 'o-')
plt.title('Évolution du RMSE pendant l\'optimisation')
plt.xlabel('Itération')
plt.ylabel('RMSE (validation croisée)')
plt.grid(True)
plt.savefig(os.path.join(models_path, 'hyperparameter_tuning_evolution.png'))
plt.show()

# Comparaison avant/après optimisation
# Prédiction avec le modèle original
y_pred_original = best_model.predict(X_test_scaled)
rmse_original = np.sqrt(mean_squared_error(y_test, y_pred_original))
r2_original = r2_score(y_test, y_pred_original)

# Création d'un DataFrame pour la comparaison
comparison = pd.DataFrame({
    'Modèle': ['Original', 'Optimisé'],
    'RMSE': [rmse_original, rmse_optimized],
    'R²': [r2_original, r2_optimized]
})

print("\nComparaison avant/après optimisation:")
print(comparison)

# Visualisation de la comparaison
plt.figure(figsize=(10, 6))
sns.barplot(x='Modèle', y='RMSE', data=comparison)
plt.title('Comparaison RMSE avant/après optimisation')
plt.ylabel('RMSE (erreur)')
plt.grid(True)
plt.savefig(os.path.join(models_path, 'optimization_comparison_rmse.png'))
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='Modèle', y='R²', data=comparison)
plt.title('Comparaison R² avant/après optimisation')
plt.ylabel('R² (qualité de l\'ajustement)')
plt.grid(True)
plt.savefig(os.path.join(models_path, 'optimization_comparison_r2.png'))
plt.show()

print("\nOptimisation des hyperparamètres terminée!")