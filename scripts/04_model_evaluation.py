# 04_model_evaluation.py
# Script pour l'évaluation approfondie du modèle optimisé

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import cross_val_score, KFold
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
evaluation_path = os.path.join(project_root, 'evaluation')

# Créer le dossier d'évaluation s'il n'existe pas
os.makedirs(evaluation_path, exist_ok=True)

# Chargement des données et du modèle optimisé
print("Chargement des données et du modèle optimisé...")
prepared_data_path = os.path.join(processed_data_path, 'prepared_data.pkl')
imputer_path = os.path.join(models_path, 'imputer.pkl')
scaler_path = os.path.join(models_path, 'scaler.pkl')
optimized_model_path = os.path.join(models_path, 'optimized_model.pkl')

try:
    prepared_data = joblib.load(prepared_data_path)
    X_train = prepared_data['X_train']
    X_test = prepared_data['X_test']
    y_train = prepared_data['y_train']
    y_test = prepared_data['y_test']
    features = prepared_data['features']
    
    imputer = joblib.load(imputer_path)
    scaler = joblib.load(scaler_path)
    model = joblib.load(optimized_model_path)
    
    # Appliquer l'imputation et la normalisation
    X_train_imputed = imputer.transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    X_train_scaled = scaler.transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    print(f"Données et modèle chargés avec succès!")
    print(f"Type du modèle: {type(model).__name__}")

except FileNotFoundError as e:
    print(f"ERREUR: Fichier non trouvé - {e}")
    print("Veuillez d'abord exécuter les scripts précédents")
    exit(1)

# 1. Évaluation sur l'ensemble de test
print("\n1. Évaluation sur l'ensemble de test")
y_pred = model.predict(X_test_scaled)

# Métriques d'évaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.4f}")

# 2. Validation croisée
print("\n2. Validation croisée")
# Préparation des données complètes
X = pd.concat([X_train, X_test])
y = pd.concat([y_train, y_test])
X_imputed = imputer.transform(X)
X_scaled = scaler.transform(X_imputed)

# Configuration de la validation croisée
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Validation croisée
cv_scores = cross_val_score(
    model, 
    X_scaled, 
    y, 
    cv=kf, 
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

# Convertir en RMSE
cv_rmse_scores = np.sqrt(-cv_scores)

# Afficher les résultats
print("Résultats de la validation croisée (RMSE):")
for i, score in enumerate(cv_rmse_scores):
    print(f"Fold {i+1}: {score:.2f}")
print(f"Moyenne: {np.mean(cv_rmse_scores):.2f}")
print(f"Écart-type: {np.std(cv_rmse_scores):.2f}")

# Visualisation des scores de validation croisée
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(cv_rmse_scores) + 1), cv_rmse_scores)
plt.axhline(y=np.mean(cv_rmse_scores), color='r', linestyle='-', label=f'Moyenne: {np.mean(cv_rmse_scores):.2f}')
plt.title('Scores RMSE par fold (validation croisée)')
plt.xlabel('Fold')
plt.ylabel('RMSE')
plt.xticks(range(1, len(cv_rmse_scores) + 1))
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(evaluation_path, 'cross_validation_scores.png'))
plt.show()

# 3. Analyse des résidus
print("\n3. Analyse des résidus")
residuals = y_test - y_pred

# Statistiques des résidus
print("Statistiques des résidus:")
print(f"Moyenne: {np.mean(residuals):.2f}")
print(f"Écart-type: {np.std(residuals):.2f}")
print(f"Minimum: {np.min(residuals):.2f}")
print(f"Maximum: {np.max(residuals):.2f}")

# Visualisation des résidus
plt.figure(figsize=(12, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='-')
plt.title('Résidus vs Valeurs prédites')
plt.xlabel('Valeurs prédites')
plt.ylabel('Résidus')
plt.grid(True)
plt.savefig(os.path.join(evaluation_path, 'residuals_vs_predicted.png'))
plt.show()

# Distribution des résidus
plt.figure(figsize=(12, 6))
sns.histplot(residuals, kde=True, bins=50)
plt.title('Distribution des résidus')
plt.xlabel('Résidus')
plt.ylabel('Fréquence')
plt.grid(True)
plt.savefig(os.path.join(evaluation_path, 'residuals_distribution.png'))
plt.show()

# QQ-plot pour vérifier la normalité des résidus
plt.figure(figsize=(10, 10))
from scipy import stats
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('QQ-Plot des résidus')
plt.grid(True)
plt.savefig(os.path.join(evaluation_path, 'residuals_qqplot.png'))
plt.show()

# 4. Analyse des erreurs par segment
print("\n4. Analyse des erreurs par segment")

# Créer des segments basés sur les valeurs réelles
y_test_array = np.array(y_test)
percentiles = np.percentile(y_test_array, [0, 25, 50, 75, 100])
labels = ['0-25%', '25-50%', '50-75%', '75-100%']

# Fonction pour attribuer un segment
def assign_segment(value):
    for i in range(len(percentiles) - 1):
        if percentiles[i] <= value < percentiles[i + 1]:
            return labels[i]
    return labels[-1]  # Pour la dernière valeur

# Créer un DataFrame pour l'analyse
error_analysis = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred,
    'Residual': residuals,
    'AbsoluteError': np.abs(residuals),
    'PercentageError': np.abs(residuals / y_test) * 100
})

# Ajouter le segment
error_analysis['Segment'] = error_analysis['Actual'].apply(assign_segment)

# Analyse par segment
segment_analysis = error_analysis.groupby('Segment').agg({
    'AbsoluteError': ['mean', 'std', 'min', 'max'],
    'PercentageError': ['mean', 'std', 'min', 'max'],
    'Actual': ['count', 'mean']
})

print("Analyse des erreurs par segment de prix:")
print(segment_analysis)

# Visualisation des erreurs par segment
plt.figure(figsize=(12, 6))
sns.boxplot(x='Segment', y='AbsoluteError', data=error_analysis)
plt.title('Distribution des erreurs absolues par segment de prix')
plt.xlabel('Segment de prix')
plt.ylabel('Erreur absolue')
plt.grid(True)
plt.savefig(os.path.join(evaluation_path, 'errors_by_segment.png'))
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='Segment', y='PercentageError', data=error_analysis)
plt.title('Distribution des erreurs en pourcentage par segment de prix')
plt.xlabel('Segment de prix')
plt.ylabel('Erreur en pourcentage (%)')
plt.grid(True)
plt.savefig(os.path.join(evaluation_path, 'percentage_errors_by_segment.png'))
plt.show()

# 5. Visualisation des prédictions vs valeurs réelles
print("\n5. Visualisation des prédictions vs valeurs réelles")

plt.figure(figsize=(12, 8))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Valeurs prédites vs Valeurs réelles')
plt.xlabel('Valeurs réelles')
plt.ylabel('Valeurs prédites')
plt.grid(True)
plt.savefig(os.path.join(evaluation_path, 'predicted_vs_actual.png'))
plt.show()

# 6. Génération d'un rapport d'évaluation
print("\n6. Génération d'un rapport d'évaluation")

# Créer un rapport markdown
report = f"""
# Rapport d'évaluation du modèle de prédiction immobilière

## 1. Informations générales

- **Type de modèle**: {type(model).__name__}
- **Nombre d'observations**: {len(y_test)}
- **Nombre de features**: {len(features)}
- **Date d'évaluation**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 2. Métriques de performance

- **RMSE (Root Mean Squared Error)**: {rmse:.2f}
- **MAE (Mean Absolute Error)**: {mae:.2f}
- **R² (Coefficient de détermination)**: {r2:.4f}

## 3. Validation croisée (5-fold)

- **RMSE moyen**: {np.mean(cv_rmse_scores):.2f}
- **Écart-type RMSE**: {np.std(cv_rmse_scores):.2f}
- **Scores par fold**: {', '.join([f"{score:.2f}" for score in cv_rmse_scores])}

## 4. Analyse des résidus

- **Moyenne des résidus**: {np.mean(residuals):.2f}
- **Écart-type des résidus**: {np.std(residuals):.2f}
- **Résidu minimum**: {np.min(residuals):.2f}
- **Résidu maximum**: {np.max(residuals):.2f}

## 5. Analyse par segment de prix

{segment_analysis.to_markdown()}

## 6. Conclusion

Le modèle {type(model).__name__} atteint un R² de {r2:.4f}, ce qui signifie qu'il explique environ {r2*100:.1f}% de la variance dans les prix immobiliers. Le RMSE de {rmse:.2f} indique l'erreur moyenne en unités de la variable cible.

L'analyse des résidus montre que {np.mean(residuals) > 0 and "le modèle a tendance à sous-estimer les prix" or "le modèle a tendance à surestimer les prix"}.

L'analyse par segment révèle que le modèle est {segment_analysis['AbsoluteError']['mean'].idxmin()} pour les propriétés dans le segment de prix {segment_analysis['AbsoluteError']['mean'].idxmin()}.

## 7. Recommandations

1. {r2 < 0.7 and "Améliorer le modèle en collectant plus de données ou en ajoutant des features pertinentes." or "Le modèle performe bien, mais pourrait bénéficier d'une mise à jour régulière avec de nouvelles données."}
2. {np.std(residuals) > rmse and "Investiguer les valeurs aberrantes qui pourraient affecter les performances du modèle." or "La distribution des erreurs est relativement stable à travers les différentes gammes de prix."}
3. {segment_analysis['PercentageError']['mean'].max() > 20 and f"Porter une attention particulière au segment {segment_analysis['PercentageError']['mean'].idxmax()} où les erreurs en pourcentage sont les plus élevées." or "Les performances sont relativement homogènes à travers les différents segments de prix."}
"""

# Sauvegarder le rapport
report_path = os.path.join(evaluation_path, 'evaluation_report.md')
with open(report_path, 'w') as f:
    f.write(report)
print(f"Rapport d'évaluation sauvegardé dans: {report_path}")

print("\nÉvaluation du modèle terminée!")