import os
import pandas as pd

# Créer le dossier sample
os.makedirs("data/sample", exist_ok=True)

# Créer un petit DataFrame d'exemple
df = pd.DataFrame({
    "surface": [100, 120, 150, 80, 200],
    "chambres": [2, 3, 4, 1, 5],
    "prix": [200000, 250000, 300000, 150000, 400000]
})

# Sauvegarder l'échantillon
df.to_csv("data/sample/dvf_features_sample.csv", index=False)
print("Échantillon CSV créé avec succès!")