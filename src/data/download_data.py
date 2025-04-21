import os
import requests
from tqdm import tqdm

def download_file(url, destination):

    if os.path.exists(destination):
        print(f"Le fichier {destination} existe déjà.")
        return
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    with open(destination, 'wb') as file, tqdm(
            desc=destination,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for data in response.iter_content(block_size):
            file.write(data)
            bar.update(len(data))

if __name__ == "__main__":
    # URL des données DVF (à remplacer par l'URL correcte si nécessaire)
    dvf_url = "https://www.data.gouv.fr/fr/datasets/r/90a98de0-f562-4328-aa16-fe0dd1dca60f"
    
    # Chemin de destination
    destination = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                              "data", "raw", "valeursfoncieres-2023.txt")
    
    # Créer le dossier s'il n'existe pas
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # Télécharger le fichier
    print(f"Téléchargement des données DVF...")
    download_file(dvf_url, destination)
    print("Téléchargement terminé !")