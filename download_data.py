import os
import requests

def download_file(url, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Téléchargement de {url}...")
    response = requests.get(url, stream=True)
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Fichier sauvegardé dans {output_path}")

def main():
    files_to_download = [
        {"url": "URL_VERS_VOS_DONNEES", "path": "data/raw/ValeursFoncieres-2024.txt"},
    ]
    for file_info in files_to_download:
        download_file(file_info["url"], file_info["path"])

if __name__ == "__main__":
    main()