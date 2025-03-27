import os
import zipfile

# Dossier de destination
os.makedirs("ms_coco_2017", exist_ok=True)

# Liens des fichiers COCO Captions
files = {
    "train2017.zip": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
    "test2017.zip": "http://images.cocodataset.org/zips/test2017.zip", 
    "annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
}

# Télécharger les fichiers avec wget
for filename, url in files.items():
    os.system(f"wget -c {url} -P ms_coco_2017/")

# Extraire les fichiers ZIP et supprimer les archives après extraction
for filename in files.keys():
    zip_path = f"ms_coco_2017/{filename}"

    if os.path.exists(zip_path):  # Vérifier si le fichier a bien été téléchargé
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("ms_coco_2017/")  # Extraire dans le dossier "ms_coco_2017/"

        os.remove(zip_path)  # Supprimer le fichier ZIP après extraction
        print(f"Deleted {zip_path}")