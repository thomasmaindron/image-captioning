import os
import zipfile
from dataset.utils.dataset_utils import process_dataset

# Dossier de destination
os.makedirs("dataset/ms_coco_2017", exist_ok=True)

# Liens des fichiers COCO Captions et leurs dossiers correspondants
files = {
    "train2017.zip": ("http://images.cocodataset.org/zips/train2017.zip", "dataset/ms_coco_2017/train2017"),
    "val2017.zip": ("http://images.cocodataset.org/zips/val2017.zip", "dataset/ms_coco_2017/val2017"),
    "test2017.zip": ("http://images.cocodataset.org/zips/test2017.zip", "dataset/ms_coco_2017/test2017"),
    "annotations_trainval2017.zip": ("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", "dataset/ms_coco_2017/annotations")
}

# Télécharger et extraire uniquement si nécessaire
for filename, (url, extracted_folder) in files.items():
    zip_path = f"dataset/ms_coco_2017/{filename}"

    # Vérifier si les données sont déjà extraites
    if os.path.exists(extracted_folder):
        print(f"Skipping {filename}, already extracted.")
        continue

    # Télécharger uniquement si le ZIP n'existe pas
    if not os.path.exists(zip_path):
        print(f"Downloading {filename}...")
        os.system(f"wget -c {url} -P dataset/ms_coco_2017/")

    # Extraire si le ZIP est bien là
    if os.path.exists(zip_path):
        print(f"Extracting {filename}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("dataset/ms_coco_2017/")

        os.remove(zip_path)  # Supprimer le fichier ZIP après extraction
        print(f"Deleted {filename}")

print("Entire MS COCO 2017 dataset downloaded!")

# Process the dataset only if some .npy files are missing (and delete any partial files if needed)
files = [
    "dataset/x_train.npy",
    "dataset/x_train_filenames.npy",
    "dataset/x_test.npy",
    "dataset/x_test_filenames.npy"
]

# Checks if all required files exist
all_exist = all(os.path.exists(f) for f in files)

if not all_exist:
    # Deletes any that do exist (to avoid mismatched files)
    for file in files:
        try:
            os.remove(file)
        except OSError:
            pass  # File didn't exist or couldn't be deleted — we ignore it
    process_dataset()