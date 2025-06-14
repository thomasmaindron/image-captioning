import os
import zipfile
from utils.dataset_utils import preprocess_dataset


if __name__ == "__main__":
    # Destination folder
    os.makedirs("dataset/ms_coco_2017", exist_ok=True)

    # Link to the COCO Captions files and their folders
    files = {
        "train2017.zip": ("http://images.cocodataset.org/zips/train2017.zip", "dataset/ms_coco_2017/train2017"),
        "val2017.zip": ("http://images.cocodataset.org/zips/val2017.zip", "dataset/ms_coco_2017/val2017"),
        "test2017.zip": ("http://images.cocodataset.org/zips/test2017.zip", "dataset/ms_coco_2017/test2017"),
        "annotations_trainval2017.zip": ("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", "dataset/ms_coco_2017/annotations")
    }

    # Download and extract only if necessary
    for filename, (url, extracted_folder) in files.items():
        zip_path = f"dataset/ms_coco_2017/{filename}"

        # Check if data is already is here
        if os.path.exists(extracted_folder):
            print(f"Skipping {filename}, already extracted.")
            continue

        # Download only if zip isn't here
        if not os.path.exists(zip_path):
            print(f"Downloading {filename}...")
            os.system(f"wget -c {url} -P dataset/ms_coco_2017/")

        # Extract if the zip is here
        if os.path.exists(zip_path):
            print(f"Extracting {filename}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall("dataset/ms_coco_2017/")

            os.remove(zip_path)  # Delete the zip file after extraction
            print(f"Deleted {filename}")

    print("Entire MS COCO 2017 dataset downloaded!")

    # Process the dataset only if some .npz files are missing (and delete any partial files if needed)
    files = [
        "dataset/x_train.npz",
        "dataset/x_val.npz",
        "dataset/x_test.npz"
    ]

    # Check if all required files exist
    all_exist = all(os.path.exists(f) for f in files)

    if not all_exist:
        # Delete any that do exist (to avoid mismatched files)
        for file in files:
            try:
                os.remove(file)
            except OSError:
                pass  # File didn't exist or couldn't be deleted — we ignore it

    # Preprocess only if the features folders don't exist
    if not os.path.exists(r"dataset/features_train") or not os.path.exists(r"dataset/features_val") or not os.path.exists(r"dataset/features_test"):
        preprocess_dataset()