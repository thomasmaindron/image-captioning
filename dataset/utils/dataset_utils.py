import numpy as np
import tensorflow as tf
import os
import random
from dataset.utils.image_utils import preprocess_image
from dataset.utils.caption_utils import load_captions, clean_captions
from tqdm import tqdm

def preprocess_folder(image_folder, output_base_dir, sample_ratio=0.2):
    """
    Preprocesses images in a folder and saves their features individually.
    
    Args:
        image_folder (str): Path to the folder containing images.
        output_base_dir (str): Base directory where features will be saved (e.g., "dataset/").
                                A subfolder will be created inside for the specific split.
        sample_ratio (float): Fraction of images to sample from the folder.
        size (tuple): Target size for resizing images.
    """
    encoder = tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top=False)
    
    all_files = [f for f in os.listdir(image_folder)]
    sample_size = int(len(all_files) * sample_ratio)
    random.seed(42) # Ensures reproducible results
    sampled_files = random.sample(all_files, sample_size)

    _, _, split_name_full = image_folder.split('/') # Ex: 'train2017'
    split_name_short = split_name_full.replace('2017', '').lower() # Ex: 'train'

    # Creation of the folder for the split features
    features_output_dir = os.path.join(output_base_dir, f"features_{split_name_short}")
    os.makedirs(features_output_dir, exist_ok=True)
    
    print(f"Extracting and saving features in: {features_output_dir}")

    for image_name in tqdm(sampled_files, desc=f"Extracting {split_name_short} features"):
        image_path = os.path.join(image_folder, image_name)

        preprocessed_image = preprocess_image(image_path)
        resnet_image = tf.keras.applications.resnet.preprocess_input(preprocessed_image)

        feature = encoder.predict(resnet_image, verbose=0)
        feature = feature.flatten()

        image_id, _ = os.path.splitext(image_name) # Ex: '0000000123'
        image_id = str(int(image_id)) # Ex: '123'

        # Save the feature in an individual .npy file
        np.save(os.path.join(features_output_dir, f"{image_id}.npy"), feature)

def preprocess_dataset():
    """
    Preprocesses images from the MS COCO 2017 dataset splits (train, validation, test)
    by extracting and saving their ResNet50 features into individual .npy files.
    """
    preprocess_folder(r"dataset/ms_coco_2017/train2017", r"dataset")
    preprocess_folder(r"dataset/ms_coco_2017/val2017", r"dataset")
    preprocess_folder(r"dataset/ms_coco_2017/test2017", r"dataset")

def load_split_coco(split = "train"):
    """
    Loads the preprocessed split of MS COCO 2017 dataset.

    Args:
        split (str): The dataset split to load. Can be 'train', 'val', or 'test'.

    Returns:
        tuple: (path_to_features_dir, captions_mapping) for 'train'/'val',
               or str: path_to_features_dir for 'test' (as test captions aren't available).

    Raises:
        ValueError: If an unsupported split is provided.
    """
    if split not in ["train", "val", "test"]:
        raise ValueError(f"Unsupported split: '{split}'. Choose from 'train', 'val', or 'test'.")

    features_dir = os.path.join("dataset", f"features_{split}")
    annotations_path = os.path.join("dataset", "ms_coco_2017", "annotations", f"captions_{split}2017.json")

    # For 'test' split, we only need the features directory, as captions are not publicly available
    if split == "test":
        return features_dir
    else:
        # Load and clean captions for 'train' and 'val'
        captions_mapping = load_captions(annotations_path)
        captions_mapping = clean_captions(captions_mapping)
        return (features_dir, captions_mapping)