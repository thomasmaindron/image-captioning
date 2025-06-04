import numpy as np
import tensorflow as tf
import os
import random
from dataset.utils.image_utils import preprocess_image
from dataset.utils.caption_utils import load_captions, clean_captions
from tqdm import tqdm

def preprocess_folder(image_folder, output_file, sample_ratio=0.2, size=(224, 224)):
    model = tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top=False)
    features = {}

    # Only the names of the images
    all_files = [f for f in os.listdir(image_folder)]
    sample_size = int(len(all_files) * sample_ratio)
    random.seed(42) # Ensures reproducible results
    sampled_files = random.sample(all_files, sample_size)

    for image_name in tqdm(sampled_files, desc="Extracting features"):
        # Construct the full path for the current image
        image_path = os.path.join(image_folder, image_name)

        preprocessed_image = preprocess_image(image_path)

        # Apply ResNet-specific preprocessing (e.g., pixel scaling)
        resnet_image = tf.keras.applications.resnet.preprocess_input(preprocessed_image)

        # Extract features
        feature = model.predict(resnet_image, verbose=0)
        feature = feature.flatten() # Flatten the 4D feature tensor into a 1D vector

        image_id, _ = image_name.split('.')
        image_id = int(image_id) # Conversion to int in order to match how ids are stored in COCO (remove all the 0 at the beginning)
        image_id = str(image_id) # Conversion back to str so it can save as .npz

        # Store feature
        features[image_id] = feature

    # Save all the extracted features to a single compressed NumPy file
    np.savez_compressed(output_file, **features) 

def preprocess_dataset():
    """
    Preprocesses a sample of images from the MS COCO 2017 train and test folders
    
    Args:
        None

    Returns:
        None
    """
    preprocess_folder(r"dataset/ms_coco_2017/train2017", r"dataset/x_train.npz")
    preprocess_folder(r"dataset/ms_coco_2017/val2017", r"dataset/x_val.npz")
    preprocess_folder(r"dataset/ms_coco_2017/test2017", r"dataset/x_test.npz")

def load_training_split_coco():
    """
    Loads the preprocessed training split of MS COCO 2017 dataset from .npz files

    Args:
        None

    Returns:
        tuple: (x_train, y_train)
    """
    # Load preprocessed images
    x_train = np.load("dataset/x_train.npz")

    # Path to the annotations
    train_annotations = "dataset/ms_coco_2017/annotations/captions_train2017.json"
    
    # Get the captions from the annotations files
    y_train = load_captions(train_annotations)

    # Clean the captions
    y_train = clean_captions(y_train)
    
    return (x_train, y_train)

def load_validation_split_coco():
    """
    Loads the preprocessed validation split of MS COCO 2017 dataset from .npz files

    Args:
        None

    Returns:
        tuple: (x_val, y_val)
    """
    # Load preprocessed images
    x_val = np.load("dataset/x_val.npz")

    # Path to the annotations
    val_annotations = "dataset/ms_coco_2017/annotations/captions_val2017.json"
    
    # Get the captions from the annotations files
    y_val = load_captions(val_annotations)

    # Clean the captions
    y_val = clean_captions(y_val)


    return (x_val, y_val)

def load_testing_split_coco():
    """
    Loads the preprocessed testing split of MS COCO 2017 dataset from .npz files

    Args:
        None

    Returns:
        dict: x_test
    """
    # Load preprocessed images
    x_test = np.load("dataset/x_test.npz")
    
    return x_test