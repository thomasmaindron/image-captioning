import numpy as np
import tensorflow as tf
import os
import random
from dataset.utils.image_utils import preprocess_image
from dataset.utils.caption_utils import load_captions, clean_captions

def preprocess_folder(image_folder, output_file, sample_ratio=0.2, size=(224, 224)):
    encoder = tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top=False)
    features = {}

    # Only the names of the images
    all_files = [f for f in os.listdir(image_folder)]
    sample_size = int(len(all_files) * sample_ratio)
    random.seed(42) # Ensures reproducible results
    sampled_files = random.sample(all_files, sample_size)

    for image_name in(sampled_files):
        # Construct the full path for the current image
        image_path = os.path.join(image_folder, image_name)

        image = preprocess_image(image_path)

        # Apply ResNet-specific preprocessing (e.g., pixel scaling)
        image = tf.keras.applications.resnet.preprocess_input(image)

        # Extract features
        feature = encoder.predict(image, verbose=0)
        feature = feature.flatten() # Flatten the 4D feature tensor into a 1D vector

        image_id, _ = image_name.split('.')
        image_id = int(image_id) # Conversion to int in order to match how ids are stored in COCO

        # Store feature
        features[image_id] = feature

    # Save all the extracted features to a single compressed NumPy file
    np.savez_compressed(output_file, features) 

def preprocess_dataset():
    """
    Preprocesses a sample of images from the MS COCO 2017 train and test folders
    
    Args:
        None

    Returns:
        None
    """
    preprocess_folder(r"dataset/ms_coco_2017/train2017", r"dataset/x_train.npz")
    preprocess_folder(r"dataset/ms_coco_2017/test2017", r"dataset/x_test.npz")

def load_coco_dataset():
    """
    Loads the preprocessed MS COCO 2017 dataset from .npy files

    Args:
        None

    Returns:
        tuple: ((x_train, x_test), (y_train, y_test))
    """
    # Load preprocessed images
    x_train = np.load("dataset/x_train.npz")
    x_test = np.load("dataset/x_test.npz")

    # Path to the annotations
    train_annotations = "dataset/ms_coco_2017/annotations/captions_train2017.json"
    test_annotations = "dataset/ms_coco_2017/annotations/captions_val2017.json"
    
    # Get the captions from the annotations files
    y_train = load_captions(train_annotations)
    y_test = load_captions(test_annotations)

    # Clean the captions
    y_train = clean_captions(y_train)
    y_test = clean_captions(y_test)
    
    return (x_train, x_test), (y_train, y_test)