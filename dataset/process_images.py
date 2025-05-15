import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import os

def is_larger_than(image_path, size=(128, 128)):
    """
    Checks if an image is larger than or equal to the target size

    Args:
        image_path (str): Path to the image
        target_size (tuple): Minimum required size of the image

    Returns:
        bool: True if the image is larger than or equal to the target size
    """
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image, channels=3)
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    return height >= size[0] and width >= size[1]

def process_image(image_path, size=(128, 128)):
    """
    Processes a single image by resizing

    Args:
        image_path (str): Path to the image
        size (tuple): Target size for resizing the image

    Returns:
        tf.Tensor: The processed image tensor
    """
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image, channels=3) # Transform the image into a tensor
    image = tf.cast(image, tf.float32) # Needed otherwise the resizing changes the color of the images
    image = tf.image.resize(image, size)
    image = tf.cast(image, tf.uint8)
    return image

def process_folder(input_folder, output_file, sample_ratio=0.2, size=(128, 128)):
    """
    Processes a random sample of images from a folder and saves them as a NumPy array

    Args:
        input_folder (str): Path to the folder containing images
        output_file (str): Path to the output .npy file
        sample_ratio (float): Proportion of images to randomly sample from the folder
        size (tuple): Target size for image resizing

    Returns:
        None
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    all_files = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if is_larger_than(os.path.join(input_folder, f), size)
    ]
    
    sample_size = int(len(all_files) * sample_ratio)

    random.seed(42)
    sampled_files = random.sample(all_files, sample_size)
    
    # Preallocate a NumPy array to store all processed images
    images_array = np.zeros((sample_size, size[0], size[1], 3), dtype=np.uint8)
    
    # Store filenames for reference
    filenames = []
    
    for i, image_path in enumerate(sampled_files):
        try:
            image = process_image(image_path)
            images_array[i] = image
            filenames.append(os.path.basename(image_path))
            
            # Show progress every 100 images
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{sample_size} images")
                
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    # Save the processed images to a NumPy file
    np.save(output_file, images_array)
    
    # Save the list of filenames
    np.save(f"{os.path.splitext(output_file)[0]}_filenames.npy", np.array(filenames))
    
    print(f"Processed and saved {len(images_array)} images to {output_file}")

def process_dataset():
    """
    Processes a sample of images from the MS COCO 2017 train and test folders
    
    Args:
        None

    Returns:
        None
    """
    process_folder("dataset/ms_coco_2017/train2017", "dataset/x_train.npy")
    process_folder("dataset/ms_coco_2017/test2017", "dataset/x_test.npy")

def load_coco_dataset():
    """
    Loads the preprocessed MS COCO 2017 dataset from .npy files

    Args:
        None

    Returns:
        tuple: ((x_train, y_train), (x_test, y_test))
    """
    x_train = np.load("dataset/x_train.npy").astype(np.float32) / 255.0
    y_train = 0
    x_test = np.load("dataset/x_test.npy").astype(np.float32) / 255.0
    y_test = 0
    return (x_train, y_train), (x_test, y_test)

process_dataset()