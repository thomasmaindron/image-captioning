import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import os

def is_larger_than(image_path, target_size=(224, 224)):
    """
    Verify if an image is larger than the target size

    Args:
        image_path (str): Path to the image
        target_size (tuple): Minimal size of the image

    Returns:
        bool: True if the image is larger than the target size
    """
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image, channels=3)
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    return height >= target_size[0] and width >= target_size[1]

def process_image(image_path):
    """
    Processes a single image.

    Args:
        image_path (str): Path to the image.

    Returns:
        Image: The processed image object.
    """
    SIZE = (224, 224)
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image, channels=3) # Transform the image into a tensor
    image = tf.cast(image, tf.float32) # Needed otherwise the resizing changes the color of the images
    image = tf.image.resize(image, SIZE) 
    image = tf.cast(image, tf.uint8)
    return image

def save_image(image_tensor, output_path):
    """
    Saves a TensorFlow image tensor to a JPEG file.

    Args:
        image_tensor (tf.Tensor): A 3D image tensor
        output_path (str): Path where the image will be saved

    Returns:
        None
    """
    image_uint8 = tf.image.convert_image_dtype(image_tensor, dtype=tf.uint8)
    encoded = tf.io.encode_jpeg(image_uint8)
    tf.io.write_file(output_path, encoded)

def process_folder(input_folder, output_folder):
    """
    Processes a sample of all images in a folder and saves them after resizing

    Args:
        input_folder (str): Path to the folder containing the images
        output_folder (str): Path to the folder in which the processed images are saved

    Returns:
        None
    """
    SIZE = (224, 224)
    SAMPLE_RATIO = 0.2

    os.makedirs(output_folder, exist_ok=True)

    all_files = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if is_larger_than(os.path.join(input_folder, f), SIZE)
    ]
    sample_size = int(len(all_files) * SAMPLE_RATIO)
    sampled_files = random.sample(all_files, sample_size)

    for image_path in sampled_files:
        try:
            image = process_image(image_path)
            filename = os.path.basename(image_path)
            save_path = os.path.join(output_folder, filename)
            save_image(image, save_path)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    print(f"Processed {len(sampled_files)} images from {input_folder} to {output_folder}")

process_folder("dataset/ms_coco_2017/train2017", "dataset/x_train")
process_folder("dataset/ms_coco_2017/test2017", "dataset/x_test")