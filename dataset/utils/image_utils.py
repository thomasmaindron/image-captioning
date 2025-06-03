import numpy as np
import tensorflow as tf
import os

def preprocess_image(image_path, size=(224, 224)):
    """
    Preprocesses a single image by resizing

    Args:
        image_path (str): Path to the image
        size (tuple): Target size for resizing the image

    Returns:
        tf.Tensor: The preprocessed image tensor
    """
    # Load the image from file
    image = tf.keras.utils.load_img(image_path, target_size=size)
    # Convert image to array
    image = tf.keras.utils.img_to_array(image)
    # Reshape data for model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    return image