import numpy as np
import tensorflow as tf
import os

def is_larger_than(image_path, size=( 224,  224)):
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

def preprocess_image(image_path, size=(224,  224)):
    """
    Preprocesses a single image by resizing

    Args:
        image_path (str): Path to the image
        size (tuple): Target size for resizing the image

    Returns:
        tf.Tensor: The preprocessed image tensor
    """
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image, channels=3) # Transform the image into a tensor
    image = tf.cast(image, tf.float32) # Needed otherwise the resizing changes the color of the images
    image = tf.image.resize(image, size)
    image = tf.cast(image, tf.uint8)
    return image.numpy()

def prepare_image_for_model(img_path, size=( 224,  224)):
        image=preprocess_image(img_path, size)
        image = tf.cast(image, tf.float32)/ 255.0
        image = tf.expand_dims(image, 0)
        return image