import numpy as np
import tensorflow as tf
import random

class CocoDataGenerator(tf.keras.utils.Sequence):
    """
    Data generator for COCO dataset with random caption selection
    """
    def __init__(self, x_images, y_multi_captions, tokenizer, batch_size=32, 
                max_length=16, shuffle=True):
        """
        Initialize the generator
        Args:
            x_images (np.array): Images array
            y_multi_captions (list): List of caption lists for each image
            tokenizer (Tokenizer): Keras tokenizer for processing text
            batch_size (int): Batch size
            max_length (int): Maximum sequence length
            shuffle (bool): Whether to shuffle the data after each epoch
        """
        self.x_images = x_images
        self.y_multi_captions = y_multi_captions
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.shuffle = shuffle
        self.indices = np.arange(len(self.x_images))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        """
        Returns the number of batches per epoch
        """
        return int(np.ceil(len(self.x_images) / self.batch_size))
    
    def __getitem__(self, idx):
        """
        Returns one batch of data
        """
        # Get batch indices
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        # Get batch images
        batch_x = self.x_images[inds]
        
        # Select a random caption for each image in the batch
        batch_captions = []
        for i in inds:
            captions = self.y_multi_captions[i]
            if captions:
                batch_captions.append(random.choice(captions))
            else:
                batch_captions.append("")
        
        # Convert captions to sequences
        sequences = self.tokenizer.texts_to_sequences(batch_captions)
        batch_y = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=self.max_length, padding='post')
        
        return batch_x, batch_y
    
    def on_epoch_end(self):
        """
        Called at the end of each epoch
        """
        if self.shuffle:
            np.random.shuffle(self.indices)

def prepare_caption_data(train_filenames, test_filenames, 
                         train_annotations_path, test_annotations_path,
                         get_image_id_fn, load_captions_fn):
    """
    Prepare caption data for training
    Args:
        train_filenames (np.array): Array of training image filenames
        test_filenames (np.array): Array of test image filenames
        train_annotations_path (str): Path to training annotations file
        test_annotations_path (str): Path to test annotations file
        get_image_id_fn (function): Function to get image ID from filename
        load_captions_fn (function): Function to load captions from annotation file
    Returns:
        tuple: (y_train_multi, y_test_multi) where each element is a list of caption lists
    """
    # Charger les captions d'entraînement et de test
    train_captions_dict = load_captions_fn(train_annotations_path)
    test_captions_dict = load_captions_fn(test_annotations_path)
    
    # Associer les captions aux images d'entraînement
    y_train_multi = []
    for filename in train_filenames:
        image_id = get_image_id_fn(filename)
        captions = train_captions_dict.get(image_id, [])
        y_train_multi.append(captions)
    
    # Associer les captions aux images de test
    y_test_multi = []
    for filename in test_filenames:
        image_id = get_image_id_fn(filename)
        captions = test_captions_dict.get(image_id, [])
        y_test_multi.append(captions)
    
    return y_train_multi, y_test_multi