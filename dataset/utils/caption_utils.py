import numpy as np
import tensorflow as tf
import json
import os

def load_captions(annotation_file):
    """
    Loads captions from COCO annotation file
    Args:
        annotation_file (str): Path to the COCO annotation file
    Returns:
        dict: Dictionary mapping image_id to list of captions
    """
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    # Create a dictionary image_id -> [captions]
    image_captions = {}
    
    # For each annotation in the file
    for annotation in annotations['annotations']:
        image_id = annotation['image_id']
        caption = annotation['caption']
        
        if image_id not in image_captions:
            image_captions[image_id] = []
        
        image_captions[image_id].append(caption)
    
    return image_captions

def get_image_id_from_filename(filename):
    """
    Extract image_id from COCO filename format (000000000001.jpg -> 1)
    Args:
        filename (str): Filename of the image
    Returns:
        int: Image ID
    """
    base_name = os.path.basename(filename)
    image_id = int(os.path.splitext(base_name)[0])
    return image_id

def prepare_caption_data(train_filenames, test_filenames, 
                         train_annotations_path, test_annotations_path):
    """
    Prepare caption data for training
    Args:
        train_filenames (np.array): Array of training image filenames
        test_filenames (np.array): Array of test image filenames
        train_annotations_path (str): Path to training annotations file
        test_annotations_path (str): Path to test annotations file
    Returns:
        tuple: (y_train_multi, y_test_multi) where each element is a list of caption lists
    """
    # Charger les captions d'entraînement et de test
    train_captions_dict = load_captions(train_annotations_path)
    test_captions_dict = load_captions(test_annotations_path)
    
    # Associer les captions aux images d'entraînement
    y_train_multi = []
    for filename in train_filenames:
        image_id = get_image_id_from_filename(filename)
        captions = train_captions_dict.get(image_id, [])
        y_train_multi.append(captions)
    
    # Associer les captions aux images de test
    y_test_multi = []
    for filename in test_filenames:
        image_id = get_image_id_from_filename(filename)
        captions = test_captions_dict.get(image_id, [])
        y_test_multi.append(captions)
    
    return y_train_multi, y_test_multi

def create_tokenizer(caption_lists, num_words=10000):
    """
    Create and fit a tokenizer on all available captions
    Args:
        caption_lists (list): List of caption lists
        num_words (int): Maximum number of words to keep
    Returns:
        Tokenizer: Fitted tokenizer
    """
    # Aplatir toutes les captions en une seule liste
    all_captions = []
    for captions in caption_lists:
        all_captions.extend(captions)
    
    # Créer et entraîner le tokenizer
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=num_words, oov_token="<UNK>")
    tokenizer.fit_on_texts(all_captions)
    
    return tokenizer

def save_tokenizer(tokenizer, filename):
    """
    Save tokenizer to a JSON file
    Args:
        tokenizer (Tokenizer): Keras tokenizer
        filename (str): Path to save the tokenizer
    """
    tokenizer_json = {
        'word_index': tokenizer.word_index,
        'index_word': tokenizer.index_word,
        'word_counts': tokenizer.word_counts,
        'document_count': tokenizer.document_count,
        'num_words': tokenizer.num_words
    }
    with open(filename, 'w') as f:
        json.dump(tokenizer_json, f)

def load_tokenizer(filename):
    """
    Load tokenizer from a JSON file
    Args:
        filename (str): Path to the saved tokenizer
    Returns:
        Tokenizer: Loaded tokenizer
    """
    with open(filename, 'r') as f:
        tokenizer_json = json.load(f)
    
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.word_index = tokenizer_json['word_index']
    tokenizer.index_word = {int(k): v for k, v in tokenizer_json['index_word'].items()}
    tokenizer.word_counts = tokenizer_json['word_counts']
    tokenizer.document_count = tokenizer_json['document_count']
    tokenizer.num_words = tokenizer_json['num_words']
    
    return tokenizer

def preprocess_captions(captions_list, start_token="<start>", end_token="<end>"):
    """
    Add start and end tokens to captions
    Args:
        captions_list (list): List of captions lists
        start_token (str): Token to add at the beginning of each caption
        end_token (str): Token to add at the end of each caption
    Returns:
        list: List of preprocessed captions lists
    """
    preprocessed = []
    for captions in captions_list:
        processed_captions = []
        for caption in captions:
            processed_caption = f"{start_token} {caption} {end_token}"
            processed_captions.append(processed_caption)
        preprocessed.append(processed_captions)
    return preprocessed