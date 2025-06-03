import numpy as np
import tensorflow as tf
import json
import re # Import the regular expression module

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
    mapping = {}
    
    # For each annotation in the file
    for annotation in annotations['annotations']:
        image_id = annotation['image_id']
        caption = annotation['caption']
        
        if image_id not in mapping:
            mapping[image_id] = []
        
        mapping[image_id].append(caption)
    
    return mapping

def clean_captions(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            # Take one caption at a time
            caption = captions[i]
            # Convert to lowercase
            caption = caption.lower()
            # Delete digits, speical chars...
            caption = re.sub(r'[^a-z0-9.,!?"\'\s]', '', caption)
            # Delete additional spaces
            caption = re.sub(r'\s+', ' ', caption).strip()
            # Add start and end tags to the caption
            caption = "<start> " + caption + " <end>"
            
            captions[i] = caption
    return mapping

def get_all_captions(mapping):
    all_captions = []
    for key in mapping:
        for caption in mapping[key]:
            all_captions.append(caption)
    return all_captions

def fit_tokenizer(all_captions):
    tokenizer =  tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    vocab_size = len(tokenizer.word_index) + 1
    max_length = max(len(caption.split()) for caption in all_captions)
    return tokenizer, vocab_size, max_length