import tensorflow as tf
import json
import re # Regular expression module

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
        image_id = str(annotation['image_id'])
        caption = annotation['caption']
        
        if image_id not in mapping:
            mapping[image_id] = []
        
        mapping[image_id].append(caption)
    
    return mapping

def clean_captions(mapping):
    """
    Cleans and formats captions by applying a series of text preprocessing steps. Each caption undergoes:
    1. Lowercasing.
    2. Removal of non-alphanumeric characters (except common punctuation).
    3. Condensation of multiple spaces and stripping leading/trailing spaces.
    4. Addition of '<start>' and '<end>' tokens at the beginning and end, respectively.

    Args:
        mapping (dict): A dictionary where keys are image IDs (str) and values are lists of captions (list of str).

    Returns:
        dict: The modified dictionary with cleaned and formatted captions.
    """
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
    """
    Extracts all captions from a dictionary mapping image IDs to lists of captions.

    Args:
        mapping (dict): A dictionary where keys are image IDs (str) and values are
                        lists of associated captions (list of str).

    Returns:
        list[str]: A single list containing all caption strings from the input mapping.
    """
    all_captions = []
    for key in mapping:
        for caption in mapping[key]:
            all_captions.append(caption)
    return all_captions

def fit_tokenizer(all_captions):
    """
    Initializes and fits a Keras Tokenizer on a list of captions,
    then calculates the maximum sequence length and vocabulary size.

    The tokenizer is also saved to 'tokenizer.json' for future use
    in training or prediction scripts.

    Args:
        all_captions (list[str]): A flat list containing all caption strings
                                  from which the vocabulary should be built.

    Returns:
        tuple: A tuple containing:
            - tokenizer (tf.keras.preprocessing.text.Tokenizer): Fitted tokenizer object.
            - max_length (int): Max length of a sequence (always this length due to padding).
            - vocab_size (int): Number of words in the tokenizer.
    """
    tokenizer =  tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    max_length = max(len(caption.split()) for caption in all_captions)
    vocab_size = len(tokenizer.word_index) + 1

    # Save the tokenizer for future trainings or predictions
    tokenizer_json = tokenizer.to_json()
    with open('tokenizer.json', 'w', encoding='utf-8') as f:
        f.write(tokenizer_json)

    return tokenizer, max_length, vocab_size