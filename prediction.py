import numpy as np
import tensorflow as tf
from dataset.utils.image_utils import preprocess_image

def index_to_word(token, tokenizer):
    """
    Convert a numerical token (index) back to its corresponding word using the tokenizer.

    Args:
        token (int): The numerical index (token) to convert.
        tokenizer (tf.keras.preprocessing.text.Tokenizer): Tokenizer object
                                                          
    Returns:
        str | None: The word corresponding to the token if found, otherwise None.
    """
    for word, index in tokenizer.word_index.items():
        if index == token:
            return word
    return None
    
def generate_caption_from_feature(image_feature, decoder, tokenizer, max_length):
    """
    Generate a caption for a given feature.
    
    Args:
        image_feature (np.array): Feature vectore of the image.
        decoder (tf.keras.Model): Trained decoder model.
        tokenizer (tf.keras.preprocessing.text.Tokenizer): Fitted tokenizer object.
        max_length (int): Max length of a sequence (always this length due to padding).
        
    Returns:
        str: Predicted caption.
    """
    in_text = "startcaption"
    
    sequence = tokenizer.texts_to_sequences([in_text])[0]
    sequence = tf.keras.utils.pad_sequences([sequence], maxlen=max_length)[0]
    
    # Add 1 dimension to work with Keras functions
    image_feature = image_feature.reshape(1, -1)
    sequence = sequence.reshape(1, -1)

    for i in range(max_length - 1):
        predicted_vector = decoder.predict([image_feature, sequence], verbose=0) # one-hot vector
        
        predicted_index = np.argmax(predicted_vector) # token predicted
        predicted_word = tokenizer.index_word.get(predicted_index, None) # word of the corresponding token
        
        if predicted_word:
            in_text += " " + predicted_word
        
        if predicted_word == "endcaption":
            break
            
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = tf.keras.utils.pad_sequences([sequence], maxlen=max_length)[0]
        sequence = sequence.reshape(1, -1)

    return in_text

def generate_caption_from_image(raw_image_path, encoder, decoder):
    """
    Generate a caption for a given image.

    Args:
        raw_image_path (str): Path to the image file.
        encoder (tf.keras.Model): Trained encoder model.
        decoder (tf.keras.Model): Trained decoder model.

    Returns:
        str: Predicted caption.
    """
    # Load the tokenizer
    with open('tokenizer.json', 'r', encoding='utf-8') as f:
        data = f.read()
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)
    max_length = 49 # HARDCODED : THIS MIGHT CHANGE IF YOU CREATED A NEW TOKENIZER !!!

    # Extract the feature from the image
    preprocessed_image = preprocess_image(raw_image_path)
    # Apply ResNet-specific preprocessing (e.g., pixel scaling)
    resnet_image = tf.keras.applications.resnet.preprocess_input(preprocessed_image)
    # Extract features
    feature = encoder.predict(resnet_image, verbose=0)
    image_feature = feature.flatten() # Flatten the 4D feature tensor into a 1D vector

    predicted_caption = generate_caption_from_feature(image_feature, decoder, tokenizer, max_length)
    return predicted_caption