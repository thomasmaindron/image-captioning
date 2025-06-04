import numpy as np
import tensorflow as tf
from dataset.utils.caption_utils import fit_tokenizer, get_all_captions
from dataset.utils.dataset_utils import load_coco_dataset
import json

from dataset.utils.image_utils import preprocess_image

def index_to_word(token, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == token:
            return word
    return None
    
def caption_generation(model, feature, tokenizer, max_length):
    # Add start tag for generation process
    in_text = "<start>"

    # Iterate over the max length of sequence
    for _ in range(max_length):
        # Encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]

        # Pad the sequence
        sequence = tf.keras.preprocessing.sequence.pad_sequences([sequence], max_length)

        # Predict the next word
        predicted_vector = model.predict([feature, sequence], verbose=0)

        # Get index with hight probability
        predicted_token = np.argmax(predicted_vector) # yhat

        # Convert token to word
        word = index_to_word(predicted_token, tokenizer)

        # Stop if word not found
        if word is None:
            break

        # Append word as input for generating next word
        predicted_sentence += " " + word

        # Stop if we reach end tag
        if word == "<end>":
            break
    
    return predicted_sentence

def image_captioning(image):
    # Load the tokenizer
    with open('tokenizer.json') as f:
        data = json.load(f)
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)
    max_length = 0
    vocab_size = len(tokenizer.word_index) + 1

    # Extract the feature from the image
    encoder = encoder()
    preprocessed_image = preprocess_image(image)
    # Apply ResNet-specific preprocessing (e.g., pixel scaling)
    resnet_image = tf.keras.applications.resnet.preprocess_input(preprocessed_image)
    # Extract features
    feature = encoder.predict(resnet_image, verbose=0)
    feature = feature.flatten() # Flatten the 4D feature tensor into a 1D vector

    feature_vector_size = 100352 # 7 x 7 x 2048 : output of ResNet50
    decoder = decoder(feature_vector_size, vocab_size, max_length)
    predicted_sentence = caption_generation(decoder, feature, tokenizer, max_length)
    return predicted_sentence
