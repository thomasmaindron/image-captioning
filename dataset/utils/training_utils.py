import numpy as np
import tensorflow as tf
import random
import os

def data_generator(features_dir, captions_mapping, data_keys, tokenizer, max_length, vocab_size):
    """
    Generate data for training.
    
    Args:
        features_dir (str): Path to the folder containing the .npy feature files
        captions_mapping (dict): Dictionary image_id -> [captions].
        data_keys (list): List of all images IDs.
        tokenizer (tf.keras.preprocessing.text.Tokenizer): Fitted tokenizer object.
        max_length (int): Max length of a sequence (always this length due to padding).
        vocab_size (int): Number of words in the tokenizer.
    """
    while True: # Infinite loop
        random.shuffle(data_keys) # Shuffle the keys for each loop (avoids overfitting)

        for key in data_keys:
            # Load the feature from the .npy file
            image_feature = np.load(os.path.join(features_dir, f"{key}.npy"))

            captions = captions_mapping[key] # get the list of captions corresponding to the image

            # We go through all the captions
            for caption in captions:
                seq = tokenizer.texts_to_sequences([caption])[0]

                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = tf.keras.utils.pad_sequences([in_seq], maxlen=max_length)[0] # input sequence (teacher forcing)
                    out_seq = tf.keras.utils.to_categorical([out_seq], num_classes=vocab_size)[0] # target word
                    
                    yield (image_feature, in_seq), out_seq