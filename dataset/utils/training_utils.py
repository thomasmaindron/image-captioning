import numpy as np
import tensorflow as tf
import random

# Create data generator to get data in batch (avoids session crash by reducing VRAM saturation)
def data_generator(features, mapping, data_keys, tokenizer, max_length, vocab_size, batch_size=32):
    X1 = [] # features
    X2 = [] # sequences
    y = [] # target

    data_keys = list(data_keys) 

    while True:

        random.shuffle(data_keys) # shuffle the keys for each epoch to avoid overfitting

        for key in data_keys:
            captions = mapping[key]

            # Process each caption
            for caption in captions:
                seq = tokenizer.texts_to_sequences([caption])[0] # tokenized caption

                # Split the sequence into X, y pairs
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i] # previous words, next word
                    # Pad input sequence 
                    in_seq =  tf.keras.utils.pad_sequences([in_seq], maxlen=max_length)[0]
                    # Encode output sequence
                    out_seq = tf.keras.utils.to_categorical([out_seq], num_classes=vocab_size)[0] # one-hot vector
                    # Store the sequences
                    X1.append(features[key]) # add the feature of the current image to the batch
                    X2.append(in_seq) # add the previous words of the current caption to the batch
                    y.append(out_seq) # add the target word of the current caption to the batch

            if len(X1) == batch_size:
                X1, X2, y = np.array(X1), np.array(X2), np.array(y)
                yield [X1, X2], y

                # Reinitialization
                X1, X2, y = list(), list(), list()
        
        # After going through all the images in this epoch (even though it didn't reach batch_size)
        if len(X1) > 0:
            X1, X2, y = np.array(X1), np.array(X2), np.array(y)
            yield [X1, X2], y

            # Reinitialization
            X1, X2, y = list(), list(), list()