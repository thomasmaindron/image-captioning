import numpy as np
import tensorflow as tf
import random

# Create data generator to get data in batch (avoids session crash)
def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size=32):
    X1 = [] # features
    X2 = [] # sequences
    y = [] # target

    n = 0

    while 1:
        for key in data_keys:
            n += 1
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
                    out_seq = tf.keras.utils.to_categorical([out_seq],num_classes=vocab_size)[0] # one-hot vector
                    # Store the sequences
                    X1.append(features[key][0])
                    X2.append(in_seq)
                    y.append(out_seq)

            if n == batch_size:
                X1, X2, y = np.array(X1), np.array(X2), np.array(y)
                yield [X1, X2], y

                # Reinitialization
                X1, X2, y = list(), list(), list()
                n = 0