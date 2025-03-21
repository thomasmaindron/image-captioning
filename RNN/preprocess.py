import numpy as np
import tensorflow as tf

def preprocess_text(sentences):
    tokenizer = tf.Tokenizer()
    tokenizer.fit_on_texts(sentences)
    vocab_size = len(tokenizer.word_index) + 1

    sequences = tokenizer.texts_to_sequences(sentences)
    max_length = max(len(seq) for seq in sequences)
    sequences = tf.pad_sequences(sequences, maxlen=max_length, padding='post')

    return tokenizer, sequences, max_length, vocab_size
