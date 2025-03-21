import numpy as np
import tensorflow as tf
from preprocess import preprocess_text
from model import build_model

def train_model():
    sentences = [
        "Bonjour comment ça va",
        "Il fait beau aujourd'hui",
        "J'aime l'intelligence artificielle"
    ]

    tokenizer, sequences, max_length, vocab_size = preprocess_text(sentences)
    
    X = sequences[:, :-1]
    y = sequences[:, -1]
    y = tf.to_categorical(y, num_classes=vocab_size)

    model = build_model(vocab_size, max_length)
    model.fit(X, y, epochs=50, verbose=1)

    model.save("text_generator.h5")
    return tokenizer, max_length, model

if __name__ == "__main__":
    train_model()
