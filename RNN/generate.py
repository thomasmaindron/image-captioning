import numpy as np
import tensorflow as tf
from preprocess import preprocess_text

def generate_sentence(seed_text, model, tokenizer, max_length):
    for _ in range(10):  # Générer 10 mots
        sequence = tokenizer.texts_to_sequences([seed_text])
        sequence = tf.pad_sequences(sequence, maxlen=max_length-1, padding='pre')
        predicted = model.predict(sequence, verbose=0)
        next_word_index = np.argmax(predicted)
        next_word = tokenizer.index_word.get(next_word_index, '')

        if not next_word:
            break
        seed_text += ' ' + next_word

    return seed_text

if __name__ == "__main__":
    model = tf.keras.models.load_model("text_generator.h5")
    sentences = ["Bonjour comment ça va", "Il fait beau aujourd'hui", "J'aime l'intelligence artificielle"]
    tokenizer, _, max_length, _ = preprocess_text(sentences)

    print(generate_sentence("Bonjour", model, tokenizer, max_length))
