import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
embedding_size = 256
units = 512
vocab_size = 10000 #10 000 mots de vocabulaire différents
max_caption_len = 50

# CHARGEMENT DU TOKENIZER
with open(r".\dataset\ms_coco_2017\tokenizer.json") as f:
    tokenizer_data = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_data)

def preprocess_image(img_path):
    image = tf.io.read_file(img_path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = image / 255.0
    image = tf.expand_dims(image, 0)
    return image

def encoder_model(embedding_size):
    # Create the ResNet50 as backbone
    backbone = tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    #freeze the weight of the backbone
    for layer in backbone.layers:
        layer.trainable = False
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(224, 224, 3)),
        backbone,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(embedding_size, activation='relu')
    ])
    return model

def decoder_model(vocab_size, embedding_size, max_caption_len):
    # Entrée pour les caractéristiques de l'image
    inputs1 = tf.keras.layers.Input(shape=(embedding_size,))
    fc_input = tf.keras.layers.Dropout(0.5)(inputs1)
    fc_input = tf.keras.layers.Dense(units, activation='relu')(fc_input)
    
    # Entrée pour la séquence de mots
    inputs2 = tf.keras.layers.Input(shape=(max_caption_len,))
    LSTM_input = tf.keras.layers.Embedding(vocab_size, embedding_size, mask_zero=True)(inputs2)
    LSTM_input = tf.keras.layers.LSTM(units)(LSTM_input)
    
    # Fusion des deux branches
    decoder = tf.keras.layers.add([fc_input, LSTM_input])
    decoder = tf.keras.layers.Dense(256, activation='relu')(decoder)
    outputs = tf.keras.layers.Dense(vocab_size, activation='softmax')(decoder)
    
    # Construction du modèle complet
    model = tf.keras.Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model

img_path = r".\dataset\ms_coco_2017\test2017\000000000001.jpg"
encoder = encoder_model(embedding_size)
decoder = decoder_model(vocab_size, embedding_size, max_caption_len) 
# # Prétraitement de l'image et extraction des caractéristiques
# image = preprocess_image(img_path)
# vector = encoder(image)
# decoder = decoder_model()  
# encoder.summary()
# decoder.summary()
def ind_to_word(integer, tokenizer):
    for word, index in tokenizer: # ou tokenizer.word_index.items() à voir
        if index == integer:
            return word
    return None
def caption_generation(image_path, tokenizer, encoder, decoder, max_length=max_caption_len):
        image = preprocess_image(image_path)
        features = encoder(image)
        text = "start"
        for i in range(max_length):
            sequence=tokenizer.texts_to_sequences([text])[0]
            sequence=pad_sequences([sequence], maxlen=max_length)
            y_pred=decoder.predict([features, sequence])
            ind_pred = np.argmax(y_pred)
            word = ind_to_word(ind_pred, tokenizer)
            if word is None:
                break
            text += " " + word
            if word == "end":
                break
        return text





