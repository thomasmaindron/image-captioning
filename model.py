import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def decoder(feature_vector_size, vocab_size, max_length):

    # Image feature layers
    inputs1 = tf.keras.layers.Input(shape=(feature_vector_size,))
    fe1 = tf.keras.layers.Dropout(0.4)(inputs1)
    fe2 = tf.keras.layers.Dense(256, activation='relu')(fe1)

    # Sequence layers
    inputs2 = tf.keras.layers.Input(shape=(max_length,))
    se1 = tf.keras.layers.Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = tf.keras.layers.Dropout(0.4)(se1)
    se3 = tf.keras.layers.LSTM(256)(se2)

    # Decoder model
    decoder1 = tf.keras.layers.add([fe2, se3])
    decoder2 = tf.keras.layers.Dense(256, activation='relu')(decoder1)
    outputs = tf.keras.layers.Dense(vocab_size, activation='softmax')(decoder2)

    model = tf.keras.models.Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model


# class ImageCaptioning:
#     def __init(self, embedding_size=256, units=512, vocab_size=10000, max_caption_len=50):
#         self.embedding_size = embedding_size
#         self.units = units
#         self.vocab_size = vocab_size
#         self.max_caption_len = max_caption_len
#         self.encoder = self.encoder_model()
#         self.decoder = self.decoder_model()

#     def encoder_model(self):
#         #Create the ResNet50 as backbone
#         backbone = tf.keras.applications.ResNet50(
#         weights='imagenet',
#         include_top=False,
#         input_shape=(224, 224, 3)
#         )
#         #freeze the weight of the backbone in order to not change them during training
#         for layer in backbone.layers:
#             layer.trainable = False
#         model = tf.keras.Sequential([
#         tf.keras.layers.InputLayer(shape=(224, 224, 3)),
#         backbone,
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(self.embedding_size, activation='relu')
#         ])
#         return model
    
#     def decoder_model(self):
#         #Entrée pour les caractéristiques de l'image
#         inputs1 = tf.keras.layers.Input(shape=(self.embedding_size,))
#         fc_input = tf.keras.layers.Dropout(0.5)(inputs1)
#         fc_input = tf.keras.layers.Dense(self.units, activation='relu')(fc_input)
        
#         # Entrée pour la séquence de mots
#         inputs2 = tf.keras.layers.Input(shape=(self.max_caption_len,))
#         LSTM_input = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size, mask_zero=True)(inputs2)
#         LSTM_input = tf.keras.layers.LSTM(self.units, return_sequences=True)(LSTM_input)
#         LSTM_input = tf.keras.layers.LSTM(self.units)(LSTM_input)

#         # Fusion des deux branches
#         decoder = tf.keras.layers.add([fc_input, LSTM_input])
#         decoder = tf.keras.layers.Dense(256, activation='relu')(decoder)
#         outputs = tf.keras.layers.Dense(self.vocab_size, activation='softmax')(decoder)

#         # Construction du modèle complet
#         model = tf.keras.Model(inputs=[inputs1, inputs2], outputs=outputs)
#         return model