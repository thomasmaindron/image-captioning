import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def encoder():
    model = tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top=False)
    return model

def decoder(feature_vector_size, max_length, vocab_size):

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