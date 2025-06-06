import tensorflow as tf

def decoder(feature_vector_size, max_length, vocab_size):
    """
    Create decoder model for caption generation

    Args:
        feature_vector_size (int): Size of the feature vector (100352 for the flatten output of ResNet50).
        max_length (int): Max length of a sequence (always this length due to padding).
        vocab_size (int): Number of words in the tokenizer.

    Returns:
        tf.keras.Model: Decoder model.
    """
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