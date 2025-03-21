import tensorflow as tf

def build_model(vocab_size, max_length):
    model = tf.Keras.Sequential([
        tf.Keras.layers.Embedding(input_dim=vocab_size, output_dim=100, input_length=max_length-1),
        tf.Keras.layers.LSTM(128, return_sequences=True),
        tf.Keras.layers.LSTM(128),
        tf.Keras.layers.Dense(128, activation='relu'),
        tf.Keras.layers.Dense(vocab_size, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
