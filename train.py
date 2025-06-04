import numpy as np
import tensorflow as tf
from dataset.utils.caption_utils import fit_tokenizer, get_all_captions
from dataset.utils.dataset_utils import load_coco_dataset
from dataset.utils.training_utils import data_generator
from model import decoder

def index_to_word(token, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == token:
            return word
    return None
    
def caption_generation(model, image, tokenizer, max_length):
    # Add start tag for generation process
    in_text = "startseq"

    # Iterate over the max length of sequence
    for i in range(max_length):
        # Encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]

        # Pad the sequence
        sequence = tf.keras.preprocessing.sequence.pad_sequences([sequence], max_length)

        # Predict the next word
        yhat = model.predict([image, sequence], verbose=0)

        # Get index with hight probability
        yhat = np.argmax(yhat)

        # Convert token to word
        word = index_to_word(yhat, tokenizer)

        # Stop if word not found
        if word is None:
            break

        # Append word as input for generating next word
        in_text += " " + word

        # Stop if we reach end tag
        if word == "<end>":
            break
    
    return in_text

(x_train, x_test), (y_train, y_test) = load_coco_dataset()

# Bring together all captions in a list and print its length
all_captions = get_all_captions(y_train)

# Create a tokenizer and fit it on train data
tokenizer, vocab_size, max_length = fit_tokenizer(all_captions)

# Create the model
feature_vector_size = 100352 # 7 x 7 x 2048 : output of ResNet50
model = decoder(feature_vector_size, vocab_size, max_length)

# Train the model
epochs = 20
batch_size = 32
data_keys = x_train.keys()
steps = len(data_keys) // batch_size

for i in range(epochs):
    # Create data generator
    generator = data_generator(x_train, y_train, data_keys, tokenizer, max_length, vocab_size, batch_size)
    # Fit for one epoch
    model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)

# Save model
model.save("saved_models/image_captioning_decoder.h5")

# # Fonction de génération de légende pour une image donnée
# def caption_generation(model, image):
#     features = model.encoder(image)
#     text = "<start>"
#     for i in range(model.max_caption_len):
#         sequence = model.tokenizer.texts_to_sequences([text])[0]
#         sequence = tf.keras.preprocessing.sequence.pad_sequences([sequence], maxlen=model.max_caption_len)
#         y_pred = model.decoder.predict([features, sequence], verbose=0)
#         ind_pred = np.argmax(y_pred[0, i])
#         word = model.tokenizer.index_word.get(ind_pred)
#         if word is None:
#             break
#         text += " " + word
#         if word == "<end>":
#             break
#     return text

# # 1. Charger les données prétraitées
# (x_train, x_test), (y_train, y_test) = load_coco_dataset()

# # 2. Créer/charger le tokenizer
# # tokenizer = load_tokenizer("dataset/ms_coco_2017/tokenizer.json")
# tokenizer = create_tokenizer(y_train)

# # 3. Créer le modèle complet (avec encodeur + décodeur)
# model_wrapper = ImageCaptioning(tokenizer=tokenizer)
# model = model_wrapper.decoder_model  # modèle pour entraînement

# # 4. Créer les générateurs de données
# train_gen = CocoDataGenerator(x_train, y_train, tokenizer, batch_size=32)
# test_gen = CocoDataGenerator(x_test, y_test, tokenizer, batch_size=32)

# # 5. Compiler le modèle
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# # 6. Entraîner le modèle
# model.fit(
#     generator_wrapper(train_gen),
#     steps_per_epoch=len(train_gen),
#     validation_data=generator_wrapper(test_gen),
#     validation_steps=len(test_gen),
#     epochs=5
# )

# # 7. Sauvegarder le modèle
# model.save("saved_models/image_captioning_decoder.h5")