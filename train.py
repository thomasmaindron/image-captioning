import numpy as np
from model import ImageCaptioning
from dataset.utils.training_utils import CocoDataGenerator
from dataset.utils.dataset_utils import load_coco_dataset
from dataset.utils.caption_utils import load_tokenizer, create_tokenizer
import tensorflow as tf

def generator_wrapper(generator):
    for x_batch, y_batch in generator:
        decoder_input = y_batch[:, :-1]
        decoder_target = y_batch[:, 1:]
        yield [x_batch, decoder_input], decoder_target

# Fonction de génération de légende pour une image donnée
def caption_generation(model, image):
    features = model.encoder(image)
    text = "<start>"
    for i in range(model.max_caption_len):
        sequence = model.tokenizer.texts_to_sequences([text])[0]
        sequence = tf.keras.preprocessing.sequence.pad_sequences([sequence], maxlen=model.max_caption_len)
        y_pred = model.decoder.predict([features, sequence], verbose=0)
        ind_pred = np.argmax(y_pred[0, i])
        word = model.tokenizer.index_word.get(ind_pred)
        if word is None:
            break
        text += " " + word
        if word == "<end>":
            break
    return text

# 1. Charger les données prétraitées
(x_train, x_test), (y_train, y_test) = load_coco_dataset()

# 2. Créer/charger le tokenizer
# tokenizer = load_tokenizer("dataset/ms_coco_2017/tokenizer.json")
tokenizer = create_tokenizer(y_train)

# 3. Créer le modèle complet (avec encodeur + décodeur)
model_wrapper = ImageCaptioning(tokenizer=tokenizer)
model = model_wrapper.decoder_model  # modèle pour entraînement

# 4. Créer les générateurs de données
train_gen = CocoDataGenerator(x_train, y_train, tokenizer, batch_size=32)
test_gen = CocoDataGenerator(x_test, y_test, tokenizer, batch_size=32)

# 5. Compiler le modèle
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# 6. Entraîner le modèle
model.fit(
    generator_wrapper(train_gen),
    steps_per_epoch=len(train_gen),
    validation_data=generator_wrapper(test_gen),
    validation_steps=len(test_gen),
    epochs=5
)

# 7. Sauvegarder le modèle
model.save("saved_models/image_captioning_decoder.h5")