import tensorflow as tf
from dataset.utils.caption_utils import fit_tokenizer, get_all_captions
from dataset.utils.dataset_utils import load_split_coco
from dataset.utils.training_utils import data_generator
from decoder import decoder
import math
import os
import json


if __name__ == "__main__":
    # --- START OF GPU MEMORY OPTIMIZATION BLOCK ---
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("Mixed precision policy set to 'mixed_float16'.")

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth activated for GPU(s).")
    else:
        print("No GPU detected, training will proceed on CPU.")
    # --- END OF GPU MEMORY OPTIMIZATION BLOCK ---


    # --- DATA LOADING ---
    (train_features_dir, y_train) = load_split_coco(split="train")

    # We go through the features folder and extract the IDs of each images from the filenames
    train_image_keys = []
    for filename in os.listdir(train_features_dir):
        if filename.endswith('.npy'):
            image_id = os.path.splitext(filename)[0] # "image_id.npy" -> "image_id"
            train_image_keys.append(image_id)
    # Filter y_train so that it only keeps the captions of the same images as the features
    # Also prevent useless words to be added to the tokenizer if fitted here
    y_train_filtered = {key: captions for key, captions in y_train.items() if key in train_image_keys}

    # If a tokenizer already exists (a training has already been done), we take it back
    if os.path.exists("tokenizer.json"):
        with open('tokenizer.json', 'r', encoding='utf-8') as f:
            data = f.read()
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)
        max_length = 49 # HARDCODED : THIS MIGHT CHANGE IF YOU CREATED A NEW TOKENIZER !!!
        vocab_size = len(tokenizer.word_index) + 1
    else:
        all_captions = get_all_captions(y_train_filtered)
        tokenizer, max_length, vocab_size = fit_tokenizer(all_captions)


    # --- MODEL CREATION ---
    feature_vector_size = 100352 # 7 x 7 x 2048 : output of ResNet50
    model = decoder(feature_vector_size, max_length, vocab_size)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])


    # --- STEPS_PER_EPOCH CALCULATION ---
    batch_size = 64

    total_train_samples = 0
    for key in train_image_keys: # We go through each image
        captions = y_train_filtered[key]
        for caption in captions: # We go through each caption
            seq = tokenizer.texts_to_sequences([caption])[0]
            if len(seq) > 1:
                total_train_samples += (len(seq) - 1) # Number of iterations to predict the caption (seq2seq)

    steps_per_epoch = math.ceil(total_train_samples / batch_size)


    # --- CREATION OF TF.DATA.DATASET ---
    # The three outputs are : feature vector, tokenized sequence, target word (one-hot vector)
    dataset_train = tf.data.Dataset.from_generator(
        lambda: data_generator(train_features_dir, y_train_filtered, train_image_keys, tokenizer, max_length, vocab_size),
        output_types=((tf.float32, tf.int32), tf.float32),
        output_shapes=((tf.TensorShape([feature_vector_size]), tf.TensorShape([max_length])), tf.TensorShape([vocab_size]))
    )

    dataset_train = dataset_train.batch(batch_size)
    dataset_train = dataset_train.prefetch(tf.data.AUTOTUNE)


    # --- MODEL TRAINING ---
    epochs = 20
    history = model.fit(dataset_train, epochs=epochs, steps_per_epoch=steps_per_epoch)


    # --- SAVE THE MODEL ---
    model.save(f"saved_models/{epochs}_epochs/{epochs}_epochs_decoder.h5")
    with open(f"saved_models/{epochs}_epochs/{epochs}_epochs_history.json"):
        json.dump(history.history, f)

    # à retirer
    print(f"La valeur de max_length est {max_length}")