import os
import numpy as np
import tensorflow as tf
from dataset.utils.dataset_utils import load_split_coco
from prediction import generate_caption_from_feature

# Load the tokenizer
with open('tokenizer.json', 'r', encoding='utf-8') as f:
    data = f.read()
tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)
max_length = 49 # HARDCODED : THIS MIGHT CHANGE IF YOU CREATED A NEW TOKENIZER !!!

# Load the trained decoder
decoder = tf.keras.models.load_model("saved_models/image_captioning_decoder.h5") 

# Prediction on data of validation split
x_val_dir, y_val = load_split_coco(split="val")
image_id = "50811" # image id to predict
# Load the feature array
feature_path = os.path.join(x_val_dir, f"{image_id}.npy")
feature_array = np.load(feature_path)

# Generate caption from the feature
predicted_caption = generate_caption_from_feature(feature_array, decoder, tokenizer, max_length)

# Compare actual captions with predicted caption
print(f"Actual captions:\n{y_val.get(image_id)}")
print(f"Predicted caption for ID {image_id}:\n{predicted_caption}")