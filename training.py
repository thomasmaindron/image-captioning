import numpy as np
import tensorflow as tf
from dataset.utils.caption_utils import fit_tokenizer, get_all_captions
from dataset.utils.dataset_utils import load_training_split_coco
from dataset.utils.training_utils import data_generator
from model import decoder

(x_train, y_train) = load_training_split_coco()

# Bring together all captions in a list and print its length
all_captions = get_all_captions(y_train)

# Create a tokenizer and fit it on train data
tokenizer, max_length, vocab_size = fit_tokenizer(all_captions)

# Create the model
feature_vector_size = 100352 # 7 x 7 x 2048 : output of ResNet50
model = decoder(feature_vector_size, max_length, vocab_size)

# Train the model
epochs = 20
batch_size = 16
data_keys = x_train.keys()
steps = len(data_keys) // batch_size

for i in range(epochs):
    # Create data generator
    generator = data_generator(x_train, y_train, data_keys, tokenizer, max_length, vocab_size, batch_size)
    # Fit for one epoch
    model.fit(generator, epochs=1, steps_per_epoch=steps)

# Save model
model.save("saved_models/image_captioning_decoder.h5")