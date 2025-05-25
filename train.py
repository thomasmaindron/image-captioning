from dataset.utils.dataset_utils import load_coco_dataset
from dataset.utils.caption_utils import prepare_caption_data, preprocess_captions
from model import ImageCaptioning
import numpy as np

(x_train, x_test), (y_train, y_test) = load_coco_dataset()
print(f"Train data shape: {len(x_train)}, {len(y_train)}")
print("x_train: ",x_train[0])
print("y_train: ",y_train[0])
model= ImageCaptioning()
model.encoder.summary()
model.decoder.summary()

# entrainement
model.decoder.fit([x_train, y_train], y_train, epochs=10, batch_size=32)