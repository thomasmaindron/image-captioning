import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Charger les annotations directement en JSON
annotations_path = "dataset/ms_coco_2017/annotations/captions_train2017.json"

with open(annotations_path, "r") as f:
    captions_data = json.load(f)

# Extraire les captions (image_id -> captions)
image_captions = {}
all_captions = []

for ann in captions_data["annotations"]:
    image_id = str(ann["image_id"])
    caption = ann["caption"]

    if image_id not in image_captions:
        image_captions[image_id] = []
    image_captions[image_id].append(caption)
    all_captions.append(caption)

# Tokenisation
tokenizer = Tokenizer(num_words=10000, oov_token="<unk>")
tokenizer.fit_on_texts(all_captions)

# Convertir les captions en séquences
sequences = tokenizer.texts_to_sequences(all_captions)

# Padding
max_length = max(len(seq) for seq in sequences)
sequences_padded = pad_sequences(sequences, maxlen=max_length, padding="post")

# Sauvegarder le tokenizer et les séquences
np.save("dataset/ms_coco_2017/tokenized_captions.npy", sequences_padded)

with open("dataset/ms_coco_2017/tokenizer.json", "w") as f:
    json.dump(tokenizer.to_json(), f)

print("Captions tokenized and saved!")