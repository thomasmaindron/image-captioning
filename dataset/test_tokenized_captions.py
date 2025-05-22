import numpy as np
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
tokenized_captions = np.load("dataset/ms_coco_2017/tokenized_captions.npy")

print(tokenized_captions.shape)  # (nombre_de_captions, longueur_max)
print(tokenized_captions[0])     # affiche la 1ère caption sous forme de séquence


#retrouver le texte à partir des index

# Charger le tokenizer
with open("dataset/ms_coco_2017/tokenizer.json") as f:
    tokenizer = tokenizer_from_json(json.load(f))

# Exemple : décoder la première caption
decoded_caption = [tokenizer.index_word.get(idx, "<unk>") for idx in tokenized_captions[0]]
print(" ".join(decoded_caption))
