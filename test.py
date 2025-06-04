from dataset.utils.dataset_utils import preprocess_dataset, load_coco_dataset
from dataset.utils.caption_utils import get_all_captions, fit_tokenizer
import os

ANNOTATION_FILE = r"dataset\ms_coco_2017\annotations\captions_val2017.json"

if not os.path.exists(r"dataset/x_train.npz") or not os.path.exists(r"dataset/x_test.npz"):
    preprocess_dataset()

# Attention ! Bien que x_train et y_train soient tous les deux des dictionnaires ayant pour clés les id des images,
# les clés de x_train sont des str tandis que les clés de y_train sont des int. De plus, y_train recouvre l'ensemble
# des captions tandis que x_train ne recouvre que 20% des images.
# On a donc len(y_train.keys()) = 5 * len(x_train.keys())

(x_train, x_test), (y_train, y_test) = load_coco_dataset()
print(x_train.keys())
print(x_test.keys())

# Put all captions in a list and print its length
all_captions = get_all_captions(y_train)
print(len(all_captions))

# Create a tokenizer and fit it on train data
tokenizer, vocab_size, max_length = fit_tokenizer(all_captions)

print(y_train[110])
print(tokenizer.texts_to_sequences(y_train[110]))