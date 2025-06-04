from dataset.utils.dataset_utils import preprocess_dataset, load_coco_dataset
from dataset.utils.caption_utils import get_all_captions, fit_tokenizer
import os
import tensorflow as tf

# Attention ! Bien que x_train et y_train soient tous les deux des dictionnaires ayant pour clés les id des images,
# les clés de x_train sont des str tandis que les clés de y_train sont des int. De plus, y_train recouvre l'ensemble
# des captions tandis que x_train ne recouvre que 20% des images.
# On a donc len(y_train.keys()) = 5 * len(x_train.keys())

# (x_train, x_test), (y_train, y_test) = load_coco_dataset()
# print(x_train.keys())
# print(x_test.keys())
# print(x_train["411685"].shape[0])

# # Put all captions in a list and print its length
# all_captions = get_all_captions(y_train)
# print(len(all_captions))

# # Create a tokenizer and fit it on train data
# tokenizer, vocab_size, max_length = fit_tokenizer(all_captions)

# print(y_train[110])
# print(tokenizer.texts_to_sequences(y_train["411685"]))

print("Version TensorFlow :", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU(s) disponible(s) :", gpus)
    # Si vous voulez forcer TensorFlow à ne pas allouer toute la VRAM d'un coup
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("Aucun GPU détecté par TensorFlow. Utilisation du CPU.")