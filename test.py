from dataset.utils.caption_utils import load_captions, clean_captions, get_all_captions
from dataset.utils.dataset_utils import preprocess_dataset
import os

ANNOTATION_FILE = r"dataset\ms_coco_2017\annotations\captions_val2017.json"


if not os.path.exists(r"dataset/x_train.npz") or not os.path.exists(r"dataset/x_test.npz"):
    preprocess_dataset()

mapping = load_captions(ANNOTATION_FILE)
print(len(mapping))
mapping = clean_captions(mapping)
print(mapping[223747])
all_captions = get_all_captions(mapping)
print(len(all_captions))
