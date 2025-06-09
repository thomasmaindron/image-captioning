import numpy as np
import os
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.spice.spice import Spice
import tensorflow as tf
from tqdm import tqdm
from dataset.utils.dataset_utils import load_split_coco
from prediction import generate_caption_from_feature

def calculate_all_caption_metrics(actual_captions_map, predicted_captions_map):
    """
    Calculates multiple image captioning evaluation metrics (BLEU, ROUGE_L, METEOR, CIDEr, SPICE)
    using the pycocoevalcap library.

    Args:
        actual_captions_map (dict): A dictionary where keys are image IDs (str)
                                    and values are a list of reference captions (list of str).
                                    Example: {'123': ['a man walks', 'an individual moves'], '456': ['a cat sleeps']}
        predicted_captions_map (dict): A dictionary where keys are image IDs (str)
                                       and values are the single predicted caption (str).
                                       Example: {'123': 'a man is walking on the street', '456': 'the cat is napping'}

    Returns:
        dict: A dictionary containing the average scores for BLEU (1-4), ROUGE_L, METEOR, CIDEr, SPICE.
              Scores are floats.
              Returns an empty dictionary if no common images are found for evaluation.
    """
    # Filter image IDs to only evaluate those present in both maps
    image_ids_to_evaluate = list(set(actual_captions_map.keys()) & set(predicted_captions_map.keys()))

    # Prepare data in the format expected by pycocoevalcap
    # gts (ground truths) : {image_id: [{ "caption": "..." }]}
    # res (results) : {image_id: [{ "caption": "..." }]} -- for a single caption per image
    gts = {} # Dictionary for reference captions (ground truths)
    res = {} # Dictionary for generated captions (results)

    for img_id in tqdm(image_ids_to_evaluate, desc="Removing start and end tokens"):
        # Actual captions
        cleaned_actual_captions = []
        for cap in actual_captions_map[img_id]:
            cleaned_cap = cap.replace("startcaption", "").replace("endcaption", "").strip()
            cleaned_actual_captions.append(cleaned_cap)
        gts[img_id] = cleaned_actual_captions

        # Predicted caption
        cleaned_predicted_caption = predicted_captions_map[img_id].replace("startcaption", "").replace("endcaption", "").strip()
        res[img_id] = [cleaned_predicted_caption]

    # Initialize metric evaluators
    # pycocoevalcap uses a common interface for all scorers.
    # The tuples are (scorer_instance, metric_name_or_list_of_names_for_BLEU)
    # Note: METEOR requires Java
    evaluators = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
        # (Meteor(), "METEOR"),
        # (Spice(), "SPICE")
    ]

    all_scores = {}

    for scorer, method_names in tqdm(evaluators, desc="Calculating metrics"):
        print(f"Calculating metric: {method_names}...")
        # compute_score returns (average_score, scores_per_image)
        # We only care about the average score here
        score, _ = scorer.compute_score(gts, res) 
        
        if isinstance(score, list): # For BLEU, 'score' is a list of 4 values
            for i, name in enumerate(method_names):
                all_scores[name] = score[i]
        else: # For other metrics, 'score' is a single value
            all_scores[method_names] = score

    return all_scores


if __name__ == "__main__":
    # Load the tokenizer
    with open('tokenizer.json', 'r', encoding='utf-8') as f:
        data = f.read()
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)
    max_length = 49 # HARDCODED : THIS MIGHT CHANGE IF YOU CREATED A NEW TOKENIZER !!!

    # Load the trained decoder
    decoder = tf.keras.models.load_model("saved_models/1_epochs/1_epochs_decoder.h5") 

    # Load the validation split
    x_val_dir, actual_captions_map = load_split_coco(split="val")

    # Caption generation for the whole split
    predicted_captions_map = {}

    for image_id in tqdm(actual_captions_map.keys(), desc="Generating captions"): 
        feature_path = os.path.join(x_val_dir, f"{image_id}.npy")
        if not os.path.exists(feature_path):
            continue # Skip images without features
        feature_array = np.load(feature_path)
        
        predicted_caption = generate_caption_from_feature(feature_array, decoder, tokenizer, max_length)
        predicted_captions_map[image_id] = predicted_caption
        
    # Calculation of all metrics
    evaluation_scores = calculate_all_caption_metrics(actual_captions_map, predicted_captions_map)

    print("\n--- Captions evaluation results ---")
    for metric, score in evaluation_scores.items():
        print(f"{metric}: {score}")