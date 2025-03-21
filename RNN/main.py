from train import train_model
from generate import generate_sentence
import tensorflow as tf

if __name__ == "__main__":
    tokenizer, max_length, model = train_model()
    print(generate_sentence("Bonjour", model, tokenizer, max_length))
