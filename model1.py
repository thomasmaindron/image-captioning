import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, Add, Concatenate
import numpy as np
import os
import json
from tqdm import tqdm
import pickle

# Paramètres
BATCH_SIZE = 64
BUFFER_SIZE = 1000
EMBEDDING_DIM = 256
UNITS = 512  # Taille du LSTM
VOCAB_SIZE = 10000
MAX_LENGTH = 40
FEATURES_SHAPE = 2048  # Dépend du CNN utilisé
ATTENTION_FEATURES_SHAPE = 64  # Taille de la feature map d'attention

# Préparation des données (exemple simplifié)
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

# Modèle d'encodeur CNN (InceptionV3 préentraîné)
def CNN_Encoder():
    inception = InceptionV3(include_top=False, weights='imagenet')
    output = inception.output
    output = tf.keras.layers.Reshape((-1, output.shape[-1]))(output)
    
    # On garde seulement le modèle jusqu'à la dernière couche convolutionnelle
    image_features_extract_model = Model(inputs=inception.input, outputs=output)
    return image_features_extract_model

# Mécanisme d'attention (optionnel mais recommandé)
class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
        
    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)
        # hidden shape == (batch_size, hidden_size)
        
        # On répète le vecteur hidden pour chaque feature spatiale
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        
        # score shape == (batch_size, 64, 1)
        score = self.V(tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis)))
        
        # attention_weights shape == (batch_size, 64, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # context_vector shape after sum == (batch_size, embedding_dim)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, attention_weights

# Modèle de décodeur LSTM avec attention
class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        
        # Dimensions clés du modèle
        self.units = units  # Taille des états cachés LSTM
        
        # Couche d'embedding pour convertir les indices de mots en vecteurs denses
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        
        # Architecture LSTM principale
        # Note importante: nous utilisons une cellule LSTM standard avec:
        #  - forget_bias=1.0 (par défaut) pour éviter l'oubli trop rapide
        #  - return_sequences=True car nous générons une séquence de mots
        #  - return_state=True pour accéder aux états cachés pour l'attention
        self.lstm = tf.keras.layers.LSTM(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform',
            recurrent_activation='sigmoid',  # porte d'entrée/oubli/sortie avec sigmoid
            activation='tanh'  # activation principale avec tanh
        )
        
        # Couches de traitement supplémentaires
        self.fc1 = tf.keras.layers.Dense(units, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.5)  # Réduire l'overfitting
        
        # Couche de sortie pour prédire le prochain mot
        self.fc2 = tf.keras.layers.Dense(vocab_size)
        
        # Mécanisme d'attention
        self.attention = BahdanauAttention(self.units)
        
    def call(self, x, features, hidden, cell):
        # Contexte d'attention à partir des features de l'image et état caché
        context_vector, attention_weights = self.attention(features, hidden)
        
        # Embedding de l'entrée x (token précédent)
        x = self.embedding(x)
        
        # Concaténation du contexte et de l'embedding
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        
        # Passage par LSTM
        output, state_h, state_c = self.lstm(x, initial_state=[hidden, cell])
        
        # Utilisation du dernier output du LSTM
        x = self.fc1(output)
        x = self.dropout(x)
        
        # Projection finale sur l'espace du vocabulaire
        x = self.fc2(x)
        
        return x, state_h, state_c, attention_weights
    
    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units)), tf.zeros((batch_size, self.units))

# Modèle complet d'image captioning
class ImageCaptioningModel(tf.keras.Model):
    def __init__(self, encoder, decoder, tokenizer, max_length):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def call(self, inputs):
        img_tensor, target = inputs
        
        # Extraction des features par l'encodeur
        features = self.encoder(img_tensor)
        
        # Initialisation des états LSTM
        hidden, cell = self.decoder.reset_state(batch_size=target.shape[0])
        
        # Séparation de l'entrée et de la sortie pour l'entraînement teacher forcing
        dec_input = tf.expand_dims(target[:, 0], 1)  # Le token de départ <start>
        loss = 0
        
        # Décodage séquentiel
        for i in range(1, target.shape[1]):
            # Prédiction du mot suivant
            predictions, hidden, cell, _ = self.decoder(dec_input, features, hidden, cell)
            
            # Calcul de la perte
            loss += self.loss_function(target[:, i], predictions)
            
            # Teacher forcing: utilisation du mot réel comme entrée pour le pas suivant
            dec_input = tf.expand_dims(target[:, i], 1)
            
        return loss / int(target.shape[1])
    
    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = tf.keras.losses.sparse_categorical_crossentropy(real, pred, from_logits=True)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)
    
    def generate_caption(self, image_path):
        # Prétraitement de l'image
        img = load_image(image_path)[0]
        img = tf.expand_dims(img, 0)
        
        # Extraction des features
        features = self.encoder(img)
        
        # Initialisation du décodeur
        hidden, cell = self.decoder.reset_state(batch_size=1)
        dec_input = tf.expand_dims([self.tokenizer.word_index['<start>']], 0)
        
        result = []
        
        # Génération mot par mot
        for i in range(self.max_length):
            predictions, hidden, cell, attention_weights = self.decoder(dec_input, features, hidden, cell)
            
            # Récupération du mot le plus probable
            predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
            
            # Arrêt si fin de séquence
            if self.tokenizer.index_word[predicted_id] == '<end>':
                break
                
            result.append(self.tokenizer.index_word[predicted_id])
            
            # Passage au mot suivant
            dec_input = tf.expand_dims([predicted_id], 0)
            
        return ' '.join(result)

# Fonction d'entraînement
@tf.function
def train_step(img_tensor, target, encoder, decoder, optimizer, tokenizer):
    loss = 0
    
    # Utilisation de GradientTape pour le calcul des gradients
    with tf.GradientTape() as tape:
        features = encoder(img_tensor)
        
        hidden, cell = decoder.reset_state(batch_size=target.shape[0])
        
        # Le premier token d'entrée est le token <start>
        dec_input = tf.expand_dims(target[:, 0], 1)
        
        # Teacher forcing - feeding the target as the next input
        for i in range(1, target.shape[1]):
            # Prédiction du prochain mot
            predictions, hidden, cell, _ = decoder(dec_input, features, hidden, cell)
            
            # Calcul de la perte
            loss += loss_function(target[:, i], predictions)
            
            # Utilisation du mot réel comme entrée pour le pas suivant
            dec_input = tf.expand_dims(target[:, i], 1)
    
    # Calcul de la perte moyenne
    total_loss = (loss / int(target.shape[1]))
    
    # Variables à optimiser
    trainable_variables = encoder.trainable_variables + decoder.trainable_variables
    
    # Calcul des gradients
    gradients = tape.gradient(loss, trainable_variables)
    
    # Application des gradients
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    
    return loss, total_loss

# Configuration de l'entraînement
def train_model(dataset, encoder, decoder, tokenizer, epochs=20):
    optimizer = tf.keras.optimizers.Adam()
    
    # Checkpoint pour sauvegarder le modèle
    checkpoint_path = "./checkpoints/train"
    ckpt = tf.train.Checkpoint(encoder=encoder,
                              decoder=decoder,
                              optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    
    start_epoch = 0
    if ckpt_manager.latest_checkpoint:
        start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
        ckpt.restore(ckpt_manager.latest_checkpoint)
    
    # Boucle d'entraînement
    for epoch in range(start_epoch, epochs):
        total_loss = 0
        
        for (batch, (img_tensor, target)) in enumerate(dataset):
            batch_loss, t_loss = train_step(img_tensor, target, encoder, decoder, optimizer, tokenizer)
            total_loss += t_loss
            
            if batch % 100 == 0:
                print(f'Epoch {epoch+1} Batch {batch} Loss {batch_loss.numpy():.4f}')
        
        # Sauvegarde du modèle tous les 2 epochs
        if (epoch + 1) % 2 == 0:
            ckpt_manager.save()
        
        print(f'Epoch {epoch+1} Loss {total_loss/len(dataset):.6f}')

# Exemple d'utilisation:
if __name__ == "__main__":
    # 1. Préparation des données (à adapter selon votre organisation MS COCO)
    annotation_file = 'path/to/coco/annotations/captions_train2017.json'
    
    # 2. Chargement et prétraitement des légendes
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    # Extraction des légendes et construction du tokenizer
    all_captions = []
    for annot in annotations['annotations']:
        caption = '<start> ' + annot['caption'] + ' <end>'
        all_captions.append(caption)
    
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<unk>", 
                         filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(all_captions)
    
    # Sauvegarde du tokenizer
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # 3. Création des modèles
    encoder = CNN_Encoder()
    decoder = RNN_Decoder(EMBEDDING_DIM, UNITS, VOCAB_SIZE)
    
    # 4. Préparation du dataset
    # (Code pour créer les tenseurs d'images et les séquences de légendes)
    # ...
    
    # 5. Entraînement
    # train_model(dataset, encoder, decoder, tokenizer)
    
    # 6. Inférence
    # model = ImageCaptioningModel(encoder, decoder, tokenizer, MAX_LENGTH)
    # caption = model.generate_caption('path/to/test/image.jpg')
    # print(caption)