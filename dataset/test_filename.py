# Exemple de lecture
import numpy as np

x_train = np.load("dataset/x_train.npy")  # shape (nb_images, 128, 128, 3)
print(x_train.shape)
import matplotlib.pyplot as plt

plt.imshow(x_train[0])
plt.show()
x_train_filenames = np.load("dataset/x_train_filenames.npy", allow_pickle=True)
print(x_train_filenames[:5])  # Affiche les 5 premiers noms de fichiers
