import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Configuration des paramètres
SAMPLE_RATE = 44100
F_MIN = 32.7
N_HARMONICS = 7
HARMONICS = [0.5, 1, 2, 3, 4, 5, 6]
HOP_LENGTH = 512
BINS_PER_SEMITONE = 1
BINS_PER_OCTAVE = 12 * BINS_PER_SEMITONE
N_BINS = BINS_PER_OCTAVE * N_HARMONICS
PATH = 'C:/Users/admin/Desktop/master2/MLA/projet/C_major.wav'

# Chargement de l'audio
y, sr = librosa.load(PATH, sr=None)
print(f"Audio chargé avec forme {y.shape} et taux d'échantillonnage {sr} Hz")

# Calcul des décalages pour chaque harmonique
shifts = np.round(12.0 * BINS_PER_SEMITONE * np.log2(HARMONICS)).astype(int)
print(f"Décalages harmoniques calculés : {shifts}")

# Exemple de manipulation de tableaux
def pad_array(arr, pad_width, mode='constant'):
    """
    Ajoute des zéros sur les côtés du tableau en fonction de pad_width.
    """
    return np.pad(arr, pad_width=pad_width, mode=mode)

def concatenate_arrays(arr1, arr2, axis):
    """
    Concatène deux tableaux le long d'un axe donné.
    """
    return np.concatenate((arr1, arr2), axis=axis)

# Exemple d'utilisation avec un tableau fictif
test_array = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])
print("\nTableau original :")
print(test_array)

# Ajout de padding à droite et à gauche
padded_right = pad_array(test_array, pad_width=((1, 0), (0, 2)))
padded_left = pad_array(test_array, pad_width=((0, 0), (1, 0)))

print("\nTableau avec padding à droite :")
print(padded_right)
print("\nTableau avec padding à gauche :")
print(padded_left)

# Exemple de concaténation
array1_2d = np.array([[1, 2], [3, 4]])
array2_2d = np.array([[5, 6], [7, 8]])
print("\nConcaténation le long de l'axe 0 (lignes) :")
concat_axis0 = concatenate_arrays(array1_2d, array2_2d, axis=0)
print(concat_axis0)

print("\nConcaténation le long de l'axe 1 (colonnes) :")
concat_axis1 = concatenate_arrays(array1_2d, array2_2d, axis=1)
print(concat_axis1)
