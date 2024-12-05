import numpy as np
import matplotlib.pyplot as plt

import librosa
import librosa.display



sample_rate=44100
f_min=32.7
n_harmonics=7
harmonics=[0.5,1,2,3,4,5,6]
hop_length=512
bins_per_semitone=1
bins_per_octave=12*bins_per_semitone
n_bins=bins_per_octave*n_harmonics
output_freq=500
path='C:/Users/admin/Desktop/master2/MLA/projet/C_major.wav'

y, sr = librosa.load(path, sr=None)
print(y.shape,sr)

shifts = np.round(12.0 * bins_per_semitone * np.log2(harmonics)).astype(int)# Calculate shifts for each harmonic
print(shifts)


# Example 2D array (representing a small spectrogram or CQT result)
test1 = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])
print("Original Array:")
print(test1.shape)
print(test1)
padded_right = np.pad(test1, pad_width=((1, 0), (0, 2)), mode='constant')
print("\nPadded on the Right:")
print(padded_right)
padded_left = np.pad(test1, pad_width=((0, 0), (1, 0)), mode='constant')
print("\nPadded on the Left:")
print(padded_left)

# Example 2D arrays
array1_2d = np.array([[1, 2], [3, 4]])
array2_2d = np.array([[5, 6], [7, 8]])
print('shape array 1 =',array1_2d.shape)
concatenated_2d_axis0 = np.concatenate((array1_2d, array2_2d), axis=0)
print("\nConcatenated 2D Arrays along axis 0:\n", concatenated_2d_axis0)
concatenated_2d_axis1 = np.concatenate((array1_2d, array2_2d), axis=-1)
print("\nConcatenated 2D Arrays along axis 1:\n", concatenated_2d_axis1)
print('shape concat 1 =',concatenated_2d_axis1.shape)
#concat selon un axe , ex selon les colonnes je met ligne 1 arr 1 en serie  ligne 1 arr 2 , met les 2 mat en serie