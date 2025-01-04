from utils import cqt , harmonic_stack , dsp 
import numpy as np
from modelv1 import model_v1
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import ReLU
import tensorflow.keras.layers as nn
from tensorflow.keras.layers import concatenate

path='C:/Users/admin/Desktop/master2/MLA/Datasets/vocadito/audio/vocadito_9.wav'#Datasets/MTG-QBH/audio projet/C_major_scale.wav
sample_rate=44100
f_min=32.7
n_harmonics=8
harmonics=[0.5,1,2,3,4,5,6,7]
hop_length=512
bins_per_semitone=3
bins_per_octave=12*bins_per_semitone
n_bins=bins_per_octave*n_harmonics
output_freq=500#pas utiliser pour le momment 



signal,sr=dsp(path)
cqt_result=cqt(signal,sr,hop_length,f_min,n_bins,bins_per_octave,plot=False)
print(cqt_result.shape)  # Should give (n_times, n_freqs)

result=harmonic_stack(cqt_result, sr, harmonics, hop_length, bins_per_semitone,output_freq,plot=False)
print(result.shape)

model=model_v1(result.shape)
model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'onset': 'binary_crossentropy',
            'note': 'binary_crossentropy',
            'multipitch': 'binary_crossentropy',
        },
        metrics={
            'onset': 'accuracy',
            'note': 'accuracy',
            'multipitch': 'accuracy',
        },
        loss_weights={'onset': 0.95, 'note': 1.0, 'multipitch': 1.0}
    )

input=np.expand_dims(result, axis=0)
output=model.predict(input)
print("Model Output Shape:", output)