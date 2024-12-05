#On fonctionne en tensorflow

from typing import Any, Callable, Dict
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import ReLU
import tensorflow.keras.layers as nn
from tensorflow.keras.layers import concatenate
#from basic_pitch import nn
#from basic_pitch.constants import SAMPLE_RATE



#On va importer le HCQT car ce n'est pas le but de réapprendre à l'implémenter, mais on peut le completer.


def model_v1(input_shape):
    """
    Builds the lightweight neural network for polyphonic transcription.
    :param input_shape: Tuple indicating the input shape (time, frequency, channels).
    :return: TensorFlow Keras model.
    """
    n_harmonics: int = 8
    n_filters_contour: int = 32
    n_filters_onsets: int = 32
    n_filters_notes: int = 32
    no_contours: bool = False
    CONTOURS_BINS_PER_SEMITONE: int = 12
    N_FREQ_BINS_CONTOURS: int = 13
    # Input layer
    inputs = Input(shape=input_shape)

    #n_harmonics: int = 8,
    """
    #############################################################
    # gestion du harmonic stacking prise dans basic pitch / models.py
    if n_harmonics > 1:
        x = nn.HarmonicStacking(
            CONTOURS_BINS_PER_SEMITONE,
            [0.5] + list(range(1, n_harmonics)),
            N_FREQ_BINS_CONTOURS,
        )(x)
    else:
        x = nn.HarmonicStacking(
            CONTOURS_BINS_PER_SEMITONE,
            [1],
            N_FREQ_BINS_CONTOURS,
        )(x)
      ###############################################################"""

    # First convolutional block (to extract multipitch posteriorgram Yp)
    x_frame = Conv2D(32, (5, 5), padding='same', activation='relu')(inputs)
    x_frame = BatchNormalization()(x_frame)
    x_frame = ReLU()(x_frame)
    x_frame = Conv2D(8, (3, 3*13), padding='same', activation='relu')(x_frame)
    x_frame = BatchNormalization()(x_frame)
    x_frame = ReLU()(x_frame)

    # Output multipitch posteriorgram Yp
    Yp = Conv2D(1, (5, 5), padding='same', activation='sigmoid', name='multipitch')(x_frame)

    # Second block (to extract note posteriorgram Yn using Yp)
    x_note = Conv2D(32, (7, 7), padding='same', activation='relu', strides=(1,3))(Yp)
    x_note = ReLU()(x_note)
    Yn = Conv2D(1, (7, 3), padding='same', activation='sigmoid', name='note')(x_note)

    # Third block (to extract onset posteriorgram Yo using audio features and Yn)
    x_audio = Conv2D(32, (5, 5), padding='same', activation='relu', strides = (1,3))(inputs)
    x_audio = BatchNormalization()(x_audio)
    x_audio = ReLU()(x_audio)
    x_concat = concatenate([x_audio, Yn], axis=3, name='concat')
    Yo = Conv2D(1, (3, 3), padding='same', activation='sigmoid', name='onset')(x_concat)

    # Define the model
    model = Model(inputs, [Yo, Yn, Yp], name="lightweight_AMT")
    return model

# Model summary
input_shape = (200, 60, 1)  # Example input shape (time frames, frequency bins, channels)
model = model_v1(input_shape)


# Compiler le modèle
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

# Entraîner le modèle
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,  # Ajustez en fonction de vos besoins
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True)
    ]
)

