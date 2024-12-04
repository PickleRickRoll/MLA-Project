from typing import Any, Callable, Dict
import numpy as np
import matplotlib.pyplot as plt

import librosa
import librosa.display
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Input
from tensorflow.keras.models import Model


def dsp(path,sr=44100):
    
    #Resampling
    target_sr = sr  # Target sampling rate 
    low_cutoff = 20     # Low-frequency cutoff (Hz)
    high_cutoff = 20000 # High-frequency cutoff (Hz)
    y, sr = librosa.load(path, sr=None)  # Keep original sampling rate
    if y.ndim > 1:
        #print(f"Audio has {y.shape[0]} channels, converting to mono.")
        y = librosa.to_mono(y)  # Flatten to mono
    if sr != target_sr:
        #print('old sr = ',sr)
        #print('target sr =',target_sr)
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    #Filtering
    

    return y , sr


def cqt(signal,sample_rate,hop_length,n_bins,bins_per_octave,plot=False):
    
    #audio_file = path 

    #y, sr = librosa.load(audio_file, sr=None)  # Preserve original sampling rate
    # y : audio time series. Multi-channel is supported.
    # sr : sampling rate of the signal


    # Compute the CQT
    cqt_result = librosa.cqt(signal, sr=sample_rate, hop_length=hop_length, n_bins=n_bins, bins_per_octave=bins_per_octave)

    #   hop_length (int, optional): : The number of samples between successive frames when computing the transform.
    #This determines the time resolution of the resulting CQT.
    #Lower values give finer time resolution but require more computation.
    #Default: 512 (typical for audio analysis).
    #   n_bins (int): The total number of frequency bins to compute in the CQT.
    #This determines the frequency range covered by the transform.
    #Example: With bins_per_octave=12, n_bins=84 spans 7 octaves (84 / 12 = 7).
    #   bins_per_octave (int):
    #The number of frequency bins per octave.
    #Commonly set to 12 for equal-tempered chromatic scales, aligning with musical notes.
    #If you want higher resolution within each octave, you can increase this value (e.g., 24 for quarter tones).


    if plot :
        # Convert to dB for better visualization
        cqt_db = librosa.amplitude_to_db(np.abs(cqt_result), ref=np.max)

        # Visualize the CQT
        plt.figure(figsize=(10, 6))
        librosa.display.specshow(cqt_db, sr=sr, hop_length=512, x_axis='time', y_axis='cqt_note', bins_per_octave=12)
        plt.colorbar(format="%+2.0f dB")
        plt.title("Constant-Q Transform (CQT)")
        plt.tight_layout()
        plt.show()

    return cqt_result


def harmonic_stack(cqt_result, sr, n_harmonics=7, hop_length=512, bins_per_semitone=12,output_freq=5*12*12,plot=False):
    """
    Parameters:
    - cqt_result: The result of the CQT (shape: [n_times, n_freqs]).
    - n_harmonics: Number of harmonics to stack.
    - bins_per_semitone: The number of bins per semitone in the CQT.
    - output_freq: Number of output frequency bins.
    - plot : true if you want to plot the cqt of the harmonically stacked 
    Returns:
    - stacked_harmonics: The stacked harmonics.
    """
    # Calculate harmonic shifts based on the harmonics
    harmonics = np.arange(1, n_harmonics + 1)
    shifts = np.round(12.0 * bins_per_semitone * np.log2(harmonics)).astype(int)# Calculate shifts for each harmonic
    
    stacked_harmonics = []

    # Iterate through each time frame
    for shift in shifts:
        if shift == 0:
            padded = cqt_result
        elif shift > 0:
            # Pad the frequency axis to the right
            padded = np.pad(cqt_result[:, shift:], ((0, 0), (0, shift)), mode='constant')
        elif shift < 0:
            # Pad the frequency axis to the left
            padded = np.pad(cqt_result[:, :shift], ((0, 0), (-shift, 0)), mode='constant')
        else:
            raise ValueError("Invalid shift value.")
        
        stacked_harmonics.append(padded)


    # Stack the harmonics along the last axis (axis=-1)
    stacked_harmonics = np.concatenate(stacked_harmonics, axis=-1)
    # Cut off unnecessary frequencies (output_freq bins)
    stacked_harmonics = stacked_harmonics[:, :output_freq]

    if plot:
        # Apply a logarithmic scale to better visualize the amplitude variation
        cqt_log = np.log(np.abs(stacked_harmonics) + 1e-6)  # add small value to avoid log(0)

        # Plot the harmonic stacking with a logarithmic scale
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(cqt_log.T, aspect='auto', cmap='inferno', origin='lower', 
                    extent=[0, cqt_result.shape[1] * hop_length / sr, 0, n_harmonics])

        # Add a colorbar linked to the imshow plot
        cbar = plt.colorbar(im, ax=ax, format="%+2.0f dB")
        cbar.set_label('Amplitude (dB)')

        # Set titles and labels
        ax.set_title("Harmonic Stacking")
        ax.set_ylabel("Harmonic")
        ax.set_xlabel("Time (seconds)")

        plt.tight_layout()
        plt.show()

    return stacked_harmonics




sample_rate=44100
hop_length=512
n_bins=84
bins_per_octave=12*12
bins_per_semitone=12
n_harmonics=7


signal,sr=dsp(path='C:/Users/admin/Desktop/master2/MLA/Datasets/MTG-QBH/audio/q4.wav')
cqt_result=cqt(signal,sample_rate=sr,hop_length=hop_length,n_bins=n_bins,bins_per_octave=bins_per_octave,plot=True)
print(cqt_result.shape)  # Should give (n_times, n_freqs)

result=harmonic_stack(cqt_result, sr, n_harmonics=7, hop_length=512, bins_per_semitone=12,output_freq=10*5*12*12,plot=True)

