import numpy as np
import matplotlib.pyplot as plt
from variables import *
import librosa
import librosa.display


"""
This file contains digital signal processing tools 


"""



def dsp(path, sr=22050):

    # Resampling
    target_sr = sr  # Target sampling rate
    y, sr = librosa.load(path, sr=None)  # Keep original sampling rate
    if y.ndim > 1:
        # print(f"Audio has {y.shape[0]} channels, converting to mono.")
        y = librosa.to_mono(y)  # Flatten to mono
    if sr != target_sr:
        # print('old sr = ',sr)
        # print('target sr =',target_sr)
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return y, sr


def cqt(signal, sample_rate, hop_size, f_min, n_bins, bins_per_octave, plot=False):

    """
        y : audio time series. Multi-channel is supported.
        sr : sampling rate of the signal
        hop_length (int, optional): : The number of samples between successive frames when computing the transform.
    This determines the time resolution of the resulting CQT.
    Lower values give finer time resolution but require more computation.
    Default: 512 (typical for audio analysis).
        n_bins (int): The total number of frequency bins to compute in the CQT.
    This determines the frequency range covered by the transform.
    Example: With bins_per_octave=12, n_bins=84 spans 7 octaves (84 / 12 = 7).
        bins_per_octave (int):
    The number of frequency bins per octave.
    Commonly set to 12 for equal-tempered chromatic scales, aligning with musical notes.
    If you want higher resolution within each octave, you can increase this value (e.g., 24 for quarter tones).

    output dim ( n t frames, n_bins , 1) a prevoir nbatch et concatiner avec 1 ---> final output : (n_batch, n_times, n_freqs, 1)
    the 1 is here to later concatenate on this ax for the hcqt
    """

    cqt_result = librosa.cqt(
        signal,
        sr=sample_rate,
        hop_length=hop_size,
        fmin=f_min,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
    )

    if plot:
        # Convert to dB for better visualization
        cqt_db = librosa.amplitude_to_db(np.abs(cqt_result), ref=np.max)

        # Visualize the CQT
        plt.figure(figsize=(10, 6))
        librosa.display.specshow(
            cqt_db,
            sr=sample_rate,
            hop_length=hop_size,
            x_axis="time",
            y_axis="cqt_note",
            bins_per_octave=bins_per_octave,
        )
        plt.colorbar(format="%+2.0f dB")
        plt.title("Constant-Q Transform (CQT)")
        plt.tight_layout()
        plt.show()
    cqt_result = cqt_result.T
    cqt_result = cqt_result.reshape(*cqt_result.shape, 1)
    # return cqt_result
    return np.abs(cqt_result)


def harmonic_stack(
    cqt_result,
    sr,
    harmonics,
    hop_length=512,
    bins_per_semitone=12,
    output_freq=5 * 12 * 12,
    plot=False,
):
    """
    Parameters:
    - cqt_result: The result of the CQT (shape: [n_times, n_freqs]).
    - harmonics: list of harmonics to stack.
    - bins_per_semitone: The number of bins per semitone in the CQT.
    - output_freq: Number of output frequency bins.
    - plot : true if you want to plot the cqt of the harmonically stacked
    Returns:
    - stacked_harmonics: The stacked harmonics.

    Inspired from basic-pitch

    input dimensions(n time frames , n bins , 1)
    time_frames=lenght signal / legnth hop frames = lsignal / (hop length/sample rate)
    n bins = n of harmonics * bins per octave

    output : (n_times, n_output_freqs, len(harmonics)) ----> final :  (n_batch, n_times, n_output_freqs, len(harmonics))

    """
    # Calculate harmonic shifts based on the harmonics

    shifts = np.round(12.0 * bins_per_semitone * np.log2(harmonics)).astype(
        int
    )  # Calculate shifts for each harmonic

    stacked_harmonics = []

    # Iterate through each time frame , inspired from basic-pitch
    for shift in shifts:
        if shift == 0:
            padded = cqt_result
        elif shift > 0:
            # Pad the frequency axis to the right (from top)
            padded = np.pad(
                cqt_result[:, shift:, :], ((0, 0), (0, shift), (0, 0)), mode="constant"
            )
        elif shift < 0:
            # Pad the frequency axis to the left
            padded = np.pad(
                cqt_result[:, :shift, :], ((0, 0), (-shift, 0), (0, 0)), mode="constant"
            )
        else:
            raise ValueError("Invalid shift value.")

        stacked_harmonics.append(padded)

    # Stack the harmonics along the last axis (axis=-1)
    stacked_harmonics = np.concatenate(stacked_harmonics, axis=-1)
    # Cut off unnecessary frequencies (output_freq bins)
    # stacked_harmonics = stacked_harmonics[:,:output_freqs ,:]

    if plot:
        cqt_log1 = librosa.amplitude_to_db(
            np.abs(stacked_harmonics[:, :, 1].T) + 1e-6, ref=np.max
        )
        cqt_log2 = librosa.amplitude_to_db(
            np.abs(stacked_harmonics[:, :, 2].T) + 1e-6, ref=np.max
        )
        cqt_log3 = librosa.amplitude_to_db(
            np.abs(stacked_harmonics[:, :, 3].T) + 1e-6, ref=np.max
        )

        plt.figure(figsize=(10, 6))
        for i, cqt_log in enumerate([cqt_log1, cqt_log2, cqt_log3], 1):
            plt.subplot(1, 3, i)
            librosa.display.specshow(
                cqt_log,
                sr=sr,
                hop_length=hop_length,
                x_axis="time",
                y_axis="cqt_note",
                bins_per_octave=bins_per_semitone * 12,
            )
            plt.title(f"Harmonic {i}")
        plt.colorbar(format="%+2.0f dB")
        plt.tight_layout()
        plt.show()

    return stacked_harmonics


def vis_cqt(result, sample_rate, hop_length, bins_per_semitone, title, cond=False):
    if cond:
        cqt_log = librosa.amplitude_to_db(np.abs(result[:, :, 0].T) + 1e-6, ref=np.max)

        plt.figure(figsize=(10, 6))
        librosa.display.specshow(
            cqt_log,
            sr=sample_rate,
            hop_length=hop_length,
            x_axis="time",
            y_axis="cqt_note",
            bins_per_octave=bins_per_semitone * 12,
        )
        plt.title(title)
        plt.colorbar(format="%+2.0f dB")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    signal, sr = dsp(path_simple_wav)
    cqt_result = cqt(signal, sr, hop_size, f_min, n_bins, bins_per_octave, plot=True)
    print(cqt_result.shape)  # Should give (n_times, n_freqs,1)
    result = harmonic_stack(
        cqt_result, sr, harmonics, hop_size, bins_per_semitone, output_freq, plot=True
    )
    print(result.shape)
