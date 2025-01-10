import os
import numpy as np
import librosa
import pretty_midi
from tqdm import tqdm
from dsp_utils import cqt as calc_cqt
from dsp_utils import dsp , harmonic_stack
from variables import *



"""

This file contains functions to create data and their labels for .wav and .mid/.midi files

Warning : Use the audio and midi files after passing them by Data_partition.py aka after cutting them into 2 sec segements !

"""

def process_audio(
    audio_file,
    f_min,
    n_bins,
    bins_per_octave,
    harmonics,
    bins_per_semitone,
    sr=22050,
    hop_size=512,
    segment_length=2.0,
):
    """
    Process an audio file into segments and extract a spectrogram (CQT).

    Parameters:
        audio_file (str): Path to the audio file.
        sr (int): Sampling rate.
        hop_size (int): Hop size for the spectrogram.
        segment_length (float): Length of each audio segment in seconds.

    Returns:
        segments (list): List of spectrogram segments (time frames per segemnt  x frequency bins x nbr of harmonics).
    """
    # Load audio 
    y, _ = dsp(path=audio_file,sr=sr)

    # Generate spectrogram (e.g., CQT)
    cqt = calc_cqt(
        signal=y,
        sample_rate=sr,
        hop_size=hop_size,
        f_min=f_min,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        plot=False,
    )
    segments = harmonic_stack(
        cqt, sr, harmonics, hop_size, bins_per_semitone, output_freq, plot=False
    )
    ''' I dont think this is necessary or right 
    # Split into segments
    num_frames = cqt.shape[0]
    segment_frames = int(segment_length * sr / hop_size)
    segments = np.array([
        cqt[i : i + segment_frames]
        for i in range(0, num_frames, segment_frames)
        if i + segment_frames <= num_frames
    ])
    '''
    return segments


def midi_to_frequency(midi_note):
    return 440.0 * 2 ** ((midi_note - 69) / 12)


def generate_frequency_bins(n_bins,sr=44100, bins_per_octave=12, min_freq=20):
    # Generate frequency bins using librosa
    # Generate a constant-Q spectrogram with bins per octave
    bins = librosa.cqt(np.zeros(1), sr=sr, bins_per_octave=bins_per_octave, fmin=min_freq, n_bins=n_bins)
    freq_bins = librosa.cqt_frequencies(len(bins), fmin=min_freq, bins_per_octave=bins_per_octave)
    return freq_bins

def midi_to_bin(midi_note, freq_bins):
    # Convert MIDI pitch to frequency
    frequency = midi_to_frequency(midi_note)
    
    # Find the nearest bin in the frequency bins
    bin_idx = np.argmin(np.abs(freq_bins - frequency))
    return bin_idx


def process_midi(midi_file, num_frames, freq_bins1,freq_bins2,time_resolution):
    """
    Generate onset, activation, and pitch posteriorgrams from a MIDI file.

    Parameters:
        midi_file (str): Path to the MIDI file.
        num_frames (int): Number of time frames in the corresponding audio segment.
        num_pitches (int): Number of MIDI pitches (default: 128).

    Returns:
        Yo : onset_posteriorgram (np.ndarray): Binary matrix of onsets (n_time_frames,frequ_bins:resoltution =1bin/semitone,1)
        Yn : note_activation_posteriorgram (np.ndarray): Binary matrix of activations  (n_time_frames,frequ_bins: resolution =1bin/semitone,1)
        Yp : pitch_activation_posteriorgram (np.ndarray): Binary matrix of activations  (n_time_frames,frequ_bins: resolution =3bin/semitone,1)
    """
   
    # Initialize posteriorgrams
    Yo = np.zeros((num_frames, len(freq_bins1),1))
    Yn = np.zeros((num_frames, len(freq_bins1),1))
    Yp = np.zeros((num_frames, len(freq_bins2),1))

    # Load MIDI file
    midi_data = pretty_midi.PrettyMIDI(str(midi_file))

    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                pitch = note.pitch
                """To verify the line below , use local and not absolute frame"""
                onset_frame = int(np.round(note.start / time_resolution))
                offset_frame = int(np.round(note.end / time_resolution))
                bin_indx1=midi_to_bin(pitch, freq_bins1)
                bin_indx2=midi_to_bin(pitch, freq_bins2)

                # Populate onset posteriorgram
                if 0 <= onset_frame < num_frames and 0 <= bin_indx1 < len(freq_bins1):
                    Yo[onset_frame,bin_indx1,0] = 1

                    # Populate activation posteriorgram
                    for frame in range(onset_frame, min(offset_frame, num_frames)):
                        Yn[frame,bin_indx1,0] = 1
                        Yp[frame,bin_indx2,0] = 1

    return Yo, Yn,Yp


def create_dataset( 
    audio_dir,
    midi_dir,
    f_min,
    n_bins,
    bins_per_octave,
    harmonics,
    sample_rate,
    hop_size,
    segment_length,
    num_frames,
    time_resolution,
    freq_bins1,
    freq_bins2
):
    """
    Create training dataset from audio and MIDI files.

    Parameters:
        audio_dir (str): Directory containing audio files.
        midi_dir (str): Directory containing MIDI files.
        sr (int): Sampling rate.
        hop_size (int): Hop size for the spectrogram.
        segment_length (float): Length of each audio segment in seconds.

    Returns:
        x_train (np.ndarray): Audio segments (num_samples x time frames x frequency bins x 1).
        y_train (dict): Dictionary of onset, activation, and pitch labels.
    """
    x_train = []
    y_train=[]

    # Get list of audio and MIDI files
    audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith(".wav")])
    midi_files = sorted([f for f in os.listdir(midi_dir) if f.endswith(".mid")])

    for audio_file, midi_file in tqdm(
        zip(audio_files, midi_files), total=len(audio_files), desc="Processing files"
    ):
        
        audio_path = os.path.join(audio_dir, audio_file)
        midi_path = os.path.join(midi_dir, midi_file)

        # Process audio
        audio_segments = process_audio(
        audio_path,
        f_min,
        n_bins,
        bins_per_octave,
        harmonics,
        bins_per_semitone,
        sr=sample_rate,
        hop_size=hop_size,
        segment_length=segment_length,
        )


        # Process MIDI
        onset_posteriorgram, activation_posteriorgram, pitch_posteriorgram = (
            process_midi(midi_path, num_frames, freq_bins1,freq_bins2,time_resolution)
        )
        x_train.append(audio_segments)
        y_train.append([onset_posteriorgram,activation_posteriorgram,pitch_posteriorgram])
        


    x_train = np.array(x_train)
    #y_train = list(zip(*y_train))
        


    """
        # Append to dataset
        for segment in audio_segments:
            x_train.append(segment[..., np.newaxis])  # Add channel dimension
            y_onsets.append(onset_posteriorgram)
            y_activations.append(activation_posteriorgram)
            y_pitches.append(pitch_posteriorgram)

    # Convert to arrays
    x_train = np.array(x_train)
    y_train = {
        "onset": np.array(y_onsets, dtype=np.float32),
        "note": np.array(y_activations, dtype=np.float32),
        "multipitch": np.array(y_pitches, dtype=np.float32),
    }
    """

    return x_train, y_train

if __name__=="__main__":


    #audio_file = path_wav
    result=process_audio(
    path_wav,
    f_min,
    n_bins,
    bins_per_octave,
    harmonics,
    bins_per_semitone,
    sr=22050,
    hop_size=512,
    segment_length=2.0,
)
    
    print(result.shape)
    
    freq_bins1=generate_frequency_bins(int(n_bins/3),sample_rate,int(bins_per_octave/3),f_min)
    freq_bins2=generate_frequency_bins(n_bins,sample_rate,bins_per_octave,f_min)

    Yo,Yn,Yp=process_midi(path_midi, num_frames, freq_bins1,freq_bins2,time_resolution)
    print(Yo.shape)
    print(Yn.shape)
    print(Yp.shape)
    print(type(Yp))