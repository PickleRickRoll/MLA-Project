# This file contains functions that are used for the preprocessing of data

import numpy as np
import pretty_midi
import matplotlib.pyplot as plt
import librosa


def trim_midi(midi_path, max_duration=7.0):
    """
    Trims a MIDI file to the first `max_duration` seconds.

    :param midi_path: Path to the input MIDI file.
    :param output_path: Path to save the trimmed MIDI file.
    :param max_duration: Maximum duration in seconds.
    """
    # Load the MIDI file
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    
    # Process each instrument
    for instrument in midi_data.instruments:
        # Filter out notes that start after max_duration
        instrument.notes = [note for note in instrument.notes if note.start < max_duration]
        
        # Adjust end times for notes that partially overlap max_duration
        for note in instrument.notes:
            if note.end > max_duration:
                note.end = max_duration

    return midi_data 


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


# Function to generate matrix posteriograms
def generate_posteriorgrams(midi_data, audio_length, hop_size, sample_rate,freq_bins1,freq_bins2):
    """
    Generate onset and activation posteriorgrams from a MIDI file.

    Parameters:
        midi_file (str): Path to the MIDI file.
        audio_length (float): Length of the audio segment in seconds.
        hop_size (int): Hop size in samples (e.g., 512).
        sample_rate (int): Audio sample rate (e.g., 22050).

    Returns:
        Yo : onset_posteriorgram (np.ndarray): Binary matrix of onsets (n_time_frames,frequ_bins:resoltution =1bin/semitone,1)
        Yn : note_activation_posteriorgram (np.ndarray): Binary matrix of activations  (n_time_frames,frequ_bins: resolution =1bin/semitone,1)
        Yp : pitch_activation_posteriorgram (np.ndarray): Binary matrix of activations  (n_time_frames,frequ_bins: resolution =3bin/semitone,1)
    """
    # Temporal resolution (seconds per frame)
    time_resolution = hop_size / sample_rate
    num_frames = int(np.ceil(audio_length / time_resolution))
    
    

    num_pitches = 128  # MIDI pitches range from 0 to 127

    # Initialize posteriorgrams
    Yo = np.zeros((num_frames, len(freq_bins1),1))
    Yn = np.zeros((num_frames, len(freq_bins1),1))
    Yp = np.zeros((num_frames, len(freq_bins2),1))

    # Load MIDI file
    #midi_data = pretty_midi.PrettyMIDI(midi_file)

    # Process each note in the MIDI file
    for note in midi_data.instruments[0].notes:  # Assuming a single instrument
        pitch = note.pitch
        onset_time = note.start
        offset_time = note.end


        # Convert times to frame indices
        onset_frame = int(np.round(onset_time / time_resolution))
        offset_frame = int(np.round(offset_time / time_resolution))
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


def visualize_posteriorgrams(Yo, Yn,Yp):
    """
    Visualize the onset and activation posteriorgrams.
    
    Parameters:
        onset (np.ndarray): Onset posteriorgram matrix (pitch x time frames).
        activation (np.ndarray): Activation posteriorgram matrix (pitch x time frames).
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    
    # Onset Posteriorgram
    axes[0].imshow(Yo[:,:,0].T, aspect="auto", origin="lower", cmap="hot")
    axes[0].set_title("Onset Posteriorgram : Yo")
    axes[0].set_xlabel("Time Frames")
    axes[0].set_ylabel("MIDI Pitches")
    
    # Activation Posteriorgram
    axes[1].imshow(Yn[:,:,0].T, aspect="auto", origin="lower", cmap="hot")
    axes[1].set_title("Note Activation Posteriorgram : Yn")
    axes[1].set_xlabel("Time Frames")
    axes[1].set_ylabel("MIDI Pitches")
    
    axes[2].imshow(Yp[:,:,0].T, aspect="auto", origin="lower", cmap="hot")
    axes[2].set_title("Pitch Activation Posteriorgram : Yn")
    axes[2].set_xlabel("Time Frames")
    axes[2].set_ylabel("MIDI Pitches")

    plt.tight_layout()
    plt.show()













if __name__=="__main__":

    path='C:/Users/admin/Desktop/master2/MLA/projet/tst files/test_data/bassoon.mid'#Datasets/MTG-QBH/audio projet/C_major_scale.wav
    sample_rate=44100
    f_min=32.7
    n_harmonics=8
    harmonics=[0.5,1,2,3,4,5,6,7]
    hop_length=512  # = sample_rate * frame lenght = 44100*11.6ms
    bins_per_semitone=3
    bins_per_octave=12*bins_per_semitone
    n_bins=bins_per_octave*n_harmonics
    output_freq=500# pas utiliser pour le momment 
    max_duration=7.0
    

    #midi_data = pretty_midi.PrettyMIDI(path)
    midi_data=trim_midi(path,max_duration)
    print("duration:",midi_data.get_end_time())
    print(midi_data.instruments[0].notes)
    """
    print(f'{"note":>10} {"start":>10} {"end":>10}')
    for instrument in midi_data.instruments:
        print("instrument:", instrument.program);
        for note in instrument.notes:
            print(f'{note.pitch:10} {note.start:10} {note.end:10}')
    """
    freq_bins1=generate_frequency_bins(int(n_bins/3),sample_rate,int(bins_per_octave/3),f_min)
    freq_bins2=generate_frequency_bins(n_bins,sample_rate,bins_per_octave,f_min)
    print(len(freq_bins1))
    Yo,Yn,Yp = generate_posteriorgrams(midi_data, max_duration, hop_length, sample_rate,freq_bins1,freq_bins2)
    print(Yo.shape)
    print(Yn.shape)
    print(Yp.shape)
    # Visualize
    visualize_posteriorgrams(Yo, Yn,Yp)
    