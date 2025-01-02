# This file contains functions that are used for the preprocessing of data

import numpy as np
import pretty_midi


# Function to generate matrix posteriograms
def generate_posteriorgrams(midi_file, audio_length, hop_size, sample_rate):
    """
    Generate onset and activation posteriorgrams from a MIDI file.

    Parameters:
        midi_file (str): Path to the MIDI file.
        audio_length (float): Length of the audio segment in seconds.
        hop_size (int): Hop size in samples (e.g., 512).
        sample_rate (int): Audio sample rate (e.g., 22050).

    Returns:
        onset_posteriorgram (np.ndarray): Binary matrix of onsets (pitch x time frames).
        activation_posteriorgram (np.ndarray): Binary matrix of activations (pitch x time frames).
    """
    # Temporal resolution (seconds per frame)
    time_resolution = hop_size / sample_rate
    num_frames = int(np.ceil(audio_length / time_resolution))
    num_pitches = 128  # MIDI pitches range from 0 to 127

    # Initialize posteriorgrams
    onset_posteriorgram = np.zeros((num_pitches, num_frames))
    activation_posteriorgram = np.zeros((num_pitches, num_frames))

    # Load MIDI file
    midi_data = pretty_midi.PrettyMIDI(midi_file)

    # Process each note in the MIDI file
    for note in midi_data.instruments[0].notes:  # Assuming a single instrument
        pitch = note.pitch
        onset_time = note.start
        offset_time = note.end

        # Convert times to frame indices
        onset_frame = int(np.round(onset_time / time_resolution))
        offset_frame = int(np.round(offset_time / time_resolution))

        # Populate onset posteriorgram
        if 0 <= onset_frame < num_frames:
            onset_posteriorgram[pitch, onset_frame] = 1

        # Populate activation posteriorgram
        for frame in range(onset_frame, min(offset_frame, num_frames)):
            activation_posteriorgram[pitch, frame] = 1

    return onset_posteriorgram, activation_posteriorgram
