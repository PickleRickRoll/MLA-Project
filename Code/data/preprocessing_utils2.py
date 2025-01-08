import os
import numpy as np
import librosa
import pretty_midi
from tqdm import tqdm
from utils import cqt as calc_cqt
from utils import dsp


def process_audio(
    audio_file,
    f_min,
    n_bins,
    bins_per_octave,
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
        segments (list): List of spectrogram segments (time frames x frequency bins).
    """
    # Load audio 
    #TODO fix here the shaping
    y, _ = dsp(path=audio_file,sr=sr)
    segment_samples = int(segment_length * sr)

    # Generate spectrogram (e.g., CQT)
    cqt = calc_cqt(
        signal=y,
        sample_rate=sr,
        hop_length=hop_size,
        f_min=f_min,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        plot=False,
    )
    # Split into segments
    num_frames = cqt.shape[0]
    segment_frames = int(segment_length * sr / hop_size)
    segments = [
        cqt[i : i + segment_frames]
        for i in range(0, num_frames, segment_frames)
        if i + segment_frames <= num_frames
    ]

    return segments


def process_midi(midi_file, num_frames, time_resolution, num_pitches=128):
    """
    Generate onset, activation, and pitch posteriorgrams from a MIDI file.

    Parameters:
        midi_file (str): Path to the MIDI file.
        num_frames (int): Number of time frames in the corresponding audio segment.
        time_resolution (float): Time per frame in seconds.
        num_pitches (int): Number of MIDI pitches (default: 128).

    Returns:
        onset_posteriorgram, activation_posteriorgram, pitch_posteriorgram: Binary matrices.
    """
    # Initialize posteriorgrams
    onset_posteriorgram = np.zeros((num_pitches, num_frames))
    activation_posteriorgram = np.zeros((num_pitches, num_frames))
    pitch_posteriorgram = np.zeros((num_pitches, num_frames))

    # Load MIDI file
    midi_data = pretty_midi.PrettyMIDI(midi_file)

    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                pitch = note.pitch
                onset_frame = int(np.round(note.start / time_resolution))
                offset_frame = int(np.round(note.end / time_resolution))

                # Populate onset
                if 0 <= onset_frame < num_frames:
                    onset_posteriorgram[pitch, onset_frame] = 1

                # Populate activation
                for frame in range(onset_frame, min(offset_frame, num_frames)):
                    activation_posteriorgram[pitch, frame] = 1

                # Populate pitch posteriorgram
                pitch_posteriorgram[pitch, onset_frame:offset_frame] = 1

    return onset_posteriorgram, activation_posteriorgram, pitch_posteriorgram


def create_dataset( #TODO FIX THIS FUNCTION
    audio_dir,
    midi_dir,
    f_min,
    n_bins,
    bins_per_octave,
    sr=22050,
    hop_size=512,
    segment_length=2.0,
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
    y_onsets, y_activations, y_pitches = [], [], []

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
            audio_file=audio_path,
            f_min=f_min,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            sr=sr,
            hop_size=hop_size,
            segment_length=segment_length,
        )
        time_resolution = hop_size / sr
        num_frames = int(segment_length * sr / hop_size)

        # Process MIDI
        onset_posteriorgram, activation_posteriorgram, pitch_posteriorgram = (
            process_midi(midi_path, num_frames, time_resolution)
        )

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

    return x_train, y_train
