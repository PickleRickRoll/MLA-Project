import os
import librosa
import numpy as np
import soundfile as sf
import pretty_midi

# The aim of this first section is to take every audio file (.wav) inside a folder then resample it and divide it into 2 second audios.

# Parameters
audio_directory = "path_to_audios"  # Replace with your directory
output_directory = "output_dir"      # Directory to save the dataset
sampling_rate = 22050                   # Desired sampling rate
clip_duration = 2                       # Duration of each audio clip in seconds

def extract_audio_segments(audio_directory, output_directory, sampling_rate=22050, clip_duration=2):
    """
    inputs :
    audio_directory : path to audio files.
    output_directory : path to output folder.
    sampling rate and clip duration : self explanatory.
    """
    samples_per_clip = sampling_rate * clip_duration  # Total samples in one clip
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Initialize dataset storage
    dataset = []

    # Process each audio file in the directory
    for audio_file in os.listdir(audio_directory):
        if audio_file.endswith(".wav"):  # Check for .wav files
            file_path = os.path.join(audio_directory, audio_file)
            print(f"Processing {file_path}")
        
            # Load audio
            audio, sr = librosa.load(file_path, sr=sampling_rate)
        
            # Split into 2-second clips
            for start_idx in range(0, len(audio), samples_per_clip):
                clip = audio[start_idx:start_idx + samples_per_clip]
            
                # Ensure the clip is exactly 2 seconds long
                if len(clip) == samples_per_clip:
                    dataset.append(clip)
                
                    # Optional: Save individual clips as .wav files
                    clip_filename = f"{audio_file}_clip_{start_idx // samples_per_clip}.wav"
                    #librosa.output.write_wav(, , sampling_rate)
                    sf.write(os.path.join(output_directory, clip_filename), clip, sampling_rate)


    # The aim of this second section is to take every midi file (.mid) inside a folder then divide it into 2 second midi midi files.


def extract_midi_segment(midi_file, start_time, segment_length):
    """
    Extract a segment from a MIDI file and adjust note times.

    Parameters:
        midi_file (str): Path to the MIDI file.
        start_time (float): Start time of the segment in seconds.
        segment_length (float): Length of the segment in seconds.

    Returns:
        new_midi (pretty_midi.PrettyMIDI): MIDI object containing the extracted segment.
    """
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    new_midi = pretty_midi.PrettyMIDI()

    for instrument in midi_data.instruments:
        new_instrument = pretty_midi.Instrument(program=instrument.program, is_drum=instrument.is_drum)
        for note in instrument.notes:
            if start_time <= note.start < start_time + segment_length:
                new_note = pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=max(0, note.start - start_time),
                    end=max(0, min(note.end, start_time + segment_length) - start_time)
                )
                new_instrument.notes.append(new_note)
        new_midi.instruments.append(new_instrument)

    return new_midi

def process_all_midi_files(input_dir, output_dir, segment_length=2.0):
    """
    Iterate through MIDI files in a directory, extract 2-second segments, and save them.

    Parameters:
        input_dir (str): Directory containing input MIDI files.
        output_dir (str): Directory to save segmented MIDI files.
        segment_length (float): Length of each segment in seconds.
    """
    os.makedirs(output_dir, exist_ok=True)
    midi_files = [f for f in os.listdir(input_dir) if f.endswith(".mid")]

    for midi_file in midi_files:
        midi_path = os.path.join(input_dir, midi_file)
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        total_duration = midi_data.get_end_time()

        segment_start = 0.0
        segment_idx = 0

        while segment_start < total_duration:
            segment_midi = extract_midi_segment(midi_path, segment_start, segment_length)
            output_path = os.path.join(output_dir, f"{os.path.splitext(midi_file)[0]}_segment_{segment_idx}.mid")
            segment_midi.write(output_path)
            segment_start += segment_length
            segment_idx += 1



