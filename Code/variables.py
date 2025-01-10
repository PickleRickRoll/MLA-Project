from pathlib import Path
import numpy as np

sample_rate = 22050
f_min = 32.7
n_harmonics = 8
harmonics = [0.5, 1, 2, 3, 4, 5, 6, 7]
hop_size = 512
bins_per_semitone = 3
bins_per_octave = 12 * bins_per_semitone
n_bins = bins_per_octave * n_harmonics
output_freq = 500  # not used for the moment
segment_length=2.0
time_resolution = hop_size / sample_rate
num_frames = int(np.floor(segment_length / time_resolution))


# mlt_ptch_tst=C3+C4+B3
script_dir = Path(__file__).parent# Get the directory of the current script
project_dir = script_dir.parent
path_wav = project_dir / "test_data"/ "bassoon1.wav"
path_midi=project_dir / "test_data"/ "bassoon.mid"