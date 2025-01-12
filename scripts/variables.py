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
num_frames = int(np.ceil(segment_length / time_resolution))



"""
These are local path that the scripts use to work properly , you can always change them to your liking
but it's recomended to keep them like this 
"""



# mlt_ptch_tst=C3+C4+B3
script_dir = Path(__file__).parent# Get the directory of the current script
project_dir = script_dir.parent
path_simple_wav = project_dir / "test_data"/ "simple_sounds" / "C_major_scale.wav"
path_wav = project_dir / "test_data"/  "bassoon1.wav"
path_midi=project_dir / "test_data"/ "bassoon.mid"
path_train=project_dir / "test_data"/"X_train"/"bassoon1.wav_clip_80.wav"#be careful of silent clips , they can return problems
path_ytrain=project_dir / "test_data"/"Y_train"/"bassoon_segment_80.mid"
model_path=project_dir / "model"/"model.keras"

raw_dir=project_dir / "test_data"
train_audio_dir=project_dir / "test_data"/"X_train"
train_midi_dir=project_dir / "test_data"/"Y_train"
