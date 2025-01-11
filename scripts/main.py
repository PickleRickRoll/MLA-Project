import numpy as np
import tensorflow as tf
from Data_partition import extract_audio_segments,process_all_midi_files
from dsp_utils import cqt, harmonic_stack, dsp, vis_cqt
from modelv1 import model_v1
from preprocessing_utils import create_dataset , generate_frequency_bins
from variables import *
from tqdm.keras import TqdmCallback
from evaluation import mir_eval




if __name__ == "__main__":
    
    freq_bins1=generate_frequency_bins(int(n_bins/3),sample_rate,int(bins_per_octave/3),f_min)
    freq_bins2=generate_frequency_bins(n_bins,sample_rate,bins_per_octave,f_min)
    extract_audio_segments(raw_dir, train_audio_dir , sample_rate, segment_length)
    process_all_midi_files(raw_dir, train_midi_dir, segment_length)
    # Load training data
    X_train, y_train = create_dataset(
        train_audio_dir,
        train_midi_dir,
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
    )

    # Unpack the tuple for each sample in y_train
    onset_array, activation_array, pitch_array = zip(*y_train)  # Unzips into three separate lists

    # Convert each list to a numpy array to maintain batch dimension
    onset_array = np.array(onset_array)  # Shape should be be (num_samples, time frames, frequency bins, 1)
    activation_array = np.array(activation_array) 
    pitch_array = np.array(pitch_array)  


    # Structure y_train as a dictionary or tuple for model training
    Y_train = {
    "onset": onset_array,
    "note": activation_array,
    "multipitch": pitch_array,
    }

    
    
    # Define model
    input_shape = X_train.shape[1:]  # Exclude batch dimension
    model = model_v1(input_shape)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            "onset": "binary_crossentropy",
            "note": "binary_crossentropy",
            "multipitch": "binary_crossentropy",
        },
        metrics={
            "onset": "accuracy",
            "note": "accuracy",
            "multipitch": "accuracy",
        },
        loss_weights={"onset": 0.95, "note": 1.0, "multipitch": 1.0},
    )

    # Train the model
    history = model.fit(
        X_train,
        Y_train,
        batch_size=32,
        epochs=20,
        validation_split=0.2,
        callbacks=[TqdmCallback(verbose=1)],
    )

    # Load and preprocess test sample
    test_audio_path =path_train
    signal, sr = dsp(test_audio_path)
    cqt_result = cqt(signal, sr, hop_size, f_min, n_bins, bins_per_octave, plot=False)
    result = harmonic_stack(
        cqt_result,
        sr,
        harmonics,
        hop_size,
        bins_per_semitone,
        output_freq=None,
        plot=False,
    )
    input_sample = np.expand_dims(result, axis=0)

    # Evaluate the model on the test sample
    output = model.predict(input_sample)
    Yo, Yn, Yp = output[0][0], output[1][0], output[2][0]
    f_measure, fmeasure_no, frame_accuracy = mir_eval(Yo, Yn, Yp,sample_rate)

    # Display metrics
    print(f"F-measure: {f_measure:.3f}")
    print(f"F-measure (no offset): {fmeasure_no:.3f}")
    print(f"Frame-level accuracy: {frame_accuracy:.3f}")
    