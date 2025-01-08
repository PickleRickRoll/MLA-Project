import numpy as np
import tensorflow as tf

from utils import cqt, harmonic_stack, dsp, vis_cqt
import librosa
import mir_eval as mval
from modelv1 import model_v1
from data.preprocessing_utils2 import create_dataset

from tqdm.keras import TqdmCallback


def mir_eval(Yo, Yp, Yn):
    # Paramètres
    sr = 44100

    # Post-traitement pour obtenir les notes estimées
    onset_threshold = 0.5
    note_threshold = 0.5

    estimated_notes = []
    for f in range(Yo.shape[0]):
        onsets = np.where(Yo[f] > onset_threshold)[0]
        for onset in onsets:
            offset = onset + np.argmax(Yn[f, onset:] < note_threshold)
            if offset == onset:
                offset = len(Yn[f])
            # Get the frequency
            freq = librosa.fft_frequencies(sr=sr)[f]
            # Filter out frequencies that are too low
            if freq > 0:  # or some other minimum frequency threshold
                estimated_notes.append(
                    (
                        librosa.frames_to_time(onset),
                        librosa.frames_to_time(offset),
                        freq,
                    )
                )

    notes = estimated_notes

    # Évaluation avec mir_eval
    ref_intervals = np.array([[start, end] for start, end, _ in notes])

    ref_pitches = np.array([librosa.hz_to_midi(freq) for _, _, freq in notes])

    est_intervals = np.array([[start, end] for start, end, _ in estimated_notes])
    est_pitches = np.array([librosa.hz_to_midi(freq) for _, _, freq in estimated_notes])

    onset_tolerance = 0.5  # 50 ms
    pitch_tolerance = 0.25  # quart de ton
    offset_ratio = 0.2  # 20% de la durée de la note

    # F-measure
    precision, recall, f_measure, average_overlap_ratio = (
        mval.transcription.precision_recall_f1_overlap(
            ref_intervals,
            ref_pitches,
            est_intervals,
            est_pitches,
            onset_tolerance=onset_tolerance,
            pitch_tolerance=pitch_tolerance,
            offset_ratio=offset_ratio,
        )
    )

    # F-measure no offset
    precision_no, recall_no, f_measure_no, average_overlap_ratio_no = (
        mval.transcription.precision_recall_f1_overlap(
            ref_intervals,
            ref_pitches,
            est_intervals,
            est_pitches,
            onset_tolerance=onset_tolerance,
            pitch_tolerance=pitch_tolerance,
            offset_ratio=None,  # Ignorer les offsets
        )
    )

    (
        frame_precision,
        frame_recall,
        frame_accuracy,
    ) = mval.transcription.precision_recall_f1_overlap(
        ref_intervals,
        ref_pitches,
        est_intervals,
        est_pitches,
        offset_ratio=None,  # to calculate accuracy based on onsets and pitches only
        onset_tolerance=0.05,  # adjust tolerance as needed
        pitch_tolerance=0.25,  # adjust tolerance as needed
    )[
        0:3
    ]  # to extract only precision, recall, accuracy # changed from mir_eval.transcription_velocity to mir_eval.transcripti

    return f_measure, f_measure_no, frame_accuracy


if __name__ == "__main__":
    # Parameters
    sample_rate = 22050
    hop_length = 512
    segment_length = 2.0  # 2 seconds
    f_min = 32.7
    bins_per_semitone = 3
    bins_per_octave = 12 * bins_per_semitone
    n_harmonics = 8
    n_bins = bins_per_octave * n_harmonics
    harmonics = [0.5, 1, 2, 3, 4, 5, 6, 7]

    # Paths to training data
    train_audio_dir = "../tst_files/train_data/X_train"
    train_midi_dir = "../tst_files/train_data/Y_train"

    # Load training data
    X_train, Y_train = create_dataset(#TODO Fix preprocessing for similar X and Y shaped outputs
        audio_dir=train_audio_dir,
        midi_dir=train_midi_dir,
        f_min=f_min,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        sr=sample_rate,
        hop_size=hop_length,
        segment_length=segment_length,
    )

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
    test_audio_path = "../tst_files/C_major_scale.wav"
    signal, sr = dsp(test_audio_path)
    cqt_result = cqt(signal, sr, hop_length, f_min, n_bins, bins_per_octave, plot=False)
    result = harmonic_stack(
        cqt_result,
        sr,
        harmonics,
        hop_length,
        bins_per_semitone,
        output_freq=None,
        plot=False,
    )
    input_sample = np.expand_dims(result, axis=0)

    # Evaluate the model on the test sample
    output = model.predict(input_sample)
    Yo, Yn, Yp = output[0][0], output[1][0], output[2][0]
    f_measure, fmeasure_no, frame_accuracy = mir_eval(Yo, Yn, Yp)

    # Display metrics
    print(f"F-measure: {f_measure:.3f}")
    print(f"F-measure (no offset): {fmeasure_no:.3f}")
    print(f"Frame-level accuracy: {frame_accuracy:.3f}")
