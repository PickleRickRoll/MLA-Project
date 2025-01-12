import numpy as np
import librosa
import mir_eval as mval



def mir_eval(grd_truth,estimated, sr ):

    note_threshold=0.2
    onset_threshold = 0.2
    estimated_notes = []
    notes=[]
    
    Yot,Ynt,Ypt=grd_truth[0],grd_truth[1],grd_truth[2]
    Yoe,Yne,Ype=estimated[0][0],estimated[1][0],estimated[2][0]

    for f in range(Yoe.shape[1]):  # Iterate over frequency bins
        onsets = np.where(Yoe[:, f] > onset_threshold)[0]  # Find onsets for this frequency bin
        for onset in onsets:
            offset = onset + np.argmax(Yne[onset:, f] < note_threshold)  # Find the offset
            if offset == onset:
                offset = Yoe.shape[0]  # Set to the last time frame if no offset is found
            # Get the frequency for this bin
            freq = librosa.fft_frequencies(sr=sr)[f]
            # Filter out frequencies that are too low
            if freq > 0:  # or some other minimum frequency threshold
                estimated_notes.append(
                    (
                        librosa.frames_to_time(onset, sr=sr),  # Convert onset time
                        librosa.frames_to_time(offset, sr=sr),  # Convert offset time
                        freq,  # Frequency of this bin
                    )
                )


    for f in range(Yot.shape[1]):  # Iterate over frequency bins
        onsets = np.where(Yot[:, f] > onset_threshold)[0]  # Find onsets for this frequency bin
        for onset in onsets:
            offset = onset + np.argmax(Ynt[onset:, f] < note_threshold)  # Find the offset
            if offset == onset:
                offset = Yot.shape[0]  # Set to the last time frame if no offset is found
            # Get the frequency for this bin
            freq = librosa.fft_frequencies(sr=sr)[f]
            # Filter out frequencies that are too low
            if freq > 0:  # or some other minimum frequency threshold
                notes.append(
                    (
                        librosa.frames_to_time(onset, sr=sr),  # Convert onset time
                        librosa.frames_to_time(offset, sr=sr),  # Convert offset time
                        freq,  # Frequency of this bin
                    )
                )





    # Évaluation avec mir_eval
    ref_intervals = np.array([[start, end] for start, end, _ in notes])

    ref_pitches = np.array([librosa.hz_to_midi(freq) for _, _, freq in notes])

    est_intervals = np.array([[start, end] for start, end, _ in estimated_notes])
    est_pitches = np.array([librosa.hz_to_midi(freq) for _, _, freq in estimated_notes])

    onset_tolerance = 0.5  # 50 ms
    pitch_tolerance = 0.25  # quart de ton
    offset_ratio = 0.2  # 20% de la durée de la note

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