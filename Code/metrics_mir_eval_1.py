import numpy as np
import librosa
import mir_eval
import matplotlib.pyplot as plt

# Fonction pour générer un signal audio synthétique
def generate_audio(duration, sr, notes):
    audio = np.zeros(int(duration * sr))
    for start, end, freq in notes:
        num_samples = int((end - start) * sr)
        t = np.linspace(start, end, num_samples, endpoint=False)
        # Check if freq is a scalar or an array
        if np.isscalar(freq):
            # If freq is a scalar, proceed as before
            audio[int(start * sr):int(start * sr) + num_samples] += np.sin(2 * np.pi * freq * t)
        else:
            # If freq is an array, generate a signal for each frequency and sum them
            signal = np.zeros_like(t)
            for f in freq:
                signal += np.sin(2 * np.pi * f * t)
            audio[int(start * sr):int(start * sr) + num_samples] += signal
    return audio

# Fonction pour simuler le modèle NMP
def simulate_nmp(audio_path, sr):
    audio, _ = librosa.load(audio_path, sr=sr) # Charger l'audio
    S = librosa.stft(audio)
    times = librosa.times_like(S)
    freqs = librosa.fft_frequencies(sr=sr)

    Yo = np.zeros_like(S, dtype=float)
    Yn = np.zeros_like(S, dtype=float)
    Yp = np.zeros_like(S, dtype=float)

    for start, end, freq in notes:
        idx_start = np.argmin(np.abs(times - start))
        idx_end = np.argmin(np.abs(times - end))

        # If freq is an array, find closest frequency for each element
        if isinstance(freq, np.ndarray):
            idx_freq = [np.argmin(np.abs(freqs - f)) for f in freq]
            # For vibrato, we activate the closest frequency bin for each time step
            for i, idx_f in enumerate(idx_freq):
                Yo[idx_f, idx_start + i] = 1  # Assuming idx_start + i is within bounds
                Yn[idx_f, idx_start + i] = 1  # Assuming idx_start + i is within bounds
                Yp[idx_f - 1:idx_f + 2, idx_start + i] = 1  # Assuming idx_start + i is within bounds
        else:
            # If freq is a single value, proceed as before
            idx_freq = np.argmin(np.abs(freqs - freq))
            Yo[idx_freq, idx_start] = 1
            Yn[idx_freq, idx_start:idx_end] = 1
            Yp[idx_freq - 1:idx_freq + 2, idx_start:idx_end] = 1

    return Yo, Yn, Yp

# Paramètres
sr = 44100
#duration = 5

# Scénarios de test
scenarios = {
    "Note isolée": [(1, 2, 440)],
    "Accord": [(1, 2, 440), (1, 2, 550), (1, 2, 660)],
    "Mélodie rapide": [(1, 1.2, 440), (1.3, 1.5, 494), (1.6, 1.8, 523), (1.9, 2.1, 587)],
    "Vibrato": [(1, 2, 440 + 5 * np.sin(2 * np.pi * 5 * np.linspace(0, 1, 100)))],
    "Glissando": [(1, 2, np.linspace(440, 880, 100))]
}

# Tester chaque scénario
'''for name, notes in scenarios.items():
    print(f"\nTest: {name}")

    # Générer l'audio
    audio = generate_audio(duration, sr, notes)'''

    # Simuler le modèle NMP
    Yo, Yn, Yp = simulate_nmp(audio, sr)

    # Post-traitement pour obtenir les notes estimées
    onset_threshold = 0.1
    note_threshold = 0.1

    estimated_notes = []
    for f in range(Yo.shape[0]):
        onsets = np.where(Yo[f] > onset_threshold)[0]
        for onset in onsets:
            offset = onset + np.argmax(Yn[f, onset:] < note_threshold)
            if offset == onset:
                offset = len(Yn[f])
            estimated_notes.append((librosa.frames_to_time(onset),
                                    librosa.frames_to_time(offset),
                                    librosa.fft_frequencies(sr=sr)[f]))

    # Évaluation avec mir_eval
    ref_intervals = np.array([[start, end] for start, end, _ in notes])
    # Handle multi-pitch scenarios
    if name in ["Vibrato", "Glissando"]:
        # Take the mean pitch for multi-pitch scenarios
        ref_pitches = np.array([librosa.hz_to_midi(np.mean(freq)) if isinstance(freq, np.ndarray) else librosa.hz_to_midi(freq) for _, _, freq in notes])
    else:
        ref_pitches = np.array([librosa.hz_to_midi(freq) for _, _, freq in notes])


    est_intervals = np.array([[start, end] for start, end, _ in estimated_notes])
    est_pitches = np.array([librosa.hz_to_midi(freq) for _, _, freq in estimated_notes])

    onset_tolerance = 0.1  # 50 ms
    pitch_tolerance = 0.5  # quart de ton
    offset_ratio = 0.3 # 20% de la durée de la note

    precision, recall, f_measure, average_overlap_ratio = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_pitches, est_intervals, est_pitches,
        onset_tolerance=onset_tolerance, pitch_tolerance=pitch_tolerance,
        offset_ratio=offset_ratio
    )

    precision_no, recall_no, f_measure_no,  average_overlap_ratio_no  = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_pitches, est_intervals, est_pitches,
        onset_tolerance=onset_tolerance, pitch_tolerance=pitch_tolerance,
        offset_ratio=None  # Ignorer les offsets
    )

    frame_precision, frame_recall, frame_accuracy = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_pitches, est_intervals, est_pitches,
        offset_ratio=None,  # to calculate accuracy based on onsets and pitches only
        onset_tolerance=0.05,  # adjust tolerance as needed
        pitch_tolerance=0.25  # adjust tolerance as needed
    )[0:3]  # to extract only precision, recall, accuracy # changed from mir_eval.transcription_velocity to mir_eval.transcripti

    print(f"F-measure: {f_measure:.3f}")
    print(f"F-measure (no offset): {f_measure_no:.3f}")
    print(f"Frame-level accuracy: {frame_accuracy:.3f}")

    # Visualisation
    plt.figure(figsize=(12, 8))
    plt.subplot(311)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(audio))),
                             sr=sr, x_axis='time', y_axis='hz')
    plt.title('Spectrogramme')
    plt.subplot(312)
    librosa.display.specshow(Yn, sr=sr, x_axis='time', y_axis='hz')
    plt.title('Yn (Note Activation)')
    plt.subplot(313)
    librosa.display.specshow(Yp, sr=sr, x_axis='time', y_axis='hz')
    plt.title('Yp (Pitch Activation)')
    plt.tight_layout()
    plt.show()

import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def create_nmp_model(input_shape):
    inputs = layers.Input(shape=input_shape)

    # Couches convolutives pour l'extraction de caractéristiques
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)

    # Sorties multiples
    yo = layers.Conv2D(1, (1, 1), activation='sigmoid', name='onset')(x)
    yn = layers.Conv2D(1, (1, 1), activation='sigmoid', name='note')(x)
    yp = layers.Conv2D(3, (1, 1), activation='sigmoid', name='pitch')(x)

    model = models.Model(inputs=inputs, outputs=[yo, yn, yp])
    return model

def preprocess_audio(audio_path, sr=22050, hop_length=512, n_bins=252):
    y, _ = librosa.load(audio_path, sr=sr)
    C = librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=n_bins, bins_per_octave=36)
    C = librosa.amplitude_to_db(np.abs(C), ref=np.max)
    return C

def post_process(yo, yn, yp, onset_threshold=0.5, note_threshold=0.5):
    notes = []
    for f in range(yo.shape[0]):
        onsets = np.where(yo[f] > onset_threshold)[0]
        for onset in onsets:
            offset = onset + np.argmax(yn[f, onset:] < note_threshold)
            if offset == onset:
                offset = yn.shape[1]
            pitch = f * 3 + np.argmax(yp[f, onset:offset].mean(axis=0))
            notes.append((onset, offset, librosa.midi_to_hz(pitch / 3 + 21)))
    return notes

# Utilisation du modèle
input_shape = (252, None, 1)  # Ajustez selon vos besoins
model = create_nmp_model(input_shape)
model.compile(optimizer='adam', loss='binary_crossentropy')

# Chargez ici un modèle pré-entraîné si disponible
# model.load_weights('chemin_vers_poids_pre_entraines.h5')

def transcribe_audio(audio_path):
    C = preprocess_audio(audio_path)
    C = np.expand_dims(C, axis=-1)
    C = np.expand_dims(C, axis=0)

    yo, yn, yp = model.predict(C)
    yo = yo[0, :, :, 0]
    yn = yn[0, :, :, 0]
    yp = yp[0, :, :, :]

    notes = post_process(yo, yn, yp)
    return notes

# Exemple d'utilisation
audio_path = 'chemin_vers_votre_fichier_audio.wav'
transcribed_notes = transcribe_audio(audio_path)
print(transcribed_notes)