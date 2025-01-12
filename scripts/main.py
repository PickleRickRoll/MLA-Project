import numpy as np
import tensorflow as tf
from Data_partition import extract_audio_segments,process_all_midi_files
from dsp_utils import cqt, harmonic_stack, dsp, vis_cqt
from modelv1 import model_v1
from preprocessing_utils import create_dataset , generate_frequency_bins,process_audio,process_midi
from variables import *
from tqdm.keras import TqdmCallback
from tensorflow.keras.models import load_model
from evaluation import mir_eval
import matplotlib.pyplot as plt




if __name__ == "__main__":


    """
    
    Traing Part of the Script
    
    """

    #'''
    freq_bins1=generate_frequency_bins(int(n_bins/3),sample_rate,int(bins_per_octave/3),f_min)
    freq_bins2=generate_frequency_bins(n_bins,sample_rate,bins_per_octave,f_min)
    extract_audio_segments(raw_dir, train_audio_dir , sample_rate, segment_length)
    process_all_midi_files(raw_dir, train_midi_dir, segment_length)
    # Load training data
    X_train, Y_train = create_dataset(
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
        batch_size=18,
        epochs=100,
        validation_split=0.2,
        callbacks=[TqdmCallback(verbose=1)],
    )
    
    print("########### Training Done ########## ")

    model.save(model_path)

    print("########### Model Saved ########## ")
    #'''

        # Extract training history data
    history_dict = history.history

    # Plot loss evolution
    plt.figure(figsize=(10, 4))
    plt.plot(history_dict['loss'], label='Training Loss')
    plt.plot(history_dict['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Evolution')
    plt.legend()
    plt.grid(True)
    plt.show()


    """

    You can comment the part above , so you can run the script for testing 
    without trainin

    """
    
    model = load_model(model_path)

    print("########### Model Loaded ########## ")
    
    #recreation of x_ train and y_train for a sample
    result=process_audio(
    path_train,
    f_min,
    n_bins,
    bins_per_octave,
    harmonics,
    bins_per_semitone,
    sample_rate,
    hop_size,
    segment_length,
)
    input_sample = np.expand_dims(result, axis=0)

    freq_bins1=generate_frequency_bins(int(n_bins/3),sample_rate,int(bins_per_octave/3),f_min)
    freq_bins2=generate_frequency_bins(n_bins,sample_rate,bins_per_octave,f_min)
    Yot,Ynt,Ypt=process_midi(path_ytrain, num_frames, freq_bins1,freq_bins2,time_resolution)
    grd_truth=[Yot,Ynt,Ypt]
    # Evaluate the model on the test sample
    output = model.predict(input_sample)
    #temp=[output[0][0],output[1][0],output[2][0]] # to verify the proper working of the function
    f_measure, fmeasure_no, frame_accuracy = mir_eval(grd_truth,output,sample_rate)

    # Display metrics
    print(f"F-measure: {f_measure:.3f}")
    print(f"F-measure (no offset): {fmeasure_no:.3f}")
    print(f"Frame-level accuracy: {frame_accuracy:.3f}")

    vis_cqt(Yot, sample_rate, hop_size, bins_per_semitone, "Yo ground truth posteriogram", cond=True)
    vis_cqt(output[0][0], sample_rate, hop_size, bins_per_semitone/3, "Yo estimated posteriogram", cond=True)

    vis_cqt(Ynt, sample_rate, hop_size, bins_per_semitone, "Yn ground truth posteriogram", cond=True)
    vis_cqt(output[1][0], sample_rate, hop_size, bins_per_semitone/3, "Yn estimated posteriogram", cond=True)
    
    vis_cqt(Ypt, sample_rate, hop_size, bins_per_semitone, "Yp ground truth posteriogram", cond=True)
    vis_cqt(output[2][0], sample_rate, hop_size, bins_per_semitone, "Yp estimated posteriogram", cond=True)


    #'''