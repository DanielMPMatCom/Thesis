import os
import numpy as np
import librosa


def filter_noise(input_dir, output_dir):
    """
    Filters noise from mel-spectrogram .npy files in the input directory and saves the processed files to the output directory.
    For each .npy file in the input directory:
        - Loads the mel-spectrogram matrix.
        - Calculates the 99.9th percentile value as a reference.
        - Converts the matrix to a decibel scale using the reference.
        - Sets all values below 0 dB to 0 (filters noise).
        - Saves the filtered matrix to the output directory with the same filename.
    Parameters:
        input_dir (str): Path to the directory containing input .npy files.
        output_dir (str): Path to the directory where filtered .npy files will be saved. Created if it does not exist.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load .npy files, apply percentile filter, and save results
    for filename in os.listdir(input_dir):
        if filename.endswith(".npy"):
            # Load mel-spectrogram matrix
            data = np.load(os.path.join(input_dir, filename))

            # Calculate the 99.9 percentile reference
            ref = np.percentile(data, 99.9)

            # Convert to decibel scale using the 99.9 percentile reference
            data_db = librosa.power_to_db(data, ref=ref)

            # Apply filter
            data_db[data_db < 0] = 0

            # Save filtered mel-spectrogram matrix
            np.save(os.path.join(output_dir, filename), data_db)
