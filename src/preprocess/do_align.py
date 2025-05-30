import matplotlib.pyplot as plt
import itertools as it
import pandas as pd
import numpy as np
import librosa
import random
import scipy
import json
import os
from scipy.signal import correlate
from scipy.ndimage import maximum_filter

hop_length, n_fft, n_mels = 512, 6096, 128
f_min, f_max = 1600, 4096
sr = 96000
hop_per_sec_factor = sr / hop_length
neighborhood_size = (5, int(hop_per_sec_factor * 0.3))
peak_threshold = 10


def get_best_alignment(
    v_a: np.ndarray,
    v_b: np.ndarray,
    delta_time_threshold: int = 10,
    max_filter_width: int = 1,
) -> tuple:
    """
    Compute the best alignment shift between two sets of event times.
    This function generates histograms for the provided event time arrays, v_a and v_b,
    using bins defined by the minimum and maximum values in the arrays and the specified
    delta_time_threshold. It then applies a maximum filter to smooth the histograms and
    calculates their cross-correlation to find the optimal alignment lag. The function
    returns this optimal shift along with the corresponding number of matching events
    (as indicated by the peak cross-correlation value).
    Parameters:
        v_a (array-like): Array or sequence of event times for the first signal.
        v_b (array-like): Array or sequence of event times for the second signal.
        delta_time_threshold (float, optional): The width of each time bin used for
            generating histograms. Default is 10.
        max_filter_width (int, optional): The width of the maximum filter applied to
            the histograms to smooth the data. Default is 1.
    Returns:
        tuple: A tuple containing:
            - shift (float): The calculated alignment shift, computed based on the
              optimal lag in the cross-correlation.
            - max_matches (int or float): The maximum value of the computed cross-correlation,
              representing the number of matching events.
    """

    bins = np.arange(
        min(v_a.min(), v_b.min()), max(v_a.max(), v_b.max()), delta_time_threshold
    )
    hist_a, _ = np.histogram(v_a, bins=bins)
    hist_a = maximum_filter(hist_a, max_filter_width)
    hist_b, _ = np.histogram(v_b, bins=bins)
    hist_b = maximum_filter(hist_b, max_filter_width)
    cross_corr = correlate(hist_a, hist_b, mode="full")
    shift_index = np.argmax(cross_corr)

    # Compute the corresponding shift
    shift = (shift_index - len(hist_a) + 1) * delta_time_threshold
    max_matches = cross_corr[shift_index]
    # shift, max_matches
    return shift, max_matches


def compute_global_alignment(pairs: list, num_signals: int) -> np.ndarray:
    """
    Compute global alignments for a collection of signals using pairwise delta differences.

    This function takes a set of pairwise signal comparisons and uses a depth-first search
    to propagate alignment differences across all signals. Each pair in the input defines
    a relative alignment (delta) between two signals. The algorithm assumes that signal 0
    is the reference (i.e., its alignment is fixed at 0), and computes the alignments for
    all other signals consistent with the given pairwise differences.

    Parameters:
        pairs (iterable of tuples): An iterable where each element is a tuple of the form
            (signal1, signal2, delta), indicating that the alignment of signal2 is offset by
            'delta' relative to signal1.
        num_signals (int): The total number of signals. This specifies the size of the alignment
            array and the range of signal identifiers.

    Returns:
        numpy.ndarray: A NumPy array of length `num_signals` where each entry corresponds to
        the computed alignment (delta value) for the respective signal.

    Notes:
        - The function constructs an undirected graph with weighted edges based on the provided
          pairs. Each edge includes a delta for one direction and a negative delta for the opposite
          direction.
        - Depth-first search (DFS) is used to traverse the graph, ensuring each signal's alignment
          is computed relative to the reference signal.
    """
    # Initialize alignment deltas (set all to 0 initially)
    alignments = np.zeros(num_signals)

    # Create a graph structure: adjacency matrix and visited tracker
    adj_matrix = {i: [] for i in range(num_signals)}

    for signal1, signal2, delta in pairs:
        adj_matrix[signal1].append((signal2, delta))
        adj_matrix[signal2].append((signal1, -delta))

    # Perform BFS or DFS to propagate deltas
    visited = [False] * num_signals

    def dfs(node: int, accumulated_delta: float) -> None:
        """
        Performs a depth-first search traversal starting from the given node, while accumulating delta values along the path.

        This recursive function:
        - Marks the current node as visited.
        - Records the accumulated delta for the current node in the 'alignments' dictionary.
        - Recursively visits each unvisited neighbor, updating the accumulated delta with the corresponding delta value from the adjacency matrix.

        Parameters:
            node (int): The identifier of the current node from which DFS is initiated.
            accumulated_delta (numeric): The running sum of delta values accumulated from the starting node to the current node.

        Side Effects:
            Updates the 'visited' dictionary to mark nodes as visited.
            Modifies the 'alignments' dictionary by setting each node's corresponding accumulated delta.
        """
        visited[node] = True
        alignments[node] = accumulated_delta

        for neighbor, delta in adj_matrix[node]:
            if not visited[neighbor]:
                dfs(neighbor, accumulated_delta + delta)

    # Assume signal 0 is the reference (alignment = 0)
    dfs(0, 0)

    return alignments


def process_folder(base_mel_path: str) -> np.ndarray:
    """
    Processes a folder of Mel spectrogram data files and computes a global alignment based on significant peaks.

    This function loads all NumPy files in the specified base directory (ignoring files that start with '!'),
    converts the amplitude spectrogram to decibel scale, and identifies significant peaks using a local maximum filter.
    For each pair of microphone channels, it computes the best alignment based on the time peaks found.
    Subsequently, the alignment adjustments are aggregated into a global alignment across all signals,
    and the resulting alignment is saved to an 'align.json' file within the base directory.

    Parameters:
        base_mel_path (str): The path to the directory containing .npy files with Mel spectrogram data.

    Returns:
        numpy.ndarray: An array representing the global alignment computed from the input files.

    Notes:
        - Only files with a '.npy' extension and that do not start with '!' are processed.
        - The function uses external libraries such as librosa, scipy, and numpy, and assumes that these are available.
        - The computation involves filtering the spectrogram to find significant peaks above a predefined decibel threshold,
          aligning these peaks between pairs of microphone channels, and then globally aligning all signals.
    """
    raw_mel_dbs = {}
    for file in os.listdir(base_mel_path):
        if file.endswith(".npy") and file[0] != "!":
            file_name, file_ext = os.path.splitext(file)
            mic_idx = int(file_name[-2:])
            mel = np.load(os.path.join(base_mel_path, file))
            mel_db = librosa.amplitude_to_db(mel, ref=np.max)
            raw_mel_dbs[mic_idx] = mel_db

    peaks_to_align = {}

    db_threshold = -40
    neighborhood_size = (7, int(hop_per_sec_factor * 0.15))
    for mic_id in sorted(raw_mel_dbs):
        mel_db = raw_mel_dbs[mic_id][80:, :]
        data_max = scipy.ndimage.maximum_filter(mel_db, neighborhood_size)
        peaks = mel_db == data_max
        significant_peaks = peaks & (data_max >= db_threshold)
        p_frec, p_time = np.where(significant_peaks)
        p_value = mel_db[significant_peaks]
        peaks_to_align[mic_id] = p_frec, p_time, p_value

    all_mic_ids = list(peaks_to_align)
    all_pairs = {}
    for i, (id1, id2) in enumerate(it.product(all_mic_ids, all_mic_ids), start=1):
        if id1 != id2:
            all_pairs[(id1, id2)] = get_best_alignment(
                peaks_to_align[id1][1], peaks_to_align[id2][1], 1
            )
        if i % 10 == 0:
            print("- iterations:", i)

    num_signals = 9

    mic_id_to_idx = {mic_id: idx for idx, mic_id in enumerate(all_mic_ids)}
    items = list(all_pairs.items())
    pairs = sorted(items, key=lambda p: p[1][1], reverse=True)
    pairs = [
        (mic_id_to_idx[mic1], mic_id_to_idx[mic2], delta)
        for (mic1, mic2), (delta, num) in pairs
        if num > 20
    ]

    global_alignments = compute_global_alignment(pairs, num_signals)
    print("**Alignment: ", global_alignments)

    with open(os.path.join(base_mel_path, "align.json"), "w") as f:
        json.dump(global_alignments.tolist(), f)

    return global_alignments


def do_align_pipeline(base_path: str, verbose: bool = False) -> None:
    """
    Process mel data folders found in the given base path.

    This function iterates over each file or folder in the specified base_path. For each entry, it builds
    the full path and prints its name if verbose output is enabled. Finally, it delegates the actual
    processing of each folder to the process_folder function.

    Parameters:
        base_path (str): The directory that contains the mel data files/folders.
        verbose (bool, optional): If True, additional output indicating processing progress will be printed.
                                  Defaults to False.

    Returns:
        None
    """
    for relative_mel_path in os.listdir(base_path):
        if verbose:
            print("Processing", relative_mel_path)
        base_mel_path = os.path.join(base_path, relative_mel_path)
        process_folder(base_mel_path)


def apply_lags_to_saved_files(
    lags: dict, path_to_files_to_be_synchronized: str, data_destination_path: str = None
) -> dict:
    """
    Apply time lags to .npy files and save the synchronized files to a specified directory.
    Parameters:
    lags (dict): A dictionary where keys are filenames and values are the lag values to be applied.
    path_to_files_to_be_synchronized (str): The path to the directory containing the .npy files to be synchronized.
    data_destination_path (str, optional): The path to the directory where the synchronized files will be saved.
                                           If None, a default directory 'aligned/all' will be used.
    Returns:
    dict: A dictionary where keys are filenames and values are the processed numpy arrays with applied lags.
    """

    # List all .npy files in the specified directory
    files = [
        f for f in os.listdir(path_to_files_to_be_synchronized) if f.endswith(".npy")
    ]

    # Define the directory to save the processed files
    if data_destination_path is not None:
        aligned_files = data_destination_path
    else:
        aligned_files = f"{os.path.dirname(os.path.abspath(__file__))}/aligned/all"

    # Create the directory if it doesn't exist
    os.makedirs(aligned_files, exist_ok=True)

    aligned_files_dict = {}

    # Process each file
    for file in files:
        # Load the .npy file
        file_data = np.load(os.path.join(path_to_files_to_be_synchronized, file))

        # Check if the file has an associated lag value
        if file in lags:
            lag = lags[file]
            # If the lag is positive, pad the beginning of the array and truncate the end
            if lag > 0:
                if file_data.ndim == 1:
                    file_data = np.pad(file_data, (lag, 0), "constant")[
                        : len(file_data)
                    ]
                elif file_data.ndim == 2:
                    file_data = np.pad(file_data, ((0, 0), (lag, 0)), "constant")[
                        : len(file_data), :
                    ]
            # If the lag is negative, pad the end of the array and truncate the beginning
            elif lag < 0:
                if file_data.ndim == 1:
                    file_data = np.pad(file_data, (0, -lag), "constant")[
                        -len(file_data) :
                    ]
                elif file_data.ndim == 2:
                    file_data = np.pad(file_data, ((0, 0), (0, -lag)), "constant")[
                        -len(file_data) :, :
                    ]

        # Save the processed data to the new directory
        np.save(os.path.join(aligned_files, file), file_data)

        aligned_files_dict[file] = file_data

    # Return the modified dictionary with adjusted peaks
    return aligned_files_dict
