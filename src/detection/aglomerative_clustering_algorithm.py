import os
import json
import librosa
import numpy as np
import pandas as pd
import seaborn as sns
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import maximum_filter
from scipy.ndimage import uniform_filter1d


hop_length, n_fft, n_mels = 512, 6096, 128
f_min, f_max = 1600, 4096
sr = 96000
time_factor = hop_length / sr


def create_image_from_max(spect: np.ndarray) -> np.ndarray:
    """
    Converts a spectrogram to decibel (dB) scale and flips it vertically.

    Parameters:
        spect (np.ndarray): Input power spectrogram.

    Returns:
        np.ndarray: The spectrogram converted to dB scale and flipped along the vertical axis.
    """
    Xdb = librosa.power_to_db(spect)
    return np.flip(Xdb, axis=0)


def show_spect(spect: np.ndarray, **kwargs) -> None:
    """
    Displays a spectrogram using librosa's specshow function.

    Parameters:
        spect (np.ndarray): The spectrogram to display.
        **kwargs: Additional keyword arguments passed to librosa.display.specshow.

    Notes:
        This function assumes that the variables `sr`, `f_min`, and `f_max` are defined in the global scope.
        The x-axis is set to 'time' and the y-axis to 'mel' by default.
    """
    librosa.display.specshow(
        spect, x_axis="time", y_axis="mel", sr=sr, fmin=f_min, fmax=f_max, **kwargs
    )


def spect_zone(spect: np.ndarray, time_ini: tuple, time_end: tuple) -> np.ndarray:
    """
    Extracts a time-based zone from a spectrogram array.

    Parameters:
        spect (np.ndarray): The spectrogram array with time along the second axis.
        time_ini (tuple): Start time as a tuple (minutes, seconds).
        time_end (tuple): End time as a tuple (minutes, seconds).

    Returns:
        np.ndarray: The sliced spectrogram corresponding to the specified time interval.

    Note:
        The variable 'time_factor' must be defined in the scope where this function is used,
        representing the duration (in seconds) of each time step in the spectrogram.
    """
    m1, s1 = time_ini
    m2, s2 = time_end
    return spect[
        :, int((m1 * 60 + s1) / time_factor) : int((m2 * 60 + s2) / time_factor)
    ]


def show_spect_bt(
    spect: np.ndarray, time_ini: float, time_end: float, **kwargs
) -> None:
    """
    Displays a segment of a spectrogram between specified start and end times.

    Parameters:
        spect (np.ndarray): The input spectrogram array.
        time_ini (float): The initial time (in seconds) of the segment to display.
        time_end (float): The ending time (in seconds) of the segment to display.
        **kwargs: Additional keyword arguments passed to librosa.display.specshow.

    Returns:
        None
    """
    to_show = spect_zone(spect, time_ini, time_end)
    librosa.display.specshow(
        to_show, x_axis="time", y_axis="mel", sr=sr, fmin=f_min, fmax=f_max, **kwargs
    )


def shift_array_2d(arr: np.ndarray, dx: int, filling_value: int = -80):
    """
    Shifts the contents of a 2D NumPy array horizontally by a specified number of columns.

    Parameters:
        arr (np.ndarray): The input 2D array to be shifted.
        dx (int): The number of columns to shift. Positive values shift to the right, negative values shift to the left, and zero means no shift.
        filling_value (int): The value to fill in the vacated positions after the shift. Defaults to -80.

    Returns:
        np.ndarray: The shifted 2D array with vacated positions filled by `filling_value`.

    Examples:
        >>> arr = np.array([[1, 2, 3], [4, 5, 6]])
        >>> shift_array_2d(arr, 1, filling_value=0)
        array([[0, 1, 2],
               [0, 4, 5]])
    """
    if dx > 0:  # Shift right
        result = np.empty_like(arr)
        result[:, :dx] = filling_value  # Fill the leftmost `dx` values with 0
        result[:, dx:] = arr[:, :-dx]
    elif dx < 0:  # Shift left
        result = np.empty_like(arr)
        result[:, dx:] = filling_value  # Fill the rightmost `|dx|` values with 0
        result[:, :dx] = arr[:, -dx:]
    else:  # dx == 0, no shift
        result = arr.copy()
    return result


# with open('../dataset/all_alignments.json', 'r') as f:
#     all_alignments = json.load(f)

# mic_idxs = {mic: idx for idx, mic in enumerate([10, 12, 13, 14, 15, 16, 17, 18, 19])}


def load_file(
    datetime: str,
    mic: int,
    all_alignments: dict,
    mic_idxs: dict = {
        mic: idx for idx, mic in enumerate([10, 12, 13, 14, 15, 16, 17, 18, 19])
    },
) -> np.ndarray:
    """
    Loads a spectrogram file for a given datetime and microphone, applies alignment shift, and returns the shifted array.

    Parameters:
        datetime (str): The name of the datetime/session directory.
        mic (int): The microphone ID.
        all_alignments (dict): Dictionary containing alignment deltas for each datetime and mic.
        mic_idxs (dict, optional): Mapping from mic ID to index in the alignment array. Defaults to mapping for mics 10-19.

    Returns:
        np.ndarray: The shifted spectrogram array for the specified mic and datetime.
    """
    file_path = f"../dataset/{datetime}/{mic}.npy"
    result = np.load(file_path)
    delta = all_alignments[datetime][mic_idxs[mic]]
    return shift_array_2d(result, int(delta), 0)


def minsec_to_idx(min, sec):
    return int((min * 60 + sec) / time_factor)


def minmax(a):
    return np.min(a), np.max(a)


def sec_to_minsec(secs):
    return f"{int(secs)//60}:{secs - int(secs)//60*60:.3f}"


def idx_to_minsec(idx):
    return sec_to_minsec(idx * time_factor)


all_mic_id = [10, 12, 13, 14, 15, 16, 17, 18, 19]


def load_all_spects(datetime_name):
    spects = {}
    for idx in all_mic_id:
        spect = load_file(datetime_name, idx)
        spects[idx] = spect
    return spects


def get_candidate_chants(spects: dict) -> list:
    """
    Extracts candidate chant segments from spectrogram data based on frequency and amplitude criteria.

    Args:
        spects (dict): A dictionary mapping microphone indices to their corresponding spectrogram numpy arrays.
                       Each spectrogram is expected to be a 2D array with shape (frequency_bins, time_steps).

    Returns:
        list of tuples: Each tuple contains:
            - c (int): The time index (in steps of 5) where the candidate chant is detected.
            - minsec (Any): The result of idx_to_minsec(c), representing the time in minutes and seconds.
            - frecs (list of int): The list of frequency bin indices with maximum amplitude for each microphone at time c.
            - vals (list of float): The list of maximum amplitude values for each microphone at time c.

    Notes:
        - Only time indices where the maximum frequency difference across microphones is at most 3,
          the first microphone's frequency is between 80 and 120 (inclusive), and more than one microphone
          has an amplitude greater than 100 are considered as candidate chants.
        - The function relies on the global variables `all_mic_id` (list of microphone indices) and
          `idx_to_minsec` (function to convert time index to minutes and seconds).
    """
    all_data = []
    for c in range(0, 600000, 5):
        frecs = []
        vals = []
        for idx in all_mic_id:
            spect = spects[idx][:, c]
            frec_id = np.argmax(spect)
            max_val = spect[frec_id]
            frecs.append(frec_id)
            vals.append(max_val)
        if (
            max(frecs) - min(frecs) <= 3
            and 80 <= frecs[0] <= 120
            and sum((1 for v in vals if v > 100)) > 1
        ):
            all_data.append((c, idx_to_minsec(c), frecs, vals))
    return all_data


def get_maximum_chant_power_per_mic(candidate_chants: list) -> np.ndarray:
    """
    Calculates the maximum chant power for each microphone across all candidate chants.

    Each element in `candidate_chants` is expected to be a tuple where the fourth element
    (index 3) is an iterable (e.g., list or array) representing the power values for each microphone.

    Args:
        candidate_chants (Iterable[Tuple[Any, Any, Any, Iterable[float]]]):
            An iterable of tuples, each containing power values per microphone as the fourth element.

    Returns:
        np.ndarray or None: An array containing the maximum power per microphone across all candidate chants,
        or None if `candidate_chants` is empty.
    """
    largest_per_mic = None
    for _, _, _, power in candidate_chants:
        if largest_per_mic is None:
            largest_per_mic = np.array(power)
        else:
            largest_per_mic = np.maximum(largest_per_mic, np.array(power))
    return largest_per_mic


def normalize_spects(spects, refs):
    spects_db = {}
    for idx, mic_id in enumerate(spects):
        as_db = librosa.power_to_db(spects[mic_id], ref=refs[idx])
        spects_db[mic_id] = as_db
    return spects_db


def extract_chants(
    norm_spect: np.ndarray,
    n_size: tuple = (100, 300),
    db_threshold: float = -30,
    frec_bounds: tuple = (70, 130),
) -> pd.DataFrame:
    """
    Extracts significant chant peaks from a normalized spectrogram within specified frequency bounds.

    Parameters:
        norm_spect (np.ndarray): The normalized spectrogram (frequency x time).
        n_size (tuple, optional): Size of the neighborhood for local maximum filtering (default is (100, 300)).
        db_threshold (float, optional): Minimum decibel threshold for peak detection (default is -30).
        frec_bounds (tuple, optional): Frequency bounds (start, end) for analysis (default is (70, 130)).

    Returns:
        pd.DataFrame: DataFrame containing detected peaks with columns:
            - 'frec': Frequency index of the peak.
            - 'time': Time index of the peak.
            - 'power': Normalized power of the peak.
            - 'time_min': Time in minutes (requires global variable 'time_factor').
    """
    spect_copy = np.ones_like(norm_spect) * -80
    spect_copy[frec_bounds[0] : frec_bounds[1]] = norm_spect[
        frec_bounds[0] : frec_bounds[1]
    ]
    norm_spect = spect_copy
    data_max = maximum_filter(norm_spect, n_size)
    peaks = norm_spect == data_max
    significant_peaks = peaks & (data_max >= db_threshold)
    p_frec, p_time = np.where(significant_peaks)
    min_val = min(db_threshold, np.min(norm_spect))
    p_value = (norm_spect[p_frec, p_time] - min_val) / (0 - min_val)
    result = pd.DataFrame(
        [
            (f, t, v)
            for f, t, v in zip(p_frec, p_time, p_value)
            if frec_bounds[0] <= f <= frec_bounds[1]
        ],
        columns=["frec", "time", "power"],
    )
    result["time_min"] = result.time * time_factor / 60
    return result


def select_co_lin_pairs(
    co_chants: pd.DataFrame, lin_chants: pd.DataFrame
) -> pd.DataFrame:
    """
    Selects and pairs co_chants and lin_chants based on a time difference criterion.

    For each entry in lin_chants, finds the first co_chant whose 'time_min' is within a specified delta range
    relative to the lin_chant's 'time_min'. The selected pairs are returned as a merged DataFrame.

    Args:
        co_chants (pd.DataFrame): DataFrame containing co_chant events with a 'time_min' column.
        lin_chants (pd.DataFrame): DataFrame containing lin_chant events with a 'time_min' column.

    Returns:
        pd.DataFrame: Merged DataFrame containing paired co_chants and lin_chants, with suffixes "_co" and "_lin"
        for their respective columns. Only pairs where a matching co_chant was found are included.
    """
    co_lin_delta = (0.0015, 0.004)
    sel_co_chants = []
    sel_lin_chants = []
    for idx in range(len(lin_chants)):
        lin = lin_chants.iloc[idx]
        delta_time = lin.time_min - co_chants.time_min
        coss = co_chants[
            (delta_time > co_lin_delta[0]) & (delta_time < co_lin_delta[1])
        ].sort_values("time_min")
        if len(coss) <= 0:
            continue
        sel_co_chants.append(coss.iloc[0])
        sel_lin_chants.append(lin)
    common_index = range(len(sel_co_chants))
    sel_co_chants = pd.DataFrame(sel_co_chants, index=common_index)
    sel_lin_chants = pd.DataFrame(sel_lin_chants, index=common_index)
    return pd.merge(
        sel_co_chants,
        sel_lin_chants,
        left_index=True,
        right_index=True,
        how="outer",
        suffixes=("_co", "_lin"),
    )


def filter_colin_pairs3(
    colin_pairs: pd.DataFrame, max_pair_distance: float = 8
) -> pd.DataFrame:
    """
    Filters and clusters pairs of colinear points based on their proximity and average power.

    This function takes a DataFrame of colinear pairs and groups them into clusters ("frogs") such that
    each new point is added to the closest existing cluster if it is within a specified maximum distance.
    Clusters are then filtered to retain only those with at least 5 points and an average 'power_lin' greater than 0.7.
    The cluster with the highest mean 'power_lin' is returned.

    Args:
        colin_pairs (pd.DataFrame): DataFrame containing at least the columns 'frec_co', 'frec_lin', and 'power_lin'.
        max_pair_distance (float, optional): Maximum squared Euclidean distance between points to be considered in the same cluster. Defaults to 8.

    Returns:
        pd.DataFrame or list: The DataFrame of the selected cluster with the highest mean 'power_lin', or an empty list if no suitable cluster is found.
    """
    chants = list(colin_pairs.itertuples())
    frogs = [[chants[0]]]
    for ch in chants[1:]:
        dist_frogs = sorted(
            [
                (
                    (ch.frec_co - f[-1].frec_co) ** 2
                    + (ch.frec_lin - f[-1].frec_lin) ** 2,
                    f,
                )
                for f in frogs
            ]
        )
        f_v, f = dist_frogs[0]
        if f_v <= max_pair_distance:
            f.append(ch)
        else:
            frogs.append([ch])
    selected_frogs = sorted(
        [
            df
            for df in [pd.DataFrame(f) for f in frogs if len(f) >= 5]
            if df.power_lin.mean() > 0.7
        ],
        key=lambda d: -d.power_lin.mean(),
    )
    if len(selected_frogs) > 0:
        return selected_frogs[0]
    return []


def get_chants_bounds(
    colin_pairs: pd.DataFrame, split_secs: int = 20, min_split_count: int = 3
) -> tuple:
    """
    Identifies contiguous time intervals ("chants") in a sequence of time-stamped events based on histogram binning.

    Parameters:
        colin_pairs (pd.DataFrame): DataFrame containing a 'time_min_lin' column with event timestamps (in seconds or minutes).
        split_secs (int, optional): The width of each histogram bin in seconds. Default is 20.
        min_split_count (int, optional): Minimum number of events required in a bin to consider it non-empty. Default is 3.

    Returns:
        range_limits (list of tuple): List of (start_time, end_time) tuples representing the bounds of each detected chant interval.
        ranges (list of tuple): List of (start_bin_index, end_bin_index) tuples corresponding to the bin indices of each interval.

    Notes:
        - Bins with fewer than `min_split_count` events are ignored.
        - Intervals are contiguous regions where bins are non-empty.
    """
    cnts, limits = np.histogram(colin_pairs.time_min_lin, bins=int(3600 / split_secs))
    cnts[cnts < min_split_count] = 0
    non_zero_mask = cnts != 0
    transitions = np.diff(non_zero_mask.astype(int), prepend=0, append=0)
    start_indices = np.where(transitions == 1)[0]
    end_indices = np.where(transitions == -1)[0]
    ranges = list(zip(start_indices, end_indices))
    range_limits = [(limits[li], limits[ls]) for li, ls in ranges]
    return range_limits, ranges


def get_chants(
    base_path: str,
    results_path: str,
    db_threshold_lin: float = -30,
    db_threshold_co: float = -40,
    split_secs: int = 20,
    min_split_count: int = 3,
    min_chanting_size: int = 3,
) -> None:
    """
    Processes all directories in the given base_path, extracts chant events for each mic,
    and saves the results as CSV files in the results_path directory. If a result file
    already exists for a directory, it is skipped.

    Parameters
    ----------
    base_path : str
        Path to the dataset containing subdirectories for each recording session.
    results_path : str
        Path to the directory where result CSV files will be saved.
    db_threshold_lin : float
        dB threshold for linear chant extraction.
    db_threshold_co : float
        dB threshold for co-chant extraction.
    split_secs : int
        Time window (in seconds) for splitting chant events.
    min_split_count : int
        Minimum number of events in a split to be considered.
    min_chanting_size : int
        Minimum number of chant pairs to keep a segment.

    Returns
    -------
    None
        The function saves the results to CSV files and prints progress.
    """
    for path_name in os.listdir(base_path):
        if not os.path.isdir(os.path.join(base_path, path_name)):
            continue
        print("Running", path_name)
        result_file_name = os.path.join(results_path, path_name + ".csv")
        if os.path.exists(result_file_name):
            print("- existing. Ignored!")
            continue

        spects = load_all_spects(path_name)
        candidate_chants = get_candidate_chants(spects)
        maximum_chant_power_per_mic = get_maximum_chant_power_per_mic(candidate_chants)
        normalized_spects = normalize_spects(spects, maximum_chant_power_per_mic)

        all_chantings = []
        for mic_idx in all_mic_id:
            norm_spect_mic = normalized_spects[mic_idx]
            lin_chants = extract_chants(
                norm_spect_mic,
                db_threshold=db_threshold_lin,
                n_size=(30, 300),
                frec_bounds=(70, 110),
            ).sort_values("time_min")
            co_chants = extract_chants(
                norm_spect_mic,
                db_threshold=db_threshold_co,
                n_size=(30, 300),
                frec_bounds=(10, 40),
            ).sort_values("time_min")

            colin_pairs = select_co_lin_pairs(co_chants, lin_chants)
            range_limits, _ = get_chants_bounds(
                colin_pairs, split_secs, min_split_count
            )
            chantings = [
                colin_pairs[colin_pairs.time_min_lin.between(lb, rb)]
                for lb, rb in range_limits
            ]
            chantings = [
                filter_colin_pairs3(ch)
                for ch in chantings
                if len(ch) > min_chanting_size
            ]
            chantings = [ch for ch in chantings if len(ch) > min_chanting_size]
            if chantings:
                chantings = pd.concat(chantings)
                chantings["mic_idx"] = mic_idx
                all_chantings.append(chantings)
        if all_chantings:
            all_chantings = pd.concat(all_chantings)
            all_chantings.to_csv(result_file_name, index=None)
