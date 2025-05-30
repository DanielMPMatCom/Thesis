import os
import glob
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from typing import Tuple, List, Dict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score


# region consistency
def corr_matrix_consistency(correlation_df):
    """
    Plots a heatmap of the given correlation matrix DataFrame with columns reordered based on integer values
    extracted from their names. Highlights the ten largest correlation values in each row with a red rectangle.
    Args:
        correlation_df (pd.DataFrame): A DataFrame representing the correlation matrix, where columns are named
            with an underscore separating two integers (e.g., '1_2').
    Returns:
        None: Displays the heatmap plot with highlighted cells.
    """

    # Reorganize the columns of correlation_df
    ordered_columns = sorted(
        correlation_df.columns,
        key=lambda x: (int(x.split("_")[1]), int(x.split("_")[0])),
    )
    correlation_df_ordered = correlation_df[ordered_columns].reindex(ordered_columns)

    # Plot the heatmap with seaborn without printing the correlation numbers
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_df_ordered, annot=False, cmap="coolwarm", cbar=False)
    plt.title("Correlation Matrix")

    # Highlight the ten largest values per row in the plot
    for i in range(correlation_df_ordered.shape[0]):
        max_indices = correlation_df_ordered.iloc[i].nlargest(10).index
        for idx in max_indices:
            plt.gca().add_patch(
                plt.Rectangle(
                    (correlation_df_ordered.columns.get_loc(idx), i),
                    1,
                    1,
                    fill=False,
                    edgecolor="red",
                    lw=2,
                )
            )
    plt.show()


# regon pattern rate


def load_binary_matrix(csv_path, frog_labels, delta_tau, total_duration):
    """
    Loads a binary matrix from a CSV file indicating frog activity over time intervals.

    Each row in the resulting matrix corresponds to a time interval of length `delta_tau`,
    and each column corresponds to a frog label from `frog_labels`. The matrix is filled
    with 1 where the frog is active at the given time interval, and -1 otherwise.

    Parameters:
        csv_path (str): Path to the CSV file containing columns 'time' and 'frog'.
        frog_labels (list): List of frog labels to include as columns in the matrix.
        delta_tau (float): Duration of each time interval.
        total_duration (float): Total duration to cover with the matrix.

    Returns:
        numpy.ndarray: A (M, N) matrix where M is the number of time intervals and N is the number of frogs.
                       Entries are 1 if the frog is active at the interval, -1 otherwise.
    """
    N = len(frog_labels)
    M = int(np.ceil(total_duration / delta_tau))
    S = -np.ones((M, N), dtype=int)
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        t_idx = int(row["time"] // delta_tau)
        if row["frog"] in frog_labels and 0 <= t_idx < M:
            frog_idx = frog_labels.index(row["frog"])
            S[t_idx, frog_idx] = 1
    return S


def compute_observed_rates(S, delta_tau):
    """
    Computes the observed rates of unique patterns in the input array S.

    Parameters:
        S (np.ndarray): A 2D NumPy array where each row represents an observation and each column represents a variable.
        delta_tau (float): The time interval associated with each observation.

    Returns:
        dict: A dictionary mapping each unique pattern (as a tuple) to its observed rate, calculated as the count of the pattern divided by (M * delta_tau), where M is the number of observations.
    """
    M = S.shape[0]
    patterns, counts = np.unique(S, axis=0, return_counts=True)
    return {tuple(p): cnt / (M * delta_tau) for p, cnt in zip(patterns, counts)}


def compute_independent_rates(S, delta_tau):
    """
    Computes the independent firing rates for all possible binary patterns of N units.

    Given a binary matrix S of shape (M, N), where each row represents a sample and each column a unit (with values -1 or 1),
    this function calculates the probability of each possible pattern under the assumption that units fire independently.
    The result is a dictionary mapping each possible pattern (as a tuple of -1 and 1) to its independent rate, normalized by delta_tau.

    Args:
        S (np.ndarray): An (M, N) array of samples with entries -1 or 1.
        delta_tau (float): Time bin width or normalization factor.

    Returns:
        dict: A dictionary where keys are tuples representing all possible patterns of -1 and 1 of length N,
              and values are the corresponding independent rates (float).
    """
    M, N = S.shape
    X = (S + 1) // 2
    p = X.mean(axis=0)
    indep = {}
    for pattern in itertools.product([-1, 1], repeat=N):
        x = (np.array(pattern) + 1) // 2
        P1 = np.prod(p**x * (1 - p) ** (1 - x))
        indep[pattern] = P1 / delta_tau
    return indep


def compute_ising_rates(J, h, delta_tau):
    """
    Compute the transition rates for all possible states in an Ising model.

    Args:
        J (np.ndarray): Interaction matrix of shape (N, N), where N is the number of spins.
        h (np.ndarray): External field vector of length N.
        delta_tau (float): Time discretization parameter.

    Returns:
        dict: A dictionary mapping each possible spin state (as a tuple) to its normalized transition rate.

    Notes:
        - The function enumerates all 2^N possible spin configurations.
        - The energy of each state is computed as: E = - (s @ h + sum(s * (s @ J)))
        - The rates are normalized by the partition function Z and delta_tau.
    """
    N = len(h)
    states = np.array(list(itertools.product([-1, 1], repeat=N)))
    energies = -(states @ h + np.sum(states * (states @ J), axis=1))
    weights = np.exp(-energies)
    Z = weights.sum()
    return {tuple(states[i]): weights[i] / Z / delta_tau for i in range(len(states))}


def plot_combined(obs_rates, indep_rates, ising_rates, title):
    obs = []
    p1 = []
    p2 = []
    for pattern, r_obs in obs_rates.items():
        if pattern in indep_rates and pattern in ising_rates:
            obs.append(r_obs)
            p1.append(indep_rates[pattern])
            p2.append(ising_rates[pattern])
    obs = np.array(obs)
    p1 = np.array(p1)
    p2 = np.array(p2)

    plt.figure(figsize=(6, 6))
    plt.loglog(obs, p1, "o", alpha=0.6, label="Independent Model")
    plt.loglog(obs, p2, "o", alpha=0.6, label="Ising Model")
    lims = [min(obs.min(), p1.min(), p2.min()), max(obs.max(), p1.max(), p2.max())]
    plt.plot(lims, lims, "k--", label="$y=x$")
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel("Observed Pattern Rate (s$^{-1}$)")
    plt.ylabel("Approximated Pattern Rate (s$^{-1}$)")
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.show()


# Example of use

# if __name__ == '__main__':
#     sample_rate = 96000
#     hop_length = 512
#     delta_tau = hop_length / sample_rate
#     total_duration = 58 * 60 + 2

#     frog_labels = [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0]
#     hour_tag = '20231021_190000'

#     csv_path = f'all_peaks_{hour_tag}.csv'
#     Jij_path = f'../data/Jij_{hour_tag}.npy'
#     hi_path  = f'hi_{hour_tag}.npy'

#     S = load_binary_matrix(csv_path, frog_labels, delta_tau, total_duration)
#     obs_rates = compute_observed_rates(S, delta_tau)
#     indep_rates = compute_independent_rates(S, delta_tau)

#     J = np.load(Jij_path)
#     h = np.load(hi_path)
#     ising_rates = compute_ising_rates(J, h, delta_tau)

# plot_combined(obs_rates, indep_rates, ising_rates,
#             '20231021_190000')


# region Time Comparison

MIC_IDX_TO_FROG = {
    10: 0.0,
    12: 1.0,
    13: 2.0,
    14: 3.0,
    15: 4.0,
    16: 5.0,
    17: 6.0,
    18: 7.0,
    19: 8.0,
}


def load_algorithm_a(file_path: str) -> pd.DataFrame:
    """Loads data from algorithm A."""
    df = pd.read_csv(file_path)
    df = df.rename(
        columns={
            "time": "time_a",
            "frog": "frog_a",
            "energy": "energy_a",
            "frequency band": "freq_a",
        }
    )
    df = df[["time_a", "frog_a", "energy_a", "freq_a"]]
    return df


def load_algorithm_b(file_path: str) -> pd.DataFrame:
    """
    Loads data from algorithm B, only 'lin' columns and maps mic_idx to frog.
    """
    df = pd.read_csv(file_path)
    df = df[["time_min_lin", "power_lin", "mic_idx", "frec_lin"]]
    df["frog_b"] = df["mic_idx"].map(MIC_IDX_TO_FROG)
    df = df.dropna(subset=["frog_b"])
    return df.rename(
        columns={
            "time_min_lin": "time_b",
            "power_lin": "energy_b",
            "frec_lin": "freq_b",
        }
    )


def align_events(
    df_a: pd.DataFrame, df_b: pd.DataFrame, tolerance: float = 1.0
) -> Dict[float, pd.DataFrame]:
    """
    Matches the closest events from A and B by frog. Uses a tolerance in seconds.
    Returns a dictionary {frog_id: DataFrame with columns [time_a, time_b]}.
    """
    aligned_data = {}
    for frog_id in sorted(df_a["frog_a"].unique()):
        a_times = (
            df_a[df_a["frog_a"] == frog_id]["time_a"]
            .sort_values()
            .reset_index(drop=True)
        )
        b_times = (
            df_b[df_b["frog_b"] == frog_id]["time_b"]
            .sort_values()
            .reset_index(drop=True)
        )

        matches = []
        used_b_indices = set()
        for a_time in a_times:
            time_diffs = (b_times - a_time).abs()
            min_diff_idx = time_diffs.idxmin()
            if (
                time_diffs[min_diff_idx] <= tolerance
                and min_diff_idx not in used_b_indices
            ):
                matches.append((a_time, b_times[min_diff_idx]))
                used_b_indices.add(min_diff_idx)

        if matches:
            df_matches = pd.DataFrame(matches, columns=["time_a", "time_b"])
            aligned_data[frog_id] = df_matches

    return aligned_data


def plot_matched_events(aligned_data: Dict[float, pd.DataFrame], title: str):
    """Generates a plot per frog with the correspondence of calls."""
    fig, ax = plt.subplots(figsize=(6, 6))

    for frog_id, df_match in aligned_data.items():
        ax.scatter(
            df_match["time_a"],
            df_match["time_b"],
            label=f"Frog {int(frog_id)}",
            alpha=0.7,
            s=10,
        )

    ax.plot(
        [0, df_match[["time_a", "time_b"]].max().max()],
        [0, df_match[["time_a", "time_b"]].max().max()],
        "k--",
        label="y = x",
    )
    ax.set_xlabel("Time Algorithm A (m)")
    ax.set_ylabel("Time Algorithm B (m)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


# Example of use

# if __name__ == '__main__':
#     path_a = 'all_peaks_20231021_190000.csv'
#     path_b = '../choruses/20231021_190000.csv'

#     df_a = load_algorithm_a(path_a)
#     df_b = load_algorithm_b(path_b)

#     aligned = align_events(df_a, df_b, tolerance=1.0)
#     plot_matched_events(aligned, 'Call times comparison (Algorithm A vs B)')


def plot_matched_events_per_frog(
    aligned_data: Dict[float, pd.DataFrame], output_folder: str = None
):
    """
    Generates a scatter plot for each frog, comparing call times between algorithm A and B.
    Fits a linear regression line and displays the R^2 score.

    If `output_folder` is specified, saves the plots as PNG files.
    """
    for frog_id, df_match in aligned_data.items():
        # Linear fit
        X = df_match["time_a"].values.reshape(-1, 1)
        y = df_match["time_b"].values
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)

        # Plot
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(
            df_match["time_a"],
            df_match["time_b"],
            alpha=0.7,
            s=10,
            color="tab:blue",
            label="Events",
        )

        # Regression line
        x_vals = np.linspace(
            df_match["time_a"].min(), df_match["time_a"].max(), 100
        ).reshape(-1, 1)
        y_vals = model.predict(x_vals)
        ax.plot(x_vals, y_vals, "r-", label=f"Linear regression\n$R^2$ = {r2:.3f}")

        # # Identity line
        # max_val = df_match[['time_a', 'time_b']].max().max()
        # ax.plot([0, max_val], [0, max_val], 'k-', label='y = x')

        # Axes and title
        ax.set_xlabel("Time Algorithm A (minutes)")
        ax.set_ylabel("Time Algorithm B (minutes)")
        ax.set_title(f"Frog {int(frog_id)}: Detections Linear Fit")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()

        # Save or show
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, f"rana_{int(frog_id)}.png")
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()


# Example of use

# plot_matched_events_per_frog(aligned)


# region comp vs manual tags


def parse_intervals_from_directory(directory: str) -> List[Tuple[float, float]]:
    """
    Searches for files in the given directory with names of the form "(m1, s1)_(m2, s2).csv"
    and returns a list of tuples (start_time_sec, end_time_sec) for each file.

    Args:
        directory: Path to the directory containing the files.
    Returns:
        List of tuples (start_time_sec, end_time_sec).
    """

    pattern = os.path.join(directory, "*.csv")
    filenames = glob.glob(pattern)
    intervals = []
    for fname in filenames:
        base = os.path.basename(fname)
        # Remove extension
        name = os.path.splitext(base)[0]
        # Split into two intervals
        left, right = name.split("_")
        m1, s1 = left.strip("()").split(",")
        m2, s2 = right.strip("()").split(",")
        start_sec = float(m1) * 60.0 + float(s1)
        end_sec = float(m2) * 60.0 + float(s2)
        intervals.append((start_sec, end_sec))
    return intervals


# intervals = parse_intervals_from_directory("../20231021_190000/")


def load_algorithm_detections(
    csv_path: str, mics_ids_gt_60: np.ndarray
) -> pd.DataFrame:
    """
    Load algorithm detections from a CSV, filtering by frog (mic_id) in mics_ids_gt_60.

    Args:
        csv_path: Path to 'matched_co_lin.csv'.
        mics_ids_gt_60: np.ndarray of allowed mic_ids (as frogs).
    Returns:
        DataFrame with columns ['frog', 'time'].
    """
    mapping = {
        10.0: 0.0,
        12.0: 1.0,
        13.0: 2.0,
        14.0: 3.0,
        15.0: 4.0,
        16.0: 5.0,
        17.0: 6.0,
        18.0: 7.0,
        19.0: 8.0,
    }
    df = pd.read_csv(csv_path)
    # Keep only detections for frogs in mics_ids_gt_60
    df = df[df["frog"].isin([mapping[float(m)] for m in mics_ids_gt_60])].copy()
    return df[["frog", "time_lin"]].rename(columns={"time_lin": "time"})


def parse_interval_from_filename(fname: str) -> float:
    """
    Parse the start time in seconds from a file name of form "(m, s)_(m, s).csv".

    Args:
        fname: file name string.
    Returns:
        Start time in seconds.
    """
    # e.g. "(1, 34)_(2, 0).csv"
    base = os.path.basename(fname)
    left = base.split("_")[0]  # "(1,"
    # get minute and second
    m_s = base.split("_")[0].strip("()")
    minute, second = m_s.split(",")
    return float(minute) * 60.0 + float(second)


def load_manual_tags(
    folder: str, mics_ids_gt_60: np.ndarray, hop_length: int = 512, sr: int = 96000
) -> pd.DataFrame:
    """
    Load all hand-tagged CSVs from a folder, convert their times to absolute seconds.

    Each file is named "(m, s)_(m, s).csv" and contains:
      - mic_id: int (10-19)
      - frec: float
      - time: int (frame index)
      - power: float
      - real_chant: 0 or 1

    Args:
        folder: Path to the folder containing the CSVs.
        hop_length: hop_length used when computing mel-spectrogram.
        sr: sample rate used when computing mel-spectrogram.
    Returns:
        DataFrame with columns ['frog', 'time', 'real_chant'].
    """
    mapping = {
        10.0: 0.0,
        12.0: 1.0,
        13.0: 2.0,
        14.0: 3.0,
        15.0: 4.0,
        16.0: 5.0,
        17.0: 6.0,
        18.0: 7.0,
        19.0: 8.0,
    }
    records = []
    csv_files = glob.glob(os.path.join(folder, "*.csv"))
    for path in csv_files:
        start_sec = parse_interval_from_filename(path)
        df = pd.read_csv(path)
        df = df[df["mic_id"].isin(mics_ids_gt_60)].copy()
        # keep only Lin (frec > 60) and real chants
        df = df[df["frec"] > 60].copy()
        # convert frame index to seconds: frame * hop_length / sr
        df["time"] = df["time"] * (hop_length / sr) + start_sec
        df["frog"] = df["mic_id"].map(mapping)
        records.append(df[["frog", "time", "real_chant"]])
    if not records:
        return pd.DataFrame(columns=["frog", "time", "real_chant"])
    return pd.concat(records, ignore_index=True)


def match_detections(
    alg_df: pd.DataFrame, tag_df: pd.DataFrame, tolerance_sec: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compare algorithm detections to manual tags.

    For each manual (real_chant==1), look for an algorithm time within ±tolerance.

    Args:
        alg_df: DataFrame from load_algorithm_detections.
        tag_df: DataFrame from load_manual_tags.
        tolerance_sec: matching window in seconds.
    Returns:
        Tuple of two 1-D arrays:
          - y_true: ground truth binary labels for each manual tag (all ones).
          - y_pred: 1 if algorithm detected within window, else 0.
    """
    # filter only real chants
    real = tag_df[tag_df["real_chant"] == 1].copy()
    y_true = []
    y_pred = []
    # sort for faster search
    alg_df_sorted = alg_df.sort_values(by="time")
    times = alg_df_sorted["time"].values
    frogs = alg_df_sorted["frog"].values
    for _, row in real.iterrows():
        y_true.append(1)
        # find matches same frog
        frog_mask = frogs == row["frog"]
        # numpy vector of time diffs
        diffs = np.abs(times[frog_mask] - row["time"])
        if np.any(diffs <= tolerance_sec):
            y_pred.append(1)
        else:
            y_pred.append(0)
    return np.array(y_true), np.array(y_pred)


def compute_confusion_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute confusion matrix, precision, recall.

    Args:
        y_true: ground truth binary labels.
        y_pred: predicted binary labels.
    Returns:
        Dict with keys: 'true_negative', 'false_positive',
                        'false_negative', 'true_positive',
                        'precision', 'recall'.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    return {
        "false_negative": int(fn),
        "true_positive": int(tp),
        "precision": precision,
        "recall": recall,
    }


# Example usage:
# alg3 = load_algorithm_detections(csv_path="matched_co_lin.csv", mics_ids_gt_60=mic_ids_gt_60)
# tags3 = load_manual_tags(folder="../20231021_190000/", mics_ids_gt_60=mic_ids_gt_60)

# y_true, y_pred = match_detections(alg3, tags3, tolerance_sec=1.0)
# metrics = compute_confusion_metrics(y_true, y_pred)
# print("Detection validation metrics:", metrics)


def get_false_positives(
    alg_df: pd.DataFrame,
    tag_df: pd.DataFrame,
    tolerance_sec: float = 1.0,
    time_intervals: list = None,
) -> pd.DataFrame:
    """
    Returns algorithm-detected calls that do not have a nearby manual tag (false positives),
    optionally restricted to a given list of time intervals.

    Args:
        alg_df: DataFrame with columns ['frog', 'time'] (algorithm detections).
        tag_df: DataFrame with columns ['frog', 'time', 'real_chant'] (manual tags).
        tolerance_sec: tolerance window in seconds.
        time_intervals: list of tuples (start, end) in seconds to filter the analysis.
    Returns:
        DataFrame with the rows from alg_df that do not have a manual match.
    """
    if time_intervals is not None:
        mask = np.zeros(len(alg_df), dtype=bool)
        for start, end in time_intervals:
            mask |= (alg_df["time"] >= start) & (alg_df["time"] <= end)
        alg_df = alg_df[mask].copy()
        mask_tag = np.zeros(len(tag_df), dtype=bool)
        for start, end in time_intervals:
            mask_tag |= (tag_df["time"] >= start) & (tag_df["time"] <= end)
        tag_df = tag_df[mask_tag].copy()

    real_tags = tag_df[tag_df["real_chant"] == 1]
    alg_fp_mask = []
    for idx, row in alg_df.iterrows():
        mask = real_tags["frog"] == row["frog"]
        diffs = np.abs(real_tags[mask]["time"] - row["time"])
        if not np.any(diffs <= tolerance_sec):
            alg_fp_mask.append(True)
        else:
            alg_fp_mask.append(False)
    return alg_df[alg_fp_mask].reset_index(drop=True)


# Example of use

# false_positives = get_false_positives(alg3, tags3, tolerance_sec=1.0, time_intervals=intervals)


def count_chants_by_frequency(directory: str) -> pd.DataFrame:
    """
    Counts the real chants with frec < 60 and frec > 60 in each CSV file in the directory.

    Args:
        directory: Path to the directory containing the CSV files.

    Returns:
        DataFrame with columns: ['file', 'count_frec_lt_60', 'count_frec_gt_60']
    """
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    results = []
    for path in csv_files:
        df = pd.read_csv(path)
        count_lt_60 = df[(df["frec"] < 60) & (df["real_chant"] == 1)].shape[0]
        count_gt_60 = df[(df["frec"] > 60) & (df["real_chant"] == 1)].shape[0]
        results.append(
            {
                "file": os.path.basename(path),
                "count_frec_lt_60": count_lt_60,
                "count_frec_gt_60": count_gt_60,
            }
        )
    return pd.DataFrame(results)


def compute_co_lin_deltas(
    directory: str, hop_length: int = 512, sr: int = 96000
) -> list:
    """
    Computes the |delta_t| (in seconds) between each lin event (frec > 60) and the closest co event (frec < 60)
    in time and with the same mic_id, for all CSV files in the directory.

    Args:
        directory: Path to the directory with the CSV files.
        hop_length: Hop length used to convert frames to seconds.
        sr: Sample rate used to convert frames to seconds.

    Returns:
        List of |delta_t| (in seconds) for all files.
    """

    delta_ts = []
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    for path in csv_files:
        df = pd.read_csv(path)
        df_real = df[df["real_chant"] == 1]
        df_co = df_real[df_real["frec"] < 60]
        df_lin = df_real[df_real["frec"] > 60]
        for _, row_lin in df_lin.iterrows():
            co_same_mic = df_co[df_co["mic_id"] == row_lin["mic_id"]]
            if not co_same_mic.empty:
                delta_frames = (co_same_mic["time"] - row_lin["time"]).abs().min()
                delta_sec = delta_frames * (hop_length / sr)
                delta_ts.append(delta_sec)
    return delta_ts


def boxplot_and_violin(
    delta_ts: list, delta_times_matched_sub: list, delta_ts_choruses_sub: list
) -> tuple:
    """
    Plots boxplot and violinplot for three distributions of delta_t.

    Args:
        delta_ts (list): List of delta_t values from manual tags.
        delta_times_matched_sub (list): List of delta_t values from energies algorithm.
        delta_ts_choruses_sub (list): List of delta_t values from clustering algorithm.

    Returns:
        tuple: Three lists with outliers removed for each input distribution.
    """
    df_box = pd.DataFrame(
        {
            "Manual tags": delta_ts,
            "Energies Algorithm": delta_times_matched_sub,
            "Clustering Algorithm": delta_ts_choruses_sub,
        }
    )

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df_box, orient="v", palette=["blue", "green", "orange"])
    plt.title("Boxplot of |delta_t| between CO and LIN events")
    plt.ylabel("Delta t (s)")

    plt.subplot(1, 2, 2)
    sns.violinplot(data=df_box, orient="v", palette=["blue", "green", "orange"])
    plt.title("Violinplot of |delta_t| between CO and LIN events")
    plt.ylabel("Delta t (s)")

    plt.tight_layout()
    plt.show()

    def remove_outliers(data):
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return [x for x in data if lower <= x <= upper]

    delta_ts_no_outliers = remove_outliers(delta_ts)
    delta_times_matched_sub_no_outliers = remove_outliers(delta_times_matched_sub)
    delta_ts_choruses_sub_no_outliers = remove_outliers(delta_ts_choruses_sub)

    return (
        delta_ts_no_outliers,
        delta_times_matched_sub_no_outliers,
        delta_ts_choruses_sub_no_outliers,
    )
