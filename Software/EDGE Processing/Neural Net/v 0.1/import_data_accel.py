"""
import_data_accel.py

PATCH PPG + accelerometer neural-network data importer.

Creates aligned PPG + XYZ accelerometer windows with reference HR labels
and motion measurements.

Participant data and participant identifiers are not stored in Git.
"""

import os
import glob
import numpy as np
import pandas as pd

data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Data"))

def find_pairs(data_dir):
    patch_files = sorted(glob.glob(os.path.join(data_dir, "*_PATCH.CSV")))
    pairs = []

    for patch_path in patch_files:

        filename = os.path.basename(patch_path)
        pair_id = filename.replace("_PATCH.CSV", "")
        acti_path = os.path.join(data_dir, f"{pair_id}_hr.csv")

        if os.path.exists(acti_path):
            pairs.append((pair_id, patch_path, acti_path))

    return pairs


def import_patch(path):
    patch = pd.read_csv(path, low_memory=False)
    patch = patch[["Date & Time", "PPGVal", "Xval", "Yval", "Zval"]].copy()

    patch["Date & Time"] = pd.to_datetime(patch["Date & Time"], errors="coerce")
    patch["PPGVal"] = pd.to_numeric(patch["PPGVal"], errors="coerce")
    patch["Xval"] = pd.to_numeric(patch["Xval"], errors="coerce")
    patch["Yval"] = pd.to_numeric(patch["Yval"], errors="coerce")
    patch["Zval"] = pd.to_numeric(patch["Zval"], errors="coerce")

    patch = patch.dropna()

    return patch


def import_actiheart(path):

    with open(path, "r", encoding="utf-8") as file:

        start_time = None

        for line in file:

            if line.startswith("Export Start Date/Time"):
                start_time_string = line.strip().split("\t")[1]
                start_time = pd.to_datetime(start_time_string)
                break

    if start_time is None:
        raise ValueError(f"Could not find Export Start Date/Time in {path}")

    actiheart = pd.read_csv(path, sep="\t", skiprows=5, header=0, engine="python", index_col=False)

    actiheart = actiheart.iloc[:, 0:2].copy()
    actiheart.columns = ["Time", "Heart Rate"]

    actiheart["Heart Rate"] = pd.to_numeric(actiheart["Heart Rate"], errors="coerce")
    actiheart["Time"] = pd.to_timedelta(actiheart["Time"].astype(str), errors="coerce")
    actiheart["Time"] = start_time.normalize() + actiheart["Time"]

    actiheart = actiheart.dropna(subset=["Time", "Heart Rate"])

    return actiheart


def create_windows(patch, actiheart, pair_id):
    sample_rate = 50
    window_seconds = 15
    hop_seconds = 1

    samples_per_window = sample_rate * window_seconds
    hop_samples = sample_rate * hop_seconds

    minimum_valid_hr_value = 40
    minimum_valid_hr_samples = 8

    windows = []
    labels = []
    recording_ids = []
    motion_levels = []

    for start in range(0, len(patch) - samples_per_window + 1, hop_samples):

        end = start + samples_per_window
        window = patch.iloc[start:end]

        window_start = window["Date & Time"].iloc[0]
        window_end = window_start + pd.Timedelta(seconds=window_seconds)

        matching_hr = actiheart[
            (actiheart["Time"] >= window_start) &
            (actiheart["Time"] < window_end)
        ]["Heart Rate"]

        dropout_hr = matching_hr[
            (matching_hr > 0) &
            (matching_hr < minimum_valid_hr_value)
        ]

        if len(dropout_hr) > 0:
            continue

        valid_hr = matching_hr[matching_hr >= minimum_valid_hr_value]

        if len(valid_hr) < minimum_valid_hr_samples:
            continue

        signals = window[["PPGVal", "Xval", "Yval", "Zval"]].to_numpy(dtype=np.float32)
        accel = signals[:, 1:4]
        accel_magnitude = np.sqrt(accel[:, 0] ** 2 + accel[:, 1] ** 2 + accel[:, 2] ** 2)
        motion_level = accel_magnitude.std()

        windows.append(signals)
        labels.append(valid_hr.mean())
        recording_ids.append(pair_id)
        motion_levels.append(motion_level)

    return windows, labels, recording_ids, motion_levels


def normalize_windows(windows):
    windows = np.asarray(windows, dtype=np.float32)

    means = windows.mean(axis=1, keepdims=True)
    stds = windows.std(axis=1, keepdims=True)

    stds[stds < 1e-8] = 1.0

    return (windows - means) / stds


def prepare_dataset(train_ids, validation_ids, test_ids):
    pairs = find_pairs(data_dir)

    all_windows = []
    all_labels = []
    all_recording_ids = []
    all_motion_levels = []

    for pair_id, patch_path, acti_path in pairs:

        patch = import_patch(patch_path)
        actiheart = import_actiheart(acti_path)

        overlap_start = max(patch["Date & Time"].min(), actiheart["Time"].min())
        overlap_end = min(patch["Date & Time"].max(), actiheart["Time"].max())

        patch = patch[
            (patch["Date & Time"] >= overlap_start) &
            (patch["Date & Time"] <= overlap_end)
        ].copy()

        actiheart = actiheart[
            (actiheart["Time"] >= overlap_start) &
            (actiheart["Time"] <= overlap_end)
        ].copy()

        windows, labels, recording_ids, motion_levels = create_windows(patch, actiheart, pair_id)

        all_windows.extend(windows)
        all_labels.extend(labels)
        all_recording_ids.extend(recording_ids)
        all_motion_levels.extend(motion_levels)

    windows = np.asarray(all_windows, dtype=np.float32)
    labels = np.asarray(all_labels, dtype=np.float32)
    recording_ids = np.asarray(all_recording_ids)
    motion_levels = np.asarray(all_motion_levels, dtype=np.float32)

    windows = normalize_windows(windows)

    train_mask = np.isin(recording_ids, train_ids)
    validation_mask = np.isin(recording_ids, validation_ids)
    test_mask = np.isin(recording_ids, test_ids)

    X_train = windows[train_mask]
    y_train = labels[train_mask]

    X_validation = windows[validation_mask]
    y_validation = labels[validation_mask]

    X_test = windows[test_mask]
    y_test = labels[test_mask]

    motion_train = motion_levels[train_mask]
    motion_validation = motion_levels[validation_mask]
    motion_test = motion_levels[test_mask]

    return X_train, y_train, X_validation, y_validation, X_test, y_test, motion_train, motion_validation, motion_test