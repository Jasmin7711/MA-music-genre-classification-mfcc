# src/functions.py

import os
import random
import shutil
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf


# ---------------------------------------------------------------------
# MP3 TO WAV
# ---------------------------------------------------------------------

def convert_mp3_to_wav(inputdir, outputdir):
    """
    Convert all MP3 files in a directory to WAV files.

    Parameters
    ----------
    inputdir : str
        Directory containing MP3 files.
    outputdir : str
        Directory where WAV files are saved.
    """
    os.makedirs(outputdir, exist_ok=True)

    for root, _, files in os.walk(inputdir):
        for file in files:
            if file.lower().endswith(".mp3"):
                mp3_path = os.path.join(root, file)

                filename_without_extension = os.path.splitext(file)[0]
                wav_path = os.path.join(outputdir, f"{filename_without_extension}.wav")

                try:
                    y, sr = librosa.load(mp3_path, sr=None, mono=True)
                    sf.write(wav_path, y, sr)
                    print(f"Converted: {mp3_path} -> {wav_path}")

                except Exception as e:
                    print(f"[ERROR] Could not convert {mp3_path}: {e}")


# ---------------------------------------------------------------------
# DATA SPLITTING
# ---------------------------------------------------------------------

def fixed_split_per_genre_grouped(
    metadata_filtered,
    base_dir,
    n_train=427,
    n_val=53,
    n_test=53,
    separator=".",
    exclude_genres=None,
    use_top_genres=True,
    top_n=11,
    random_state=42
):
    """
    Create fixed train/validation/test splits grouped by genre.

    Parameters
    ----------
    metadata_filtered : pandas.DataFrame
        Metadata with track_id and track_genre_top.
    base_dir : str
        Directory containing WAV files.
    n_train : int
        Number of tracks per genre for training.
    n_val : int
        Number of tracks per genre for validation.
    n_test : int
        Number of tracks per genre for testing.
    separator : str
        Separator used to extract track IDs from filenames.
        Kept as 'separator' to match the original notebooks.
    exclude_genres : list or None
        Genres to exclude.
    use_top_genres : bool
        Whether to select the top_n most frequent genres before exclusion.
    top_n : int
        Number of top genres to keep.
    random_state : int
        Random seed.

    Returns
    -------
    splits : dict
        Dictionary with train, val and test DataFrames.
    split_files : dict
        Dictionary with selected track IDs per split.
    """
    df = metadata_filtered.copy()

    if exclude_genres is None:
        exclude_genres = []

    # Remove rows without genre
    df = df.dropna(subset=["track_genre_top"]).copy()

    # Select top genres if requested
    if use_top_genres:
        top_genres = (
            df["track_genre_top"]
            .value_counts()
            .head(top_n)
            .index
            .tolist()
        )
        df = df[df["track_genre_top"].isin(top_genres)].copy()

    # Exclude selected genres
    if exclude_genres:
        df = df[~df["track_genre_top"].isin(exclude_genres)].copy()

    # Ensure track_id is comparable
    df["track_id"] = df["track_id"].astype(str)

    random.seed(random_state)

    splits = {
        "train": [],
        "val": [],
        "test": []
    }

    split_files = {
        "train": [],
        "val": [],
        "test": []
    }

    required_tracks = n_train + n_val + n_test

    for genre, group in df.groupby("track_genre_top"):
        group = group.copy()
        group = group.sample(frac=1, random_state=random_state).reset_index(drop=True)

        if len(group) < required_tracks:
            print(
                f"[WARN] Genre '{genre}' has only {len(group)} tracks, "
                f"but {required_tracks} are required. Available tracks are used."
            )

        train_part = group.iloc[:n_train]
        val_part = group.iloc[n_train:n_train + n_val]
        test_part = group.iloc[n_train + n_val:n_train + n_val + n_test]

        splits["train"].append(train_part)
        splits["val"].append(val_part)
        splits["test"].append(test_part)

        split_files["train"].extend(train_part["track_id"].tolist())
        split_files["val"].extend(val_part["track_id"].tolist())
        split_files["test"].extend(test_part["track_id"].tolist())

    for split_name in splits:
        if splits[split_name]:
            splits[split_name] = pd.concat(splits[split_name], ignore_index=True)
        else:
            splits[split_name] = pd.DataFrame(columns=df.columns)

    print("Split sizes:")
    for split_name, split_df in splits.items():
        print(f"{split_name}: {len(split_df)} tracks")
        print(split_df["track_genre_top"].value_counts())
        print()

    return splits, split_files


# ---------------------------------------------------------------------
# MFCC SEGMENT EXTRACTION
# ---------------------------------------------------------------------

def extract_random_mfcc_segments(
    file_path,
    length_in_seconds=3,
    n_samples=25,
    sr=22050,
    n_mfcc=20
):
    """
    Extract random MFCC segments from a single audio file.

    Parameters
    ----------
    file_path : str
        Path to WAV file.
    length_in_seconds : int or float
        Length of each random segment.
    n_samples : int
        Number of random segments.
    sr : int
        Sampling rate for loading the audio.
    n_mfcc : int
        Number of MFCC coefficients.

    Returns
    -------
    list
        List of MFCC matrices.
    """
    mfcc_segments = []

    try:
        y, _ = librosa.load(file_path, sr=sr)
    except Exception as e:
        print(f"[ERROR] {file_path} could not be loaded: {e}")
        return mfcc_segments

    total_frames = len(y)
    segment_length_frames = int(length_in_seconds * sr)

    if total_frames < segment_length_frames:
        print(f"[SKIP] {file_path} shorter than {length_in_seconds} seconds")
        return mfcc_segments

    max_start = total_frames - segment_length_frames

    for _ in range(n_samples):
        start_frame = random.randint(0, max_start)
        end_frame = start_frame + segment_length_frames

        y_segment = y[start_frame:end_frame]

        mfcc = librosa.feature.mfcc(
            y=y_segment,
            sr=sr,
            n_mfcc=n_mfcc
        )

        mfcc_segments.append(mfcc)

    return mfcc_segments


def extract_mfcc_random_mean_df(
    file_path,
    track_id,
    length_in_seconds=10,
    n_samples=6,
    sr=44100,
    n_mfcc=20
):
    """
    Extract MFCC mean and standard deviation features for random segments.

    Parameters
    ----------
    file_path : str
        Full path to the WAV file.
    track_id : str or int
        Track ID for reference.
    length_in_seconds : int or float
        Length of each random segment in seconds.
    n_samples : int
        Number of random segments per WAV file.
    sr : int
        Sample rate for loading.
    n_mfcc : int
        Number of MFCC coefficients.

    Returns
    -------
    pandas.DataFrame
        DataFrame with MFCC mean and standard deviation features.
    """
    feature_list = []

    try:
        y, _ = librosa.load(file_path, sr=sr)
    except Exception as e:
        print(f"[ERROR] {file_path} konnte nicht geladen werden: {e}")
        return pd.DataFrame()

    total_frames = len(y)
    segment_length_frames = int(length_in_seconds * sr)

    if total_frames < segment_length_frames:
        print(f"[SKIP] {file_path} kürzer als {length_in_seconds} Sekunden")
        return pd.DataFrame()

    for i in range(n_samples):
        max_start = total_frames - segment_length_frames
        start_frame = random.randint(0, max_start)
        end_frame = start_frame + segment_length_frames

        y_segment = y[start_frame:end_frame]

        mfcc = librosa.feature.mfcc(
            y=y_segment,
            sr=sr,
            n_mfcc=n_mfcc
        )

        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        feature_dict = {
            "track_id": track_id,
            "sample": i
        }

        feature_dict.update(
            {f"mfcc_mean_{j}": value for j, value in enumerate(mfcc_mean)}
        )

        feature_dict.update(
            {f"mfcc_std_{j}": value for j, value in enumerate(mfcc_std)}
        )

        feature_list.append(feature_dict)

    return pd.DataFrame(feature_list)


# ---------------------------------------------------------------------
# MFCC EXTRACTION FOR FULL DIRECTORIES
# ---------------------------------------------------------------------

def extract_and_save_random_mfcc(
    inputdir,
    length_in_seconds=3,
    n_samples=25,
    sr=22050,
    n_mfcc=20
):
    """
    Extract random MFCC matrices from all WAV files in a directory.

    Returns a DataFrame with one row per extracted segment.
    """
    all_segments = []

    for filename in os.listdir(inputdir):
        if filename.lower().endswith(".wav"):
            file_path = os.path.join(inputdir, filename)
            track_id = os.path.splitext(filename)[0]

            mfcc_segments = extract_random_mfcc_segments(
                file_path=file_path,
                length_in_seconds=length_in_seconds,
                n_samples=n_samples,
                sr=sr,
                n_mfcc=n_mfcc
            )

            for i, mfcc in enumerate(mfcc_segments):
                all_segments.append({
                    "track_id": track_id,
                    "file_name": filename,
                    "segment": i,
                    "X": mfcc
                })

    return pd.DataFrame(all_segments)


def extract_and_save_segments_mfcc(
    inputdir,
    n_segments=10,
    sr=22050,
    n_mfcc=20
):
    """
    Split each WAV file into a fixed number of consecutive segments
    and extract MFCCs for each segment.

    Parameters
    ----------
    inputdir : str
        Directory containing WAV files.
    n_segments : int
        Number of segments per audio file.
    sr : int
        Sampling rate.
    n_mfcc : int
        Number of MFCC coefficients.

    Returns
    -------
    pandas.DataFrame
        DataFrame with MFCC matrices.
    """
    all_segments = []

    for filename in os.listdir(inputdir):
        if filename.lower().endswith(".wav"):
            file_path = os.path.join(inputdir, filename)
            track_id = os.path.splitext(filename)[0]

            try:
                y, _ = librosa.load(file_path, sr=sr)
            except Exception as e:
                print(f"[ERROR] {file_path} could not be loaded: {e}")
                continue

            segment_length = len(y) // n_segments

            if segment_length <= 0:
                print(f"[SKIP] {file_path} too short for {n_segments} segments")
                continue

            for i in range(n_segments):
                start = i * segment_length

                if i == n_segments - 1:
                    end = len(y)
                else:
                    end = (i + 1) * segment_length

                y_segment = y[start:end]

                mfcc = librosa.feature.mfcc(
                    y=y_segment,
                    sr=sr,
                    n_mfcc=n_mfcc
                )

                all_segments.append({
                    "track_id": track_id,
                    "file_name": filename,
                    "segment": i,
                    "X": mfcc
                })

    return pd.DataFrame(all_segments)


# ---------------------------------------------------------------------
# AUDIO AUGMENTATION
# ---------------------------------------------------------------------

def _add_noise(y, noise_factor=0.005):
    noise = np.random.randn(len(y))
    return y + noise_factor * noise


def _pitch_shift(y, sr):
    n_steps = random.uniform(-2, 2)
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)


def _time_stretch(y):
    rate = random.uniform(0.9, 1.1)
    return librosa.effects.time_stretch(y=y, rate=rate)


def _change_gain(y):
    gain_db = random.uniform(-6, 6)
    gain_factor = 10 ** (gain_db / 20)
    return y * gain_factor


def _time_shift(y):
    shift = random.randint(-int(0.2 * len(y)), int(0.2 * len(y)))
    return np.roll(y, shift)


def _fix_length(y_aug, target_length):
    """
    Ensure that augmented audio has the same length as the original audio.
    """
    if len(y_aug) > target_length:
        return y_aug[:target_length]

    if len(y_aug) < target_length:
        return np.pad(y_aug, (0, target_length - len(y_aug)), mode="constant")

    return y_aug


def augment_audio_files(
    inputdir,
    metadata_df,
    outdir,
    n_augmentations=5
):
    """
    Create augmented WAV files for all valid WAV files in a directory.

    Parameters
    ----------
    inputdir : str
        Directory containing original WAV files.
    metadata_df : pandas.DataFrame
        Metadata with track_id and track_genre_top.
    outdir : str
        Directory where augmented WAV files are saved.
    n_augmentations : int
        Number of augmented versions per original file.
    """
    os.makedirs(outdir, exist_ok=True)

    metadata_df = metadata_df.copy()
    metadata_df["track_id"] = metadata_df["track_id"].astype(str)

    valid_track_ids = set(metadata_df["track_id"].tolist())

    augmentation_functions = [
        ("noise", _add_noise),
        ("pitch", _pitch_shift),
        ("stretch", _time_stretch),
        ("gain", _change_gain),
        ("shift", _time_shift)
    ]

    for filename in os.listdir(inputdir):
        if not filename.lower().endswith(".wav"):
            continue

        file_path = os.path.join(inputdir, filename)

        track_id_raw = os.path.splitext(filename)[0]

        try:
            track_id = str(int(track_id_raw.split("_")[0]))
        except ValueError:
            print(f"[SKIP] Track_ID could not be extracted: {filename}")
            continue

        if track_id not in valid_track_ids:
            print(f"[SKIP] {filename}: keine gültigen Metadaten oder Genre fehlt.")
            continue

        try:
            y, sr = librosa.load(file_path, sr=None, mono=True)
        except Exception as e:
            print(f"[ERROR] {filename} could not be loaded: {e}")
            continue

        # Copy original file to augmented directory as well
        original_out_path = os.path.join(outdir, filename)

        try:
            sf.write(original_out_path, y, sr)
        except Exception as e:
            print(f"[ERROR] Could not write original file {filename}: {e}")

        for i in range(n_augmentations):
            aug_name, aug_func = random.choice(augmentation_functions)

            try:
                if aug_name == "pitch":
                    y_aug = aug_func(y, sr)
                else:
                    y_aug = aug_func(y)

                y_aug = _fix_length(y_aug, len(y))

                output_filename = f"{track_id_raw}_aug{i + 1}_{aug_name}.wav"
                output_path = os.path.join(outdir, output_filename)

                sf.write(output_path, y_aug, sr)
                print(f"[SAVED] {output_filename}")

            except Exception as e:
                print(f"[ERROR] Augmentation failed for {filename}: {e}")