#!/usr/bin/env python3
"""
preprocess_v0_windowed.py — Sliding-window MFCC+TDD features for real-time inference

Same feature vector as preprocess.py (116 features) but extracted from short overlapping
windows instead of full 6-second trials.  This multiplies the training data by ~58×
and matches the window size used by inference_rt.py at deploy time.

Window : 200 ms = 400 samples  (configurable via --win-ms)
Hop    : 100 ms = 200 samples  (configurable via --hop-ms)
→ ~58 windows per 6-second trial
→ 100 trials (2 participants) → ~5 800 windows total

Cycle-wise split still applies cleanly: all windows from a trial stay in the same split.

Output (data/processed_v0_windowed/):
    features.npy   (W, 116)  float32
    labels.npy     (W,)      int32
    metadata.npy   (W, 4)    int32   [participant_id, cycle, gesture_id, trial_global_idx]
    label_map.json
    config.json    window/hop/fs parameters (read by inference_rt.py)

Usage:
    python preprocess_v0_windowed.py
    python preprocess_v0_windowed.py --win-ms 300 --hop-ms 100
    python preprocess_v0_windowed.py --input data/raw --output data/processed_v0_windowed
"""

import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import librosa

warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

FS               = 2000
SAMPLES_PER_TRIAL = FS * 6
N_CHANNELS        = 2
# Windowed MFCC params — tuned for short (200 ms) windows at 2000 Hz.
# n_fft=64 (power-of-2, ~32 ms) gives 33 freq bins covering 0-1000 Hz.
N_MFCC           = 8
MFCC_N_FFT       = 64
MFCC_WIN_SAMPLES = 64   # win_length == n_fft
MFCC_HOP_SAMPLES = 32   # 50% overlap
# FIXED 2026-06-03: n_mels was never set, so librosa used its default 128 over only
# ~33 FFT bins -> 66 empty filters (degenerate features). 20 gives clean coverage.
# MUST stay identical to emg_inference_node.py (N_MELS).
N_MELS           = 20

# 2026-06-03: 8-gesture scheme (matches emg_collection.ino). 6,7 = extras.
# To EXCLUDE the extras from ML training, filter labels to 0..5 at train time
# (or run with only gestures 0-5 present). They are collected for evaluation only.
GESTURE_NAMES = [
    "REST", "EXTENSION", "FLEXION", "ULNAR_DEV", "RADIAL_DEV",
    "GRIP", "SUPINATION", "PRONATION",
]
TRAIN_GESTURE_IDS = [0, 1, 2, 3, 4, 5]   # SUPINATION(6)/PRONATION(7) excluded from training

CSV_COLUMNS = [
    "participant_id", "cycle", "gesture_name", "gesture_id",
    "trial", "sample_idx", "ch1", "ch2",
]


def ac_couple(sig: np.ndarray) -> np.ndarray:
    """Remove per-channel DC offset using median subtraction.

    Applied per-window (not per-trial) so training and inference are identical:
    inference only ever sees the current 400-sample buffer, never the full trial.
    Median is more robust than mean to baseline spikes, and handles ch2's
    hardware-fixed high DC baseline without normalising amplitude to std=1
    (which would destroy the amplitude discrimination that TDD relies on).
    """
    return sig - np.median(sig, axis=0, keepdims=True)


def mfcc_features(channel: np.ndarray) -> np.ndarray:
    mfcc = librosa.feature.mfcc(
        y=channel.astype(np.float32), sr=FS,
        n_mfcc=N_MFCC, n_fft=MFCC_N_FFT,
        hop_length=MFCC_HOP_SAMPLES, win_length=MFCC_WIN_SAMPLES,
        n_mels=N_MELS,
    )
    return np.concatenate([mfcc.mean(1), mfcc.std(1), mfcc.max(1), mfcc.min(1)])


def tdd_features(channel: np.ndarray) -> np.ndarray:
    x   = channel.astype(np.float64)
    rms = np.sqrt(np.mean(x ** 2))
    rsq = rms ** 2
    sp  = np.sum(np.abs(x)) / (np.sqrt(len(x)) * np.sqrt(np.sum(x ** 2)) + 1e-10)
    diff = np.diff(x)
    irr  = (np.sqrt(np.mean(diff ** 2)) / (rms + 1e-10))
    wl   = np.sum(np.abs(diff))
    thr  = 0.05 * rms
    zc   = float(np.sum((np.abs(diff) > thr) & (x[:-1] * x[1:] < 0)))
    return np.array([rsq, sp, irr, rms, wl, zc], dtype=np.float64)


def extract_features(window: np.ndarray) -> np.ndarray:
    """window shape: (win_samples, 2) — returns (116,) float32."""
    parts = []
    for ch in range(N_CHANNELS):
        parts.append(mfcc_features(window[:, ch]))
        parts.append(tdd_features(window[:, ch]))
    return np.concatenate(parts).astype(np.float32)


def load_raw_csvs(raw_dir: str):
    """Return list of (participant_id, DataFrame) sorted by participant."""
    frames = {}
    for fname in sorted(os.listdir(raw_dir)):
        if not fname.endswith(".csv"):
            continue
        path = os.path.join(raw_dir, fname)
        try:
            df = pd.read_csv(path, names=CSV_COLUMNS, dtype=str,
                             usecols=range(len(CSV_COLUMNS)))
            # Drop non-data rows (header repeats, status lines, etc.)
            mask = df["sample_idx"].str.match(r"^\d+$", na=False)
            df   = df[mask].copy()
            df   = df.astype({
                "participant_id": int, "cycle": int, "gesture_id": int,
                "trial": int, "sample_idx": int, "ch1": int, "ch2": int,
            })
            if df.empty:
                print(f"  Warning: {fname} has no valid data rows, skipping")
                continue
            pid = int(df["participant_id"].iloc[0])
            frames.setdefault(pid, []).append(df)
        except Exception as e:
            print(f"  Warning: skipping {fname}: {e}")
    merged = {}
    for pid, dfs in frames.items():
        merged[pid] = pd.concat(dfs, ignore_index=True)
    return sorted(merged.items())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   default="data/raw")
    parser.add_argument("--output",  default="data/processed_v0_windowed")
    parser.add_argument("--win-ms",  type=int, default=200,
                        help="Window length in ms (default 200)")
    parser.add_argument("--hop-ms",  type=int, default=100,
                        help="Hop size in ms (default 100)")
    parser.add_argument("--drop-ms", type=int, default=50,
                        help="Ignore this many ms at trial start/end (transients)")
    args = parser.parse_args()

    WIN  = int(args.win_ms  * FS / 1000)
    HOP  = int(args.hop_ms  * FS / 1000)
    DROP = int(args.drop_ms * FS / 1000)

    os.makedirs(args.output, exist_ok=True)

    print(f"Window {args.win_ms} ms ({WIN} samples)  "
          f"Hop {args.hop_ms} ms ({HOP} samples)  "
          f"Drop {args.drop_ms} ms at edges")

    participants = load_raw_csvs(args.input)
    if not participants:
        raise FileNotFoundError(f"No CSV files found in {args.input}")

    all_features, all_labels, all_meta = [], [], []
    trial_global = 0

    for pid, df in participants:
        print(f"\nParticipant {pid}")
        for cycle in sorted(df["cycle"].unique()):
            cdf = df[df["cycle"] == cycle]
            for gid in sorted(cdf["gesture_id"].unique()):
                gdf = cdf[cdf["gesture_id"] == gid].sort_values("sample_idx")
                sig = gdf[["ch1", "ch2"]].values.astype(np.float32)

                if len(sig) < SAMPLES_PER_TRIAL:
                    print(f"  P{pid} C{cycle} G{gid}: only {len(sig)} samples, skipping")
                    continue

                sig  = sig[:SAMPLES_PER_TRIAL]

                # Slide windows, skipping edge transients
                win_count = 0
                pos = DROP
                while pos + WIN <= SAMPLES_PER_TRIAL - DROP:
                    window = ac_couple(sig[pos : pos + WIN].copy())
                    feat   = extract_features(window)
                    all_features.append(feat)
                    all_labels.append(gid)
                    all_meta.append([pid, cycle, gid, trial_global])
                    pos += HOP
                    win_count += 1

                trial_global += 1

            print(f"  Cycle {cycle}: {trial_global} trials processed so far")

    features = np.array(all_features, dtype=np.float32)
    labels   = np.array(all_labels,   dtype=np.int32)
    meta     = np.array(all_meta,     dtype=np.int32)

    np.save(os.path.join(args.output, "features.npy"), features)
    np.save(os.path.join(args.output, "labels.npy"),   labels)
    np.save(os.path.join(args.output, "metadata.npy"), meta)

    with open(os.path.join(args.output, "label_map.json"), "w") as f:
        json.dump({str(i): n for i, n in enumerate(GESTURE_NAMES)}, f, indent=2)

    config = {"win_ms": args.win_ms, "win_samples": WIN,
              "hop_ms": args.hop_ms, "hop_samples": HOP,
              "drop_ms": args.drop_ms, "fs": FS,
              "n_mfcc": N_MFCC, "mfcc_n_fft": MFCC_N_FFT,
              "mfcc_win": MFCC_WIN_SAMPLES, "mfcc_hop": MFCC_HOP_SAMPLES,
              "n_features": features.shape[1]}
    with open(os.path.join(args.output, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'─'*50}")
    print(f"Total windows : {len(features)}")
    print(f"Feature shape : {features.shape}")
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Windows/class : {dict(zip([GESTURE_NAMES[i] for i in unique], counts.tolist()))}")
    print(f"\nSaved to {args.output}/")


if __name__ == "__main__":
    main()
