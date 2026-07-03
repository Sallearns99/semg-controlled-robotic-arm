#!/usr/bin/env python3
"""
preprocess_v0_windowed.py — Sliding-window MFCC+TDD features for real-time inference

Same per-window feature vector as preprocess.py but extracted from short
overlapping windows instead of full trials, multiplying the training data and
matching the window inference_rt.py sees at deploy time.

Window : 200 ms = 100 samples @ 500 Hz  (configurable via --win-ms)
Hop    : 100 ms =  50 samples @ 500 Hz  (configurable via --hop-ms)

Feature vector: per channel (N_MFCC*4 MFCC stats + 6 TDD); with the default
2 model channels that is (8*4 + 6) * 2 = 76 features. (Windowed V0 uses the
smaller N_MFCC=8 tuned for short windows, unlike preprocess.py's N_MFCC=13.)

Output (data/processed_v0_windowed/):
    features.npy   (W, F)  float32
    labels.npy     (W,)    int32
    metadata.npy   (W, 4)  int32   [participant_id, cycle, gesture_id, trial_global_idx]
    label_map.json
    config.json    window/hop/fs/channels (read by inference_rt.py)

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

import emg_config as cfg

# librosa (and its numba dependency) is imported lazily inside mfcc_features().
# It is only needed to compute MFCCs for the CLASSICAL feature path; the CRNN
# (rcnn_model.py) runs on the raw window and never calls extract_features, so a
# CRNN-only deploy (e.g. on a Raspberry Pi) can skip installing librosa/numba.
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

FS                = cfg.FS                 # 500
SAMPLES_PER_TRIAL = cfg.SAMPLES_PER_TRIAL  # 2500
MODEL_CHANNELS    = cfg.MODEL_CHANNELS
N_CHANNELS        = cfg.N_CHANNELS
GESTURE_NAMES     = cfg.GESTURE_NAMES

# Windowed MFCC params — re-derived for 500 Hz / 100-sample (200 ms) windows.
# n_fft=32 (~64 ms) -> 17 freq bins over 0-250 Hz; hop=16 gives ~5 frames per
# 100-sample window, enough to summarise each coefficient with mean/std/max/min.
N_MFCC           = 8
MFCC_N_FFT       = 32
MFCC_WIN_SAMPLES = 32   # win_length == n_fft
MFCC_HOP_SAMPLES = 16   # 50% overlap


def ac_couple(sig: np.ndarray) -> np.ndarray:
    """Remove per-channel DC offset using median subtraction.

    Applied per-window (not per-trial) so training and inference are identical:
    inference only ever sees the current window buffer, never the full trial.
    Median is robust to baseline spikes and removes cheez_raw's ~2048 mid-rail
    if it is ever selected as a model channel, without normalising amplitude to
    std=1 (which would destroy the amplitude discrimination TDD relies on).
    """
    return sig - np.median(sig, axis=0, keepdims=True)


def zscore(sig: np.ndarray) -> np.ndarray:
    """Per-channel z-score (subtract mean, divide by std). Removes both offset
    AND gain, which is what drifts cycle-to-cycle, at the cost of absolute
    amplitude. `sig` is (T, n_channels)."""
    return (sig - sig.mean(0, keepdims=True)) / (sig.std(0, keepdims=True) + 1e-8)


# Normalisation schemes (see --normalize). Drift = per-feature distributions
# shift between cycles; normalising out per-trial/window gain+offset fights it.
#   window_median : current default — median-subtract each window. DEPLOYABLE
#                   (inference sees only a window), keeps amplitude.
#   window_zscore : z-score each window. DEPLOYABLE + drift-robust (kills gain
#                   drift); loses absolute amplitude within the window.
#   trial_zscore  : z-score the WHOLE trial per channel before windowing. Most
#                   drift-robust, BUT uses whole-trial stats -> NOT directly
#                   deployable in real time (a rolling/calibration baseline is
#                   needed at inference to mirror it). Good for measuring the
#                   drift ceiling and for offline/replay classification. WARNING:
#                   divides by per-trial std -> ERASES amplitude, which on this
#                   data is the main discriminator (so it underperforms here).
#   cycle_rest    : REST-relative per-cycle gain/offset removal. For each
#                   (participant, cycle) it reads that cycle's REST trial and
#                   rescales EVERY trial in the cycle by (sig - rest_mean) /
#                   rest_scale (per channel). This divides out the slow per-cycle
#                   gain+offset DRIFT while KEEPING each gesture's elevation
#                   above rest (the discriminative amplitude), unlike trial_zscore
#                   which normalises that away. DEPLOYABLE via a short REST
#                   calibration recorded at the start of each session (the live
#                   node must compute rest_mean/rest_scale from that and apply
#                   the identical transform). Per-window median-subtract still
#                   follows, exactly like window_median.
NORMALIZERS = ("window_median", "window_zscore", "trial_zscore", "cycle_rest")

# REST gesture id used as the per-cycle reference for cycle_rest.
REST_GID = 0


def cycle_rest_reference(cdf, channels):
    """Per-channel (mean, scale) from this cycle's REST trial, for cycle_rest.

    `cdf` is one cycle's rows (all gestures). Returns (mean, scale) float32
    arrays of shape (len(channels),), or None if the cycle has no usable REST
    trial. `scale` is the REST std floored by 20% of the cycle's overall std
    (and a tiny absolute floor) so a very quiet REST baseline cannot blow the
    division up into amplified noise.
    """
    rest = cdf[cdf["gesture_id"] == REST_GID].sort_values("sample_idx")
    if len(rest) < SAMPLES_PER_TRIAL:
        return None
    r = rest[list(channels)].values.astype(np.float64)[:SAMPLES_PER_TRIAL]
    mean      = r.mean(0)
    rest_std  = r.std(0)
    cycle_std = cdf[list(channels)].values.astype(np.float64).std(0)
    scale = np.maximum.reduce([rest_std, 0.2 * cycle_std,
                               np.full_like(rest_std, 1e-6)])
    return mean.astype(np.float32), scale.astype(np.float32)


def mfcc_features(channel: np.ndarray) -> np.ndarray:
    import librosa  # lazy: only the classical feature path needs it (see header note)
    mfcc = librosa.feature.mfcc(
        y=channel.astype(np.float32), sr=FS,
        n_mfcc=N_MFCC, n_fft=MFCC_N_FFT,
        hop_length=MFCC_HOP_SAMPLES, win_length=MFCC_WIN_SAMPLES,
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
    """window shape: (win_samples, N_CHANNELS) — returns (F,) float32."""
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
            df = cfg.load_csv(path)
            if df.empty:
                print(f"  Warning: {fname} has no valid data rows, skipping")
                continue
            pid = int(df["participant_id"].iloc[0])
            frames.setdefault(pid, []).append(df)
        except Exception as e:
            print(f"  Warning: skipping {fname}: {e}")
    return sorted({pid: pd.concat(dfs, ignore_index=True)
                   for pid, dfs in frames.items()}.items())


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
    parser.add_argument("--normalize", default="window_median", choices=NORMALIZERS,
                        help="Drift-robust normalisation scheme (see NORMALIZERS). "
                             "window_median = original; window_zscore = deployable + "
                             "drift-robust; trial_zscore = strongest but needs an "
                             "inference-side equivalent.")
    parser.add_argument("--mains-filter", default="off", choices=cfg.MAINS_FILTER_CHOICES,
                        help="Notch out mains interference + harmonics before features. "
                             "v3_env is dominated by 50 Hz mains; '50' cleans it "
                             "(Egypt mains). 'off' = original behaviour.")
    parser.add_argument("--mains-causal", action="store_true",
                        help="Use the CAUSAL streaming notch (mains_notch.apply_causal) instead "
                             "of zero-phase filtfilt. Set this for any model you will run live, "
                             "so training matches inference_rt.py's real-time --mains filter exactly.")
    args = parser.parse_args()
    NORM = args.normalize
    MAINS_HZ = 0 if args.mains_filter == "off" else int(args.mains_filter)
    if args.mains_causal and MAINS_HZ:
        from mains_notch import apply_causal as _mains_apply
    else:
        _mains_apply = lambda s, fs, hz: cfg.apply_mains_filter(s, fs, hz)

    WIN  = int(args.win_ms  * FS / 1000)
    HOP  = int(args.hop_ms  * FS / 1000)
    DROP = int(args.drop_ms * FS / 1000)

    os.makedirs(args.output, exist_ok=True)

    print(f"Channels {MODEL_CHANNELS}  fs {FS} Hz  trial {SAMPLES_PER_TRIAL} samples")
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

            rest_ref = None
            if NORM == "cycle_rest":
                rest_ref = cycle_rest_reference(cdf, MODEL_CHANNELS)
                if rest_ref is None:
                    print(f"  [WARN] P{pid} C{cycle}: no usable REST trial; "
                          f"this cycle falls back to window_median")

            for gid in sorted(cdf["gesture_id"].unique()):
                gdf = cdf[cdf["gesture_id"] == gid].sort_values("sample_idx")
                sig = gdf[MODEL_CHANNELS].values.astype(np.float32)

                if len(sig) < SAMPLES_PER_TRIAL:
                    print(f"  P{pid} C{cycle} G{gid}: only {len(sig)} samples, skipping")
                    continue

                sig = sig[:SAMPLES_PER_TRIAL]
                if MAINS_HZ:
                    sig = _mains_apply(sig, FS, MAINS_HZ)  # notch mains before everything
                if NORM == "trial_zscore":
                    sig = zscore(sig)            # whole-trial per-channel z-score
                elif NORM == "cycle_rest" and rest_ref is not None:
                    rest_mean, rest_scale = rest_ref
                    sig = (sig - rest_mean) / rest_scale   # REST-relative per cycle

                # Slide windows, skipping edge transients
                pos = DROP
                while pos + WIN <= SAMPLES_PER_TRIAL - DROP:
                    window = sig[pos : pos + WIN].copy()
                    if NORM == "window_zscore":
                        window = zscore(window)
                    else:                        # window_median, or trial_zscore (already scaled)
                        window = ac_couple(window)
                    all_features.append(extract_features(window))
                    all_labels.append(int(gid))
                    all_meta.append([pid, cycle, int(gid), trial_global])
                    pos += HOP

                trial_global += 1

            print(f"  Cycle {cycle}: {trial_global} trials processed so far")

    features = np.array(all_features, dtype=np.float32)
    labels   = np.array(all_labels,   dtype=np.int32)
    meta     = np.array(all_meta,     dtype=np.int32)

    np.save(os.path.join(args.output, "features.npy"), features)
    np.save(os.path.join(args.output, "labels.npy"),   labels)
    np.save(os.path.join(args.output, "metadata.npy"), meta)

    with open(os.path.join(args.output, "label_map.json"), "w") as f:
        json.dump(cfg.label_map(), f, indent=2)

    config = {"win_ms": args.win_ms, "win_samples": WIN,
              "hop_ms": args.hop_ms, "hop_samples": HOP,
              "drop_ms": args.drop_ms, "fs": FS,
              "normalize": NORM, "mains_filter": MAINS_HZ,
              "mains_causal": bool(args.mains_causal and MAINS_HZ),
              "model_channels": MODEL_CHANNELS, "n_channels": N_CHANNELS,
              "n_mfcc": N_MFCC, "mfcc_n_fft": MFCC_N_FFT,
              "mfcc_win": MFCC_WIN_SAMPLES, "mfcc_hop": MFCC_HOP_SAMPLES,
              "n_features": int(features.shape[1]) if features.size else 0}
    with open(os.path.join(args.output, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'-'*50}")
    print(f"Total windows : {len(features)}")
    print(f"Feature shape : {features.shape}")
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Windows/class : {dict(zip([GESTURE_NAMES[i] for i in unique], counts.tolist()))}")
    print(f"\nSaved to {args.output}/")


if __name__ == "__main__":
    main()
