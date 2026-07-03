#!/usr/bin/env python3
"""
emg_config.py — Single source of truth for the dataset format.

Every preprocessing / training / inference script imports the canonical
constants and CSV loaders from here so the format only lives in one place.
(Import resolves when scripts are run from this directory, which matches
how they are invoked: `python preprocess_v0_windowed.py`.)

Matches semg_data_collection/semg_data_collection.ino:
    * 500 Hz, CheezsEMG-paced sampling
    * 5 s per trial  -> 2500 samples/trial
    * 12 gestures
    * 11-column CSV that stores FOUR waves:
        v3_env, cheez_raw, cheez_filt, cheez_env  (+ wear bit)

Channel design note
-------------------
The four waves are TWO physical sensors: the Muscle Sensor v3 (v3_env),
and the CHEEZ.semg counted three ways — cheez_filt is a bandpass of
cheez_raw, and cheez_env = |cheez_filt| smoothed. raw/filt/env are
deterministic transforms of one another, so feeding all three as model
channels is redundant and over-weights the CHEEZ sensor 3x vs the v3.

Therefore the loader STORES all four waves, but only the subset listed in
MODEL_CHANNELS is fed to the model. Default = v3_env + cheez_env (two
comparable envelopes, both resting near 0), which preserves the report's
2-channel design. Change MODEL_CHANNELS in one place to experiment.
"""

import re
import numpy as np
import pandas as pd

# ── Acquisition format (must match the firmware) ─────────────────
FS                = 500            # Hz (CheezsEMG native rate)
TRIAL_SECONDS     = 5
SAMPLES_PER_TRIAL = FS * TRIAL_SECONDS   # 2500
NUM_CYCLES        = 5

# ── Gesture manifest (must match GESTURE_NAMES in the .ino) ──────
NUM_GESTURES = 12
GESTURE_NAMES = [
    "REST",             # 0
    "EXTENSION",        # 1
    "FLEXION",          # 2
    "ULNAR_DEV",        # 3
    "RADIAL_DEV",       # 4
    "GRIP",             # 5
    "DOWNWARD_ROT_CW",  # 6
    "DOWNWARD_ROT_CCW", # 7
    "FRONT_ROT_CW",     # 8
    "FRONT_ROT_CCW",    # 9
    "EXTEND_ELBOW",     # 10
    "RETRACT_ELBOW",    # 11
]

# ── CSV layout (11 columns, in order) ────────────────────────────
CSV_COLUMNS = [
    "participant_id", "cycle", "gesture_name", "gesture_id",
    "trial", "sample_idx",
    "v3_env", "cheez_raw", "cheez_filt", "cheez_env", "wear",
]

# All four signal waves stored in the CSV (excludes the wear bit).
WAVE_COLS = ["v3_env", "cheez_raw", "cheez_filt", "cheez_env"]

# Which waves actually feed the model. Edit this ONE line to change the
# channel set everywhere (preprocessing, training, inference, viz).
MODEL_CHANNELS = ["v3_env", "cheez_env"]
N_CHANNELS     = len(MODEL_CHANNELS)

# Waves whose rest baseline is NOT near zero and therefore need DC removal
# (AC-coupling) before energy-based steps like STAE. cheez_raw sits on a
# ~2048 mid-rail and swings both ways; the envelopes rest near 0.
AC_COUPLE_COLS = {"cheez_raw"}

# Column dtypes for loading.
_INT_COLS   = ["participant_id", "cycle", "gesture_id", "trial", "sample_idx", "wear"]
_FLOAT_COLS = WAVE_COLS


# ── Loading ──────────────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:
    """Load one collection CSV, dropping header/status lines.

    Returns a DataFrame with the 11 columns typed (ints + float waves).
    """
    df = pd.read_csv(path, names=CSV_COLUMNS, dtype=str,
                     usecols=range(len(CSV_COLUMNS)))
    # Keep only rows whose sample_idx is a plain integer (drops the header
    # repeat, DATA_START/DATA_END markers, status lines, etc.).
    mask = df["sample_idx"].astype(str).str.match(r"^\d+$", na=False)
    df   = df[mask].copy()
    cast = {c: int for c in _INT_COLS}
    cast.update({c: float for c in _FLOAT_COLS})
    return df.astype(cast)


def ac_couple_selected(sig: np.ndarray, channels) -> np.ndarray:
    """Median-subtract only the channels listed in AC_COUPLE_COLS.

    `sig` is (T, len(channels)) with columns ordered like `channels`.
    With the default MODEL_CHANNELS (two envelopes) this is a no-op.
    """
    out = sig.copy()
    for j, ch in enumerate(channels):
        if ch in AC_COUPLE_COLS:
            out[:, j] = out[:, j] - np.median(out[:, j])
    return out


def iter_trials(df: pd.DataFrame, channels=None,
                samples: int = SAMPLES_PER_TRIAL):
    """Yield (meta, signal) for each complete trial.

    meta   : dict(participant_id, cycle, gesture_id, gesture_name)
    signal : (samples, len(channels)) float32 — selected waves only,
             trimmed to `samples`. Raw values (no normalization here).
    """
    if channels is None:
        channels = MODEL_CHANNELS
    for (pid, cycle, gid), grp in df.groupby(["participant_id", "cycle", "gesture_id"]):
        grp = grp.sort_values("sample_idx")
        if len(grp) < samples:
            print(f"  [SKIP] P{pid} C{cycle} G{gid}: {len(grp)}/{samples} samples")
            continue
        sig  = grp[list(channels)].values[:samples].astype(np.float32)
        gid  = int(gid)
        meta = {
            "participant_id": int(pid),
            "cycle":          int(cycle),
            "gesture_id":     gid,
            "gesture_name":   GESTURE_NAMES[gid] if gid < NUM_GESTURES else str(gid),
        }
        yield meta, sig


def label_map() -> dict:
    """{ "0": "REST", ... } for label_map.json."""
    return {str(i): n for i, n in enumerate(GESTURE_NAMES)}


# ── Trial-level normalization (shared by ALL preprocessors) ───────
# These act on a whole (T, C) trial. Window-level schemes (window_median,
# window_zscore) live only in preprocess_v0_windowed.py because they need
# windows; the trial-level ones below are the schemes the deep pipelines
# (v1 STAE, v2 STFT) and full-trial V0 can share, so the SAME --normalize
# value means the same thing across model families.
#   none         : leave the signal as-is (envelopes already rest near 0).
#   trial_zscore : per-channel z-score over the whole trial (removes offset
#                  AND gain -> drops absolute amplitude, the main discriminator
#                  on this data, so expect it to underperform).
#   cycle_rest   : REST-relative per cycle. (sig - rest_mean) / rest_scale using
#                  THAT cycle's REST trial; divides out slow per-cycle gain/offset
#                  drift while KEEPING each gesture's elevation above rest.
REST_GID          = 0
TRIAL_NORMALIZERS = ("none", "trial_zscore", "cycle_rest")


def zscore(sig: np.ndarray) -> np.ndarray:
    """Per-channel z-score of a (T, C) trial (std floored to avoid /0)."""
    mu = sig.mean(axis=0, keepdims=True)
    sd = sig.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-10, 1.0, sd)
    return (sig - mu) / sd


def cycle_rest_references(df: pd.DataFrame, channels=None,
                          samples: int = SAMPLES_PER_TRIAL) -> dict:
    """Map {(participant_id, cycle): (rest_mean, rest_scale)} per channel, taken
    from each cycle's REST trial, for the cycle_rest scheme.

    rest_scale = max(rest_std, 0.2 * cycle_std, 1e-6) per channel so a very quiet
    REST baseline cannot blow the division up into amplified noise. Cycles with
    no usable REST trial are omitted (callers should fall back to no-op there).
    """
    if channels is None:
        channels = MODEL_CHANNELS
    refs = {}
    for (pid, cycle), cdf in df.groupby(["participant_id", "cycle"]):
        rest = cdf[cdf["gesture_id"] == REST_GID].sort_values("sample_idx")
        if len(rest) < samples:
            continue
        r         = rest[list(channels)].values.astype(np.float64)[:samples]
        mean      = r.mean(0)
        rest_std  = r.std(0)
        cycle_std = cdf[list(channels)].values.astype(np.float64).std(0)
        scale = np.maximum.reduce([rest_std, 0.2 * cycle_std,
                                   np.full_like(rest_std, 1e-6)])
        refs[(int(pid), int(cycle))] = (mean.astype(np.float32),
                                        scale.astype(np.float32))
    return refs


# ── Cycle-wise train/val/test split (shared) ─────────────────────
# Splitting by recording CYCLE (not subject) is the only scheme that works with
# few participants: every subject contributes to every split. Matches
# train_classical.py / train_v4.py (train 1-3 / val 4 / test 5).
TRAIN_CYCLES = (1, 2, 3)
VAL_CYCLES   = (4,)
TEST_CYCLES  = (5,)


def cycle_split_indices(meta: np.ndarray,
                        train=TRAIN_CYCLES, val=VAL_CYCLES, test=TEST_CYCLES):
    """Index arrays (train, val, test) into rows of `meta` by its cycle column.

    `meta` is (N, >=2) with column 1 = cycle (1-indexed). Works with any
    participant count >= 1.
    """
    cyc = meta[:, 1]
    return (np.where(np.isin(cyc, train))[0],
            np.where(np.isin(cyc, val))[0],
            np.where(np.isin(cyc, test))[0])


# ── Mains-interference notch filter (shared) ─────────────────────
# Diagnosed 2026-06-07: v3_env is DOMINATED by 50 Hz mains (peak +21 dB vs +18 dB
# in the 0.5-10 Hz EMG band) — the Muscle Sensor v3 hardware envelope does NOT
# suppress mains and the firmware does no filtering, so v3_env is effectively
# raw-contaminated, not a clean envelope. cheez_env is already clean (CheezsEMG
# envelopes on-device). A zero-phase notch at the mains freq + harmonics removes
# the line while leaving the EMG band untouched. Egypt mains = 50 Hz.
MAINS_FILTER_CHOICES = ("off", "50", "60")


def apply_mains_filter(sig: np.ndarray, fs: float = FS,
                       mains_hz: int = 50, q: float = 30.0) -> np.ndarray:
    """Zero-phase notch out `mains_hz` and its harmonics (up to Nyquist) from a
    (T, C) signal. `mains_hz=0`/falsy -> no-op. Applied to the raw per-trial
    signal BEFORE normalization/feature extraction."""
    if not mains_hz:
        return sig
    from scipy.signal import iirnotch, filtfilt
    nyq  = fs / 2.0
    out  = sig.astype(np.float64).copy()
    harmonics = [mains_hz * k for k in range(1, int(nyq // mains_hz) + 1) if mains_hz * k < nyq]
    for f0 in harmonics:
        b, a = iirnotch(f0 / nyq, q)
        for c in range(out.shape[1]):
            out[:, c] = filtfilt(b, a, out[:, c])
    return out.astype(np.float32)


def normalize_trial(sig: np.ndarray, scheme: str, rest_ref=None) -> np.ndarray:
    """Apply a trial-level scheme to a (T, C) signal. `rest_ref` is the
    (mean, scale) tuple for this trial's (pid, cycle) when scheme=='cycle_rest'
    (a no-op if it is None, e.g. a cycle missing its REST trial)."""
    if scheme == "trial_zscore":
        return zscore(sig)
    if scheme == "cycle_rest":
        if rest_ref is None:
            return sig
        mean, scale = rest_ref
        return (sig - mean) / scale
    return sig  # "none"
