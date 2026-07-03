#!/usr/bin/env python3
"""
train_emg_model.py
──────────────────
Python rewrite of the MATLAB sEMG training pipeline.

PREPROCESSING — WHY AND WHAT:
  The existing dataset (BIOPAC MP36, Ag/AgCl electrodes) contains raw
  differential EMG. The deployment sensor (Muscle Sensor v3) outputs
  amplified, full-wave rectified, and RC-smoothed EMG — a signal envelope.

  To bridge this domain gap we simulate the Muscle Sensor v3 output:
    Step 1: Full-wave rectification  →  abs(signal)
    Step 2: Low-pass Butterworth     →  smooth envelope (≈ RC smoothing stage)

  This makes the training data match what the real sensor will produce.

FEATURES — WHY EACH ONE WAS CHOSEN OR REPLACED:
  We keep 13 MFCC + 6 time-domain descriptors per channel (76 total)
  so the existing model architecture needs no changes.

  MFCC (13 per channel):
    Still valid on rectified EMG. The rectified + smoothed signal is an
    amplitude envelope; MFCC captures its spectral shape (i.e., how
    quickly the envelope modulates), which differs across gestures.

  Time-Domain Descriptors (6 per channel) — REPLACED:
    OLD features for raw bipolar EMG:
      - Sparseness:   designed for zero-mean signal → meaningless on rectified
      - Zero Crossing: signal is always ≥ 0 → always 0, useless
    NEW features for rectified+smoothed EMG:
      1. MAV  — Mean Absolute Value = mean(sig); fundamental amplitude
      2. RMS  — sqrt(mean(sig²)); signal power
      3. VAR  — Variance; spread of envelope around its mean
      4. WL   — Waveform Length = Σ|Δsig|; envelope complexity / speed
      5. SSC  — Slope Sign Changes; replaces ZC; counts direction reversals
                in the envelope (frequency content proxy, valid on rectified)
      6. HM   — Hjorth Mobility = sqrt(var(Δsig)/var(sig)); ratio of
                envelope rate-of-change to amplitude (spectral centroid proxy)

Usage:
    python3 train_emg_model.py \
        --train_dir   /home/salma/Downloads/Split_Data3/Train \
        --val_dir     /home/salma/Downloads/Split_Data3/Validation \
        --out_dir     ./emg_model_output \
        --max_stages  20 \
        --epochs      15 \
        --patience    3 \
        --smooth_hz   10

Requirements:
    pip3 install torch scipy numpy scikit-learn onnx onnxruntime matplotlib tqdm
"""

import os
import re
import sys
import json
import time
import argparse
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt
from scipy.fft import fft, dct
from scipy.signal import get_window

warnings.filterwarnings('ignore')

# ── Check dependencies ────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
except ImportError:
    sys.exit('[ERROR] PyTorch not installed. Run: pip3 install torch')

try:
    from sklearn.metrics import confusion_matrix, classification_report
except ImportError:
    sys.exit('[ERROR] scikit-learn not installed. Run: pip3 install scikit-learn')

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kw: x  # no-op fallback

# ── Constants ─────────────────────────────────────────────────────────────────
FS          = 2000      # sampling frequency Hz (confirmed from .mat metadata)
N_CH        = 4         # sEMG channels
N_MFCC      = 13        # MFCC coefficients per channel
N_TDD       = 6         # time-domain descriptors per channel
FEAT_PER_CH = N_MFCC + N_TDD   # 19 per channel
TOTAL_FEAT  = FEAT_PER_CH * N_CH  # 76 total

INFERENCE_WINDOW_SAMPLES = 2000   # must match WINDOW_SAMPLES in inference_node.py

ALL_GESTURES = [
    'Abduction', 'Adduction', 'Extension', 'Flexion',
    'Grip', 'Pronation', 'Radial', 'Rest', 'Supination', 'Ulnar'
]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PREPROCESSING — Simulate Muscle Sensor v3 output
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_muscle_sensor_v3(signal: np.ndarray, fs: int = FS,
                               smooth_hz: float = 10.0) -> np.ndarray:
    """
    Transform raw differential EMG to match Muscle Sensor v3 output.

    The sensor circuit (from datasheet schematic):
      AD8226 instrumentation amp → full-wave rectifier (TL084 + diodes) →
      RC smoother (TL084, R=80.6k) → output

    Parameters
    ----------
    signal    : [samples × channels]  raw differential EMG
    fs        : sampling rate Hz
    smooth_hz : cutoff frequency Hz for the RC-smoothing simulation

    Returns
    -------
    [samples × channels]  rectified + smoothed signal, all values ≥ 0
    """
    # Step 1: Full-wave rectification (replicates rectifier stage)
    rectified = np.abs(signal)

    # Step 2: Low-pass Butterworth (replicates RC smoothing stage)
    # 2nd order, cutoff = smooth_hz, zero-phase with filtfilt
    nyq = fs / 2.0
    b, a = butter(2, smooth_hz / nyq, btype='low')
    smoothed = np.zeros_like(rectified)
    for ch in range(rectified.shape[1]):
        smoothed[:, ch] = filtfilt(b, a, rectified[:, ch])

    # Ensure non-negative (filtfilt can introduce tiny negatives near edges)
    return np.clip(smoothed, 0, None)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE EXTRACTION — adjusted for rectified+smoothed signal
# ═══════════════════════════════════════════════════════════════════════════════

def _mfcc(sig_1ch: np.ndarray, fs: int = FS) -> np.ndarray:
    """
    Compute mean MFCCs from a single-channel signal.
    Matches MATLAB:  mfcc(sig, fs, NumCoeffs=13,
                          WindowLength=50, OverlapLength=30)
    Hop = 50 - 30 = 20 samples.
    Returns: [N_MFCC] float32
    """
    n_mels  = 26
    win_len = int(0.025 * fs)   # 50 samples @ 2000 Hz
    hop_len = win_len - int(0.015 * fs)  # hop = 50 - 30 = 20
    sig     = sig_1ch.astype(np.float64)
    win     = get_window('hamming', win_len)

    n_frames = max(1, (len(sig) - win_len) // hop_len + 1)
    pspec    = np.zeros((win_len // 2 + 1, n_frames))

    for i in range(n_frames):
        s     = i * hop_len
        frame = sig[s: s + win_len]
        if len(frame) < win_len:
            frame = np.pad(frame, (0, win_len - len(frame)))
        pspec[:, i] = np.abs(fft(frame * win, n=win_len)[:win_len // 2 + 1]) ** 2

    # Mel filterbank
    lo_mel  = 2595 * np.log10(1 + 0 / 700)
    hi_mel  = 2595 * np.log10(1 + (fs / 2) / 700)
    mel_pts = np.linspace(lo_mel, hi_mel, n_mels + 2)
    hz_pts  = 700 * (10 ** (mel_pts / 2595) - 1)
    bins    = np.floor((win_len + 1) * hz_pts / fs).astype(int)

    fb = np.zeros((n_mels, win_len // 2 + 1))
    for m in range(1, n_mels + 1):
        for k in range(int(bins[m - 1]), int(bins[m])):
            fb[m - 1, k] = (k - bins[m - 1]) / (bins[m] - bins[m - 1] + 1e-8)
        for k in range(int(bins[m]), int(bins[m + 1])):
            fb[m - 1, k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m] + 1e-8)

    fbo   = np.dot(fb, pspec)
    fbo   = np.where(fbo == 0, np.finfo(float).eps, fbo)
    fbo   = 20 * np.log10(fbo)
    mfccs = dct(fbo, type=2, axis=0, norm='ortho')[:N_MFCC]
    return np.mean(mfccs, axis=1).astype(np.float32)


def _tdd_rectified(sig_1ch: np.ndarray) -> np.ndarray:
    """
    6 time-domain descriptors for rectified+smoothed EMG.
    These replace the raw-EMG TDD features that don't work on rectified signals.

    1. MAV  — Mean Absolute Value: fundamental amplitude feature
    2. RMS  — Root Mean Square: signal power
    3. VAR  — Variance: envelope spread
    4. WL   — Waveform Length: envelope complexity
    5. SSC  — Slope Sign Changes: replaces Zero Crossing (still meaningful
              on rectified signal — counts envelope direction reversals)
    6. HM   — Hjorth Mobility: sqrt(var(Δsig)/var(sig)) — spectral centroid
              proxy; doesn't depend on zero crossings, works on any signal

    Returns: [6] float32
    """
    sig = sig_1ch.astype(np.float64)
    eps = 1e-8

    # 1. MAV — mean of rectified (already rectified so = mean)
    mav = float(np.mean(sig))

    # 2. RMS
    rms = float(np.sqrt(np.mean(sig ** 2)))

    # 3. VAR — variance
    var = float(np.var(sig))

    # 4. WL — waveform length
    wl = float(np.sum(np.abs(np.diff(sig))))

    # 5. SSC — slope sign changes (sign changes in the first difference)
    d = np.diff(sig)
    ssc = float(np.sum(np.abs(np.diff(np.sign(d))) > 0))

    # 6. HM — Hjorth Mobility
    hm = float(np.sqrt(np.var(d) / (np.var(sig) + eps)))

    return np.array([mav, rms, var, wl, ssc, hm], dtype=np.float32)


def extract_features(signal_4ch: np.ndarray, fs: int = FS) -> np.ndarray:
    """
    signal_4ch : [samples × 4] rectified+smoothed float32
    Returns    : [76] float32  =  4 × (13 MFCC + 6 TDD)
    """
    assert signal_4ch.ndim == 2 and signal_4ch.shape[1] == N_CH, \
        f'Expected [N×4], got {signal_4ch.shape}'
    feats = []
    for ch in range(N_CH):
        feats.append(_mfcc(signal_4ch[:, ch], fs))   # [13]
        feats.append(_tdd_rectified(signal_4ch[:, ch]))  # [6]
    return np.concatenate(feats).astype(np.float32)   # [76]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def parse_filename(filepath: str) -> dict | None:
    m = re.match(r'Sub_(\d+)_Rep_(\d+)_(\w+)\.mat', os.path.basename(filepath))
    return (dict(subject=int(m.group(1)), rep=int(m.group(2)),
                 gesture=m.group(3), file=filepath)
            if m else None)


def load_dataset(data_dir: str, smooth_hz: float = 10.0,
                 verbose: bool = True) -> tuple:
    """
    Load all .mat files, apply sensor preprocessing, extract features.

    Returns
    -------
    X       : [N × 76] float32 feature matrix
    y       : [N]      int class indices
    labels  : [N]      string gesture names
    classes : list of class names (sorted)
    """
    files = sorted(Path(data_dir).glob('*.mat'))
    if not files:
        sys.exit(f'[ERROR] No .mat files found in {data_dir}')

    features_list, labels_list = [], []
    errors = 0

    if verbose:
        print(f'\n  Loading {len(files)} files from {data_dir}')
        files_iter = tqdm(files, desc='  Features', ncols=70)
    else:
        files_iter = files

    for fpath in files_iter:
        info = parse_filename(str(fpath))
        if info is None:
            continue
        if info['gesture'] not in ALL_GESTURES:
            continue

        try:
            mat = sio.loadmat(str(fpath), simplify_cells=True)
            sig = np.array(mat['segmented_emg'], dtype=np.float32)  # [12000×4]
        except Exception as e:
            print(f'\n  [ERR] {fpath.name}: {e}')
            errors += 1
            continue

        # Ensure [samples × channels]
        if sig.ndim == 1:
            sig = sig.reshape(-1, 1)
        if sig.shape[0] < sig.shape[1]:
            sig = sig.T

        # Pad/trim to exactly N channels
        if sig.shape[1] < N_CH:
            sig = np.pad(sig, ((0, 0), (0, N_CH - sig.shape[1])))
        elif sig.shape[1] > N_CH:
            sig = sig[:, :N_CH]

        # ── PREPROCESSING ────────────────────────────────────────────────────
        processed = simulate_muscle_sensor_v3(sig, fs=FS, smooth_hz=smooth_hz)

        # ── WINDOW EXTRACTION ────────────────────────────────────────────────
        # Slice one INFERENCE_WINDOW_SAMPLES window from the centre of the
        # signal so training features match exactly what the inference node
        # computes at runtime (2000 samples, not the full 12000).
        n = processed.shape[0]
        half = INFERENCE_WINDOW_SAMPLES // 2
        centre = n // 2
        start  = max(0, centre - half)
        end    = start + INFERENCE_WINDOW_SAMPLES
        if end > n:                      # safety: shift window back if too close to end
            end   = n
            start = max(0, end - INFERENCE_WINDOW_SAMPLES)
        window = processed[start:end, :]

        # ── FEATURE EXTRACTION ───────────────────────────────────────────────
        feats = extract_features(window, fs=FS)

        features_list.append(feats)
        labels_list.append(info['gesture'])

    if errors > 0:
        print(f'\n  [WARN] {errors} files failed to load')

    X      = np.stack(features_list, axis=0)   # [N, 76]
    labels = np.array(labels_list)

    classes = sorted(set(labels))
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([cls_to_idx[l] for l in labels], dtype=np.int64)

    if verbose:
        print(f'  Loaded: {len(X)} samples, {len(classes)} classes')
        for c in classes:
            n = (labels == c).sum()
            print(f'    {c:12s}: {n}')

    return X, y, labels, classes


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PYTORCH DATASET & MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class EMGDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # Reshape to [N, 76, 1] — matches the MATLAB sequenceInput(76) convention
        # Conv1D treats 76 as channels, 1 as sequence length
        self.X = torch.from_numpy(X).float().unsqueeze(-1)  # [N, 76, 1]
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class EMGGestureNet(nn.Module):
    """
    Enhanced CNN-BiLSTM matching the MATLAB buildEnhancedFeatureNetwork().

    Input : [B, 76, 1]  (features as channels, seq_len=1)
    Output: [B, num_classes] logits

    Architecture:
      Branch 1: Conv1D(76→64, k=7) + BN + ReLU + Dropout(0.2)
      Branch 2: Conv1D(76→64, k=3) + BN + ReLU + Dropout(0.2)
      DepthConcat → [B, 128, 1]
      Conv1D(128→128, k=5) + BN + ReLU + Dropout(0.3)
      Conv1D(128→64,  k=3) + BN + ReLU + Dropout(0.3)   ← enhanced layer
      BiLSTM(64→128 hidden, bidirectional) → [B, 256]
      Dropout(0.5)
      FC(256→64) + BN + ReLU + Dropout(0.4)              ← enhanced layer
      FC(64→num_classes)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # Branch 1 — larger receptive field
        self.b1_conv = nn.Conv1d(76, 64, kernel_size=7, padding=3)
        self.b1_bn   = nn.BatchNorm1d(64)
        self.b1_drop = nn.Dropout(0.2)

        # Branch 2 — smaller receptive field
        self.b2_conv = nn.Conv1d(76, 64, kernel_size=3, padding=1)
        self.b2_bn   = nn.BatchNorm1d(64)
        self.b2_drop = nn.Dropout(0.2)

        # Feature refinement after concat [128 ch]
        self.conv2    = nn.Conv1d(128, 128, kernel_size=5, padding=2)
        self.bn2      = nn.BatchNorm1d(128)
        self.drop1    = nn.Dropout(0.3)

        # Additional conv (enhanced)
        self.conv3    = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn3      = nn.BatchNorm1d(64)
        self.drop2    = nn.Dropout(0.3)

        # BiLSTM — input_size=64, hidden=128, bidirectional → output=256
        self.bilstm   = nn.LSTM(input_size=64, hidden_size=128,
                                batch_first=False, bidirectional=True)
        self.drop_lstm = nn.Dropout(0.5)

        # FC head (enhanced)
        self.fc1      = nn.Linear(256, 64)
        self.bn_fc1   = nn.BatchNorm1d(64)
        self.drop_fc1 = nn.Dropout(0.4)
        self.fc2      = nn.Linear(64, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 76, 1]

        # Branches
        x1 = self.b1_drop(torch.relu(self.b1_bn(self.b1_conv(x))))  # [B,64,1]
        x2 = self.b2_drop(torch.relu(self.b2_bn(self.b2_conv(x))))  # [B,64,1]

        x  = torch.cat([x1, x2], dim=1)  # [B, 128, 1]

        # Refinement
        x  = self.drop1(torch.relu(self.bn2(self.conv2(x))))  # [B, 128, 1]
        x  = self.drop2(torch.relu(self.bn3(self.conv3(x))))  # [B, 64, 1]

        # BiLSTM — expects [seq, B, features]
        x  = x.permute(2, 0, 1)          # [1, B, 64]
        x, _ = self.bilstm(x)            # [1, B, 256]
        x  = self.drop_lstm(x[-1])       # [B, 256]

        # FC head
        x  = self.drop_fc1(torch.relu(self.bn_fc1(self.fc1(x))))  # [B, 64]
        x  = self.fc2(x)                 # [B, num_classes]
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TRAINING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def get_lr(stage: int) -> float:
    """Learning rate schedule matching the MATLAB adaptive training."""
    if stage == 1:   return 5e-4
    if stage <= 5:   return 2e-4
    if stage <= 10:  return 1e-4
    return 5e-5


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss, correct, total = 0., 0, 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                logits = model(X_batch)
                loss   = criterion(logits, y_batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(dim=1) == y_batch).sum().item()
        total      += len(y_batch)

    return total_loss / total, correct / total * 100


@torch.no_grad()
def evaluate(model, loader, device, classes):
    model.eval()
    all_preds, all_labels = [], []

    for X_batch, y_batch in loader:
        logits = model(X_batch.to(device))
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(y_batch.numpy())

    preds  = np.array(all_preds)
    labels = np.array(all_labels)
    acc    = (preds == labels).mean() * 100

    cm      = confusion_matrix(labels, preds, labels=list(range(len(classes))))
    recall  = np.diag(cm) / (cm.sum(axis=1) + 1e-8)  # per-class recall
    return acc, recall, preds, labels


def compute_class_weights(train_recall: np.ndarray,
                          val_recall:   np.ndarray,
                          prev_weights: np.ndarray,
                          momentum:     float = 0.7) -> np.ndarray:
    """
    Adaptive class weights matching the MATLAB self-adaptive training.
    Uses validation recall (not training recall) as the difficulty signal.
    Applies exponential smoothing to prevent oscillation.
    """
    # 30% train + 70% val to bias toward generalisation difficulty
    combined_recall = 0.3 * train_recall + 0.7 * val_recall

    # Higher weight for harder classes (lower recall)
    new_weights = (1 - combined_recall) + 0.2   # 0.2 minimum boost

    # Momentum smoothing
    weights = momentum * prev_weights + (1 - momentum) * new_weights

    # Normalise to mean=1, clamp [0.5, 3.0]
    weights = weights / (weights.mean() + 1e-8)
    weights = np.clip(weights, 0.5, 3.0)
    return weights.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ONNX EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

def export_onnx(model, out_dir: str, classes: list,
                train_mean: np.ndarray, train_std: np.ndarray):
    """Export model + normalisation stats for ROS2 inference node."""
    model.eval()
    dummy = torch.zeros(1, 76, 1)

    onnx_path = os.path.join(out_dir, 'emg_model.onnx')
    try:
        torch.onnx.export(
            model, dummy, onnx_path,
            dynamo=False,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['features'],
            output_names=['logits'],
        )
        print(f'  ONNX model saved: {onnx_path}')
    except Exception as e:
        print(f'  [WARN] ONNX export failed: {e}')
        print('  Saving PyTorch state dict instead.')
        torch.save(model.state_dict(),
                   os.path.join(out_dir, 'emg_model.pt'))

    # Normalisation stats
    norm_path = os.path.join(out_dir, 'emg_norm_stats.json')
    with open(norm_path, 'w') as f:
        json.dump({'mean': train_mean.tolist(),
                   'std':  train_std.tolist()}, f, indent=2)
    print(f'  Norm stats saved: {norm_path}')

    # Class map
    cmap_path = os.path.join(out_dir, 'emg_class_map.json')
    with open(cmap_path, 'w') as f:
        json.dump({'class_names': classes}, f, indent=2)
    print(f'  Class map saved: {cmap_path}')


# ═══════════════════════════════════════════════════════════════════════════════
# 7. MAIN TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--train_dir',  required=True)
    ap.add_argument('--val_dir',    required=True)
    ap.add_argument('--out_dir',    default='./emg_model_output')
    ap.add_argument('--smooth_hz',  type=float, default=10.0,
                    help='Low-pass cutoff Hz for Muscle Sensor v3 smoothing simulation')
    ap.add_argument('--max_stages', type=int, default=20)
    ap.add_argument('--epochs',     type=int, default=15,
                    help='Epochs per adaptive stage')
    ap.add_argument('--patience',   type=int, default=3,
                    help='Stages without improvement before early stop')
    ap.add_argument('--batch',      type=int, default=32)
    ap.add_argument('--workers',    type=int, default=4)
    ap.add_argument('--no_amp',     action='store_true',
                    help='Disable automatic mixed precision (if GPU issues)')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = torch.cuda.is_available() and not args.no_amp
    print(f'\n{"="*60}')
    print(f'  Python sEMG Training Pipeline')
    print(f'{"="*60}')
    print(f'  Device       : {device}')
    print(f'  AMP          : {use_amp}')
    print(f'  smooth_hz    : {args.smooth_hz} Hz  (Muscle Sensor v3 simulation)')
    print(f'  Features     : {N_MFCC} MFCC + {N_TDD} TDD × {N_CH} ch = {TOTAL_FEAT}')
    print(f'  Max stages   : {args.max_stages}')
    print(f'  Epochs/stage : {args.epochs}')
    print(f'  Patience     : {args.patience}')

    # ── Load + preprocess data ────────────────────────────────────────────────
    print(f'\n{"─"*60}')
    print('  STEP 1 — Loading & preprocessing training data')
    X_tr, y_tr, lbl_tr, classes = load_dataset(
        args.train_dir, smooth_hz=args.smooth_hz)

    print(f'\n  STEP 2 — Loading & preprocessing validation data')
    X_val, y_val, lbl_val, _ = load_dataset(
        args.val_dir, smooth_hz=args.smooth_hz, verbose=True)

    num_classes = len(classes)
    print(f'\n  Classes ({num_classes}): {classes}')

    # ── Z-score normalisation ─────────────────────────────────────────────────
    print(f'\n  STEP 3 — Z-score normalisation')
    train_mean = X_tr.mean(axis=0)   # [76]
    train_std  = X_tr.std(axis=0)    # [76]
    X_tr  = (X_tr  - train_mean) / (train_std  + 1e-8)
    X_val = (X_val - train_mean) / (train_std  + 1e-8)
    print(f'  Mean range  : [{train_mean.min():.4f}, {train_mean.max():.4f}]')
    print(f'  Std  range  : [{train_std.min():.4f}, {train_std.max():.4f}]')

    # ── Datasets & loaders ────────────────────────────────────────────────────
    ds_tr  = EMGDataset(X_tr,  y_tr)
    ds_val = EMGDataset(X_val, y_val)

    # Validation loader — fixed, no sampling needed
    val_loader = DataLoader(ds_val, batch_size=args.batch,
                            shuffle=False, num_workers=args.workers,
                            pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = EMGGestureNet(num_classes=num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\n  Model params: {n_params:,}')

    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # ── Self-adaptive training ────────────────────────────────────────────────
    print(f'\n{"─"*60}')
    print('  STEP 4 — Self-Adaptive Training')
    print(f'{"─"*60}')

    class_weights = np.ones(num_classes, dtype=np.float32)  # start balanced
    best_val_acc  = 0.0
    best_state    = None
    stagnation    = 0
    history       = []

    for stage in range(1, args.max_stages + 1):
        t0 = time.time()
        print(f'\n{"═"*60}')
        print(f'  STAGE {stage}/{args.max_stages}')
        print(f'{"═"*60}')

        # Print current weights
        print('\n  Class weights:')
        for i, (c, w) in enumerate(zip(classes, class_weights)):
            print(f'    {c:12s}: {w:.3f}')

        # ── Build weighted criterion and sampler for this stage ───────────────
        cw_tensor  = torch.tensor(class_weights).to(device)
        criterion  = nn.CrossEntropyLoss(weight=cw_tensor)

        # Weighted sampler: over-sample hard classes in addition to loss weighting
        sample_weights = class_weights[y_tr]
        sampler = WeightedRandomSampler(
            torch.from_numpy(sample_weights).double(),
            num_samples=len(y_tr), replacement=True)
        train_loader = DataLoader(
            ds_tr, batch_size=args.batch, sampler=sampler,
            num_workers=args.workers, pin_memory=True)

        # ── Optimiser with stage LR ───────────────────────────────────────────
        lr = get_lr(stage)
        optimizer = optim.Adam(model.parameters(), lr=lr,
                               weight_decay=0.0005)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, args.epochs // 2),
            gamma=0.7)

        print(f'\n  LR: {lr:.2e}')

        # ── Train for args.epochs ─────────────────────────────────────────────
        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, scaler)
            scheduler.step()

            if epoch % 5 == 0 or epoch == args.epochs:
                val_acc, val_recall, _, _ = evaluate(
                    model, val_loader, device, classes)
                print(f'  Epoch {epoch:3d}/{args.epochs}  '
                      f'loss={tr_loss:.4f}  '
                      f'tr_acc={tr_acc:.1f}%  '
                      f'val_acc={val_acc:.1f}%')

        # ── Stage evaluation ──────────────────────────────────────────────────
        val_acc,  val_recall,  _, _ = evaluate(model, val_loader,  device, classes)
        train_acc, tr_recall,  _, _ = evaluate(model, train_loader, device, classes)
        # Note: train_loader uses weighted sampler, so tr_recall is approximate
        # Rebuild unweighted train loader for exact metrics
        plain_tr_loader = DataLoader(ds_tr, batch_size=args.batch,
                                     shuffle=False, num_workers=args.workers)
        train_acc, tr_recall, _, _ = evaluate(model, plain_tr_loader, device, classes)

        elapsed = time.time() - t0
        print(f'\n  Stage {stage} results:')
        print(f'    Train acc   : {train_acc:.2f}%')
        print(f'    Val   acc   : {val_acc:.2f}%')
        print(f'    Best so far : {best_val_acc:.2f}%')
        print(f'    Time        : {elapsed:.0f}s')

        # ── Save best model ───────────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            stagnation   = 0
            print(f'  ✓ NEW BEST — Val Acc: {val_acc:.2f}%')
        else:
            stagnation += 1
            print(f'  ✗ No improvement for {stagnation} stage(s)')

        # ── Update class weights ──────────────────────────────────────────────
        class_weights = compute_class_weights(
            tr_recall, val_recall, class_weights, momentum=0.7)

        # Show hardest classes
        sorted_idx = np.argsort(val_recall)
        print('\n  Hardest classes (by val recall):')
        for i in sorted_idx[:3]:
            print(f'    {classes[i]:12s}  recall={val_recall[i]*100:.1f}%  '
                  f'weight={class_weights[i]:.3f}')

        history.append(dict(
            stage=stage, train_acc=train_acc, val_acc=val_acc,
            best_val_acc=best_val_acc))

        # ── Early stopping ────────────────────────────────────────────────────
        if stagnation >= args.patience:
            print(f'\n  ⚠ Early stopping after {stage} stages '
                  f'({args.patience} without improvement)')
            break

    # ── Final evaluation with best model ─────────────────────────────────────
    print(f'\n{"═"*60}')
    print('  FINAL EVALUATION')
    print(f'{"═"*60}')

    model.load_state_dict(best_state)
    plain_tr_loader = DataLoader(ds_tr, batch_size=args.batch,
                                 shuffle=False, num_workers=args.workers)
    final_tr_acc,  _, final_tr_preds,  final_tr_lbl  = evaluate(
        model, plain_tr_loader, device, classes)
    final_val_acc, _, final_val_preds, final_val_lbl = evaluate(
        model, val_loader, device, classes)

    print(f'\n  Best model results:')
    print(f'    Train accuracy : {final_tr_acc:.2f}%')
    print(f'    Val   accuracy : {final_val_acc:.2f}%')

    print('\n  Validation classification report:')
    print(classification_report(
        final_val_lbl, final_val_preds,
        target_names=classes, digits=3))

    # ── Save everything ───────────────────────────────────────────────────────
    print(f'\n{"─"*60}')
    print('  Saving model and artefacts...')

    export_onnx(model, args.out_dir, classes, train_mean, train_std)

    # Also save full checkpoint
    ckpt_path = os.path.join(args.out_dir, 'checkpoint.pt')
    torch.save({
        'model_state_dict': best_state,
        'classes':          classes,
        'train_mean':       train_mean,
        'train_std':        train_std,
        'val_acc':          final_val_acc,
        'train_acc':        final_tr_acc,
        'history':          history,
        'args': vars(args),
    }, ckpt_path)
    print(f'  Checkpoint saved: {ckpt_path}')

    print(f'\n{"═"*60}')
    print(f'  DONE — Best validation accuracy: {final_val_acc:.2f}%')
    print(f'  Output: {args.out_dir}/')
    print(f'{"═"*60}\n')


if __name__ == '__main__':
    main()
