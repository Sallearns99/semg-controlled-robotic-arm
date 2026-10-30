#!/usr/bin/env python3
"""
train_emg_v4.py  —  Expanded feature set + cross-channel correlation
─────────────────────────────────────────────────────────────────────

CHANGES vs v3 (and why each one helps):

  Problem: Pronation/Adduction/Supination stuck at 46–56%
  These are forearm rotation gestures with similar envelope amplitudes.
  They need features that capture frequency content and spatial
  relationships between channels.

  NEW FEATURES per channel (19 → 26):
    Time Domain (6 → 9):
      + HC    Hjorth Complexity = Mobility(diff)/Mobility(sig)
               captures regularity — rotations differ from flexions here
      + DASDV Difference Absolute Standard Deviation = std(diff(sig))
               sensitive to fast fluctuations in envelope
      + LOG   Log Detector = exp(mean(log(x)))
               emphasises low-amplitude activity; adduction has less activation
    Frequency Domain (new, 4):
      + MNF   Mean Frequency — shifts between wrist and forearm gestures
      + MDF   Median Frequency — more robust than MNF to noise
      + PKF   Peak Frequency — pronation/supination have different dominants
      + SE    Spectral Entropy — low=single dominant, high=spread spectrum

  Cross-channel features (NEW, 6 = C(4,2) pairs):
    Pearson correlation between every pair of channels.
    Different muscles activate for pronation vs supination →
    channel correlations are the most discriminative feature for rotation.

  Total: 4 × 26 + 6 cross-channel = 110 features

  ARCHITECTURE FIXES:
    - Wider first layer (256→512) to handle 110-dim input
    - Higher dropout (0.3→0.4) to close the 80% tr / 66% val gap
    - SWA (Stochastic Weight Averaging) in final 20% of epochs
      for better generalisation without tuning

  ONNX FIX:
    - Move model to CPU before export (fixes CUDA device mismatch error)

Usage:
    python3 train_emg_v4.py \
        --train_dir ~/Downloads/Split_Data3/Train \
        --val_dir   ~/Downloads/Split_Data3/Validation \
        --out_dir   ~/emg_model_output_v4 \
        --stride    500 \
        --epochs    120
"""

import os, sys, re, json, time, argparse, warnings
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, get_window
from scipy.fft import fft, dct

warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
except ImportError:
    sys.exit('[ERROR] pip install torch')

try:
    from sklearn.metrics import confusion_matrix, classification_report
except ImportError:
    sys.exit('[ERROR] pip install scikit-learn')

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kw: x

# ── Constants ─────────────────────────────────────────────────────────────────
FS        = 2000
N_CH      = 4
WINDOW    = 2000
SMOOTH_HZ = 10.0

N_MFCC    = 13
N_MELS    = 26
N_TD      = 9   # time-domain per channel (was 6)
N_FD      = 4   # frequency-domain per channel (new)
N_PER_CH  = N_MFCC + N_TD + N_FD   # 26
N_CROSS   = 6   # C(4,2) cross-channel correlations
TOTAL_FEAT = N_CH * N_PER_CH + N_CROSS  # 110

ALL_GESTURES = sorted([
    'Abduction', 'Adduction', 'Extension', 'Flexion', 'Grip',
    'Pronation', 'Radial', 'Rest', 'Supination', 'Ulnar'
])


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def butter_lp(cutoff, fs, order=2):
    b, a = butter(order, cutoff / (fs / 2.0), btype='low')
    return b, a


def simulate_sensor(sig, b, a):
    """Rectify + LP smooth → Muscle Sensor v3 envelope. sig: [N, 4]"""
    rect = np.abs(sig)
    out  = np.zeros_like(rect)
    for ch in range(rect.shape[1]):
        out[:, ch] = filtfilt(b, a, rect[:, ch])
    return np.clip(out, 0.0, None)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def _mfcc(sig, fs=FS):
    """13 mean MFCCs. Window 25ms / overlap 15ms."""
    win_len  = int(0.025 * fs)
    hop_len  = win_len - int(0.015 * fs)
    sig      = sig.astype(np.float64)
    w        = get_window('hamming', win_len)
    n_frames = max(1, (len(sig) - win_len) // hop_len + 1)
    pspec    = np.zeros((win_len // 2 + 1, n_frames))

    for i in range(n_frames):
        s  = i * hop_len
        fr = sig[s: s + win_len]
        if len(fr) < win_len:
            fr = np.pad(fr, (0, win_len - len(fr)))
        pspec[:, i] = np.abs(fft(fr * w, n=win_len)[:win_len // 2 + 1]) ** 2

    lo  = 2595 * np.log10(1 + 0 / 700)
    hi  = 2595 * np.log10(1 + (fs / 2) / 700)
    mel = np.linspace(lo, hi, N_MELS + 2)
    hz  = 700 * (10 ** (mel / 2595) - 1)
    bn  = np.floor((win_len + 1) * hz / fs).astype(int)

    fb = np.zeros((N_MELS, win_len // 2 + 1))
    for m in range(1, N_MELS + 1):
        for k in range(int(bn[m-1]), int(bn[m])):
            fb[m-1, k] = (k - bn[m-1]) / (bn[m] - bn[m-1] + 1e-8)
        for k in range(int(bn[m]), int(bn[m+1])):
            fb[m-1, k] = (bn[m+1] - k) / (bn[m+1] - bn[m] + 1e-8)

    fbo   = np.dot(fb, pspec)
    fbo   = np.where(fbo == 0, np.finfo(float).eps, fbo)
    fbo   = 20 * np.log10(fbo)
    mfccs = dct(fbo, type=2, axis=0, norm='ortho')[:N_MFCC]
    return np.mean(mfccs, axis=1).astype(np.float32)


def _tdd(sig):
    """
    9 time-domain features for rectified+smoothed EMG.
    Returns [9] float32.
    """
    s   = sig.astype(np.float64)
    eps = 1e-8
    d   = np.diff(s)
    d2  = np.diff(d)

    var_s  = np.var(s)  + eps
    var_d  = np.var(d)  + eps
    var_d2 = np.var(d2) + eps
    mob_s  = np.sqrt(var_d  / var_s)
    mob_d  = np.sqrt(var_d2 / var_d)

    return np.array([
        float(np.mean(s)),                                # 1. MAV
        float(np.sqrt(np.mean(s ** 2))),                  # 2. RMS
        float(var_s),                                     # 3. VAR
        float(np.sum(np.abs(d))),                         # 4. WL
        float(np.sum(np.abs(np.diff(np.sign(d))) > 0)),   # 5. SSC
        float(mob_s),                                     # 6. HM  Hjorth Mobility
        float(mob_d / (mob_s + eps)),                     # 7. HC  Hjorth Complexity (NEW)
        float(np.std(d)),                                 # 8. DASDV (NEW)
        float(np.exp(np.mean(np.log(s + eps)))),          # 9. LOG detector (NEW)
    ], dtype=np.float32)


def _freq(sig, fs=FS):
    """
    4 frequency-domain features.
    Returns [4] float32.
    """
    n     = len(sig)
    win   = get_window('hamming', n)
    ps    = np.abs(fft(sig.astype(np.float64) * win, n=n)[:n // 2]) ** 2
    freqs = np.linspace(0, fs / 2, n // 2)
    total = ps.sum() + 1e-8

    mnf   = float(np.sum(freqs * ps) / total)
    cumps = np.cumsum(ps)
    mdf   = float(freqs[np.searchsorted(cumps, cumps[-1] / 2)])
    pkf   = float(freqs[np.argmax(ps)])
    p_n   = ps / total
    p_n   = p_n[p_n > 0]
    se    = float(-np.sum(p_n * np.log(p_n)))

    return np.array([mnf, mdf, pkf, se], dtype=np.float32)


def _cross(window):
    """6 Pearson correlations between channel pairs. Returns [6] float32."""
    corrs = []
    for i in range(N_CH):
        for j in range(i + 1, N_CH):
            a, b = window[:, i], window[:, j]
            denom = np.std(a) * np.std(b) + 1e-8
            corrs.append(float(np.mean((a - a.mean()) * (b - b.mean())) / denom))
    return np.array(corrs, dtype=np.float32)


def extract_features(window):
    """
    window : [2000 × 4]
    returns: [110] float32

    Layout (must match inference_node.py):
      ch0: [13 MFCC | 9 TD | 4 FD]
      ch1: [13 MFCC | 9 TD | 4 FD]
      ch2: [13 MFCC | 9 TD | 4 FD]
      ch3: [13 MFCC | 9 TD | 4 FD]
      cross: [6]
    """
    feats = []
    for ch in range(N_CH):
        s = window[:, ch]
        feats.append(_mfcc(s))   # [13]
        feats.append(_tdd(s))    # [9]
        feats.append(_freq(s))   # [4]
    feats.append(_cross(window)) # [6]
    return np.concatenate(feats).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DATA LOADING WITH SLIDING WINDOW AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def parse_filename(path):
    m = re.match(r'Sub_(\d+)_Rep_(\d+)_(\w+)\.mat', os.path.basename(path))
    return (int(m.group(1)), int(m.group(2)), m.group(3)) if m else None


def load_dataset(data_dir, smooth_hz=SMOOTH_HZ, stride=500,
                 window=WINDOW, verbose=True):
    b, a  = butter_lp(smooth_hz, FS)
    files = sorted(Path(data_dir).glob('*.mat'))
    if not files:
        sys.exit(f'[ERROR] No .mat files in {data_dir}')

    X_list, y_list = [], []
    errors = 0

    it = tqdm(files, desc='  Features', ncols=70) if verbose else files
    for fpath in it:
        info = parse_filename(str(fpath))
        if info is None or info[2] not in ALL_GESTURES:
            continue
        try:
            mat = sio.loadmat(str(fpath), simplify_cells=True)
            sig = np.array(mat['segmented_emg'], dtype=np.float32)
        except Exception:
            errors += 1
            continue

        if sig.ndim == 1: sig = sig.reshape(-1, 1)
        if sig.shape[0] < sig.shape[1]: sig = sig.T
        if sig.shape[1] < N_CH:
            sig = np.pad(sig, ((0, 0), (0, N_CH - sig.shape[1])))
        sig = sig[:, :N_CH]
        sig = simulate_sensor(sig, b, a)

        n      = sig.shape[0]
        starts = list(range(0, n - window + 1, stride))
        if not starts:
            sig    = np.vstack([sig, np.zeros((window - n, N_CH), np.float32)])
            starts = [0]

        for s in starts:
            X_list.append(extract_features(sig[s: s + window]))
            y_list.append(info[2])

    if errors:
        print(f'\n  [WARN] {errors} files failed')

    X          = np.stack(X_list).astype(np.float32)
    cls_to_idx = {c: i for i, c in enumerate(ALL_GESTURES)}
    y          = np.array([cls_to_idx[l] for l in y_list], np.int64)

    if verbose:
        print(f'\n  Total windows : {len(X)}  |  Features: {X.shape[1]}')
        counts = {}
        for l in y_list: counts[l] = counts.get(l, 0) + 1
        for c in ALL_GESTURES:
            print(f'    {c:12s}: {counts.get(c, 0)}')

    return X, y, ALL_GESTURES


class EMGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MODEL — ResidualMLP (wider to handle 110-dim input)
# ═══════════════════════════════════════════════════════════════════════════════

class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim, bias=False), nn.BatchNorm1d(dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, dim, bias=False), nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()
    def forward(self, x): return self.act(x + self.net(x))


class ResidualMLP(nn.Module):
    """
    Input : [B, 110]
    Output: [B, num_classes]
    """
    def __init__(self, in_dim=TOTAL_FEAT, num_classes=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 512, bias=False),
            nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.4),
            ResBlock(512, dropout=0.4),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
            ResBlock(256, dropout=0.3),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2),
            ResBlock(128, dropout=0.2),
            nn.Linear(128, 64, bias=False),
            nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.2),
        )
        self.classifier = nn.Linear(64, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x): return self.classifier(self.encoder(x))


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def get_lr(epoch, warmup=5, base_lr=3e-4, min_lr=1e-5, total=100):
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    t = (epoch - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * t))


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += len(y)
    return total_loss / total, 100 * correct / total


@torch.no_grad()
def evaluate(model, loader, device, n_classes):
    model.eval()
    preds, truths = [], []
    for x, y in loader:
        preds.extend(model(x.to(device)).argmax(1).cpu().numpy())
        truths.extend(y.numpy())
    preds, truths = np.array(preds), np.array(truths)
    acc    = 100 * (preds == truths).mean()
    cm     = confusion_matrix(truths, preds, labels=list(range(n_classes)))
    recall = np.diag(cm) / (cm.sum(axis=1) + 1e-8)
    return acc, recall, preds, truths


def update_weights(prev, tr_recall, val_recall, momentum=0.8):
    combined = 0.3 * tr_recall + 0.7 * val_recall
    new      = (1 - combined) + 0.2
    w        = momentum * prev + (1 - momentum) * new
    w        = w / (w.mean() + 1e-8)
    return np.clip(w, 0.5, 3.0).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ONNX EXPORT  (fixed CUDA device mismatch)
# ═══════════════════════════════════════════════════════════════════════════════

def export_onnx(model, out_dir, classes, train_mean, train_std):
    # Always export from CPU to avoid "mat2 on cuda, others on cpu" error
    model_cpu = ResidualMLP(TOTAL_FEAT, len(classes))
    model_cpu.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
    model_cpu.eval()

    path  = os.path.join(out_dir, 'emg_model.onnx')
    dummy = torch.zeros(1, TOTAL_FEAT)

    try:
        torch.onnx.export(
            model_cpu, dummy, path,
            dynamo=False, opset_version=14,
            export_params=True, do_constant_folding=True,
            input_names=['features'], output_names=['logits'],
            dynamic_axes={'features': {0: 'batch'}, 'logits': {0: 'batch'}},
        )
        import onnxruntime as ort
        sess = ort.InferenceSession(path)
        out  = sess.run(['logits'],
                        {'features': np.zeros((1, TOTAL_FEAT), np.float32)})[0]
        print(f'  ONNX saved & verified — output shape: {out.shape}')
    except Exception as e:
        print(f'  [WARN] ONNX failed: {e}')
        torch.save(model.state_dict(), os.path.join(out_dir, 'emg_model.pt'))
        return

    with open(os.path.join(out_dir, 'emg_norm_stats.json'), 'w') as f:
        json.dump({'mean': train_mean.tolist(), 'std': train_std.tolist(),
                   'n_features': int(TOTAL_FEAT)}, f, indent=2)
    with open(os.path.join(out_dir, 'emg_class_map.json'), 'w') as f:
        json.dump({'class_names': classes}, f, indent=2)
    print(f'  Norm stats ({TOTAL_FEAT}-dim) + class map saved')


# ═══════════════════════════════════════════════════════════════════════════════
# 7. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--train_dir',  required=True)
    ap.add_argument('--val_dir',    required=True)
    ap.add_argument('--out_dir',    default='~/emg_model_output_v4')
    ap.add_argument('--smooth_hz',  type=float, default=10.0)
    ap.add_argument('--stride',     type=int,   default=500)
    ap.add_argument('--epochs',     type=int,   default=120)
    ap.add_argument('--batch',      type=int,   default=128)
    ap.add_argument('--lr',         type=float, default=3e-4)
    ap.add_argument('--patience',   type=int,   default=20)
    ap.add_argument('--swa_start',  type=int,   default=90,
                    help='Epoch to begin SWA (typically last 25%% of training)')
    ap.add_argument('--workers',    type=int,   default=4)
    args = ap.parse_args()

    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'\n{"="*60}')
    print('  ResidualMLP v4 — Expanded Features + Cross-Channel')
    print(f'{"="*60}')
    print(f'  Device     : {device}')
    n_win = (12000 - WINDOW) // args.stride + 1
    print(f'  Features   : {N_CH}×({N_MFCC}MFCC+{N_TD}TD+{N_FD}FD)+{N_CROSS}cross = {TOTAL_FEAT}')
    print(f'  Stride     : {args.stride}  → ~{n_win} windows/file (~{1600*n_win:,} train)')
    print(f'  Epochs     : {args.epochs}  |  SWA from epoch {args.swa_start}')
    print(f'  Patience   : {args.patience}')

    print(f'\n{"─"*60}')
    print('  STEP 1 — Training data')
    t0 = time.time()
    X_tr, y_tr, classes = load_dataset(args.train_dir, args.smooth_hz, args.stride)
    print(f'  Done in {time.time()-t0:.1f}s')

    print(f'\n  STEP 2 — Validation data  (stride=window, no overlap)')
    t0 = time.time()
    X_val, y_val, _ = load_dataset(args.val_dir, args.smooth_hz, stride=WINDOW)
    print(f'  Done in {time.time()-t0:.1f}s')

    n_classes = len(classes)

    print(f'\n  STEP 3 — Z-score normalisation')
    train_mean = X_tr.mean(0)
    train_std  = X_tr.std(0)
    X_tr  = (X_tr  - train_mean) / (train_std  + 1e-8)
    X_val = (X_val - train_mean) / (train_std  + 1e-8)

    ds_tr  = EMGDataset(X_tr,  y_tr)
    ds_val = EMGDataset(X_val, y_val)
    val_loader = DataLoader(ds_val, batch_size=args.batch,
                            shuffle=False, num_workers=args.workers)

    model     = ResidualMLP(TOTAL_FEAT, n_classes).to(device)
    swa_model = AveragedModel(model)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\n  Model params : {n_params:,}')

    print(f'\n{"─"*60}')
    print('  STEP 4 — Adaptive training')
    print(f'{"─"*60}')

    cls_weights  = np.ones(n_classes, np.float32)
    best_val_acc = 0.0
    best_state   = None
    stagnation   = 0
    swa_active   = False

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    swa_sched = SWALR(optimizer, swa_lr=5e-5, anneal_epochs=10)

    for epoch in range(1, args.epochs + 1):
        if epoch < args.swa_start:
            lr = get_lr(epoch - 1, warmup=5, base_lr=args.lr,
                        min_lr=1e-5, total=args.swa_start)
            for pg in optimizer.param_groups: pg['lr'] = lr
        else:
            if not swa_active:
                swa_active = True
                print(f'\n  → SWA active from epoch {epoch}')
            swa_sched.step()
            lr = optimizer.param_groups[0]['lr']

        if epoch == 1 or epoch % 5 == 0:
            sw      = cls_weights[y_tr]
            sampler = WeightedRandomSampler(
                torch.from_numpy(sw).double(), len(y_tr), replacement=True)
            tr_loader = DataLoader(ds_tr, batch_size=args.batch,
                                   sampler=sampler, num_workers=args.workers,
                                   pin_memory=device.type == 'cuda')

        cw_tensor = torch.tensor(cls_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=cw_tensor, label_smoothing=0.1)
        tr_loss, _ = train_epoch(model, tr_loader, criterion, optimizer, device)

        if swa_active:
            swa_model.update_parameters(model)

        if epoch % 5 == 0 or epoch == args.epochs:
            eval_model = model
            if swa_active:
                update_bn(tr_loader, swa_model, device=device)
                eval_model = swa_model

            val_acc, val_recall, _, _ = evaluate(eval_model, val_loader, device, n_classes)
            plain_tr = DataLoader(ds_tr, batch_size=args.batch,
                                  shuffle=False, num_workers=args.workers)
            tr_acc_full, tr_recall, _, _ = evaluate(eval_model, plain_tr, device, n_classes)

            cls_weights = update_weights(cls_weights, tr_recall, val_recall)

            improved = val_acc > best_val_acc
            if improved:
                best_val_acc = val_acc
                src = eval_model.module if hasattr(eval_model, 'module') else eval_model
                best_state = {k: v.clone().cpu() for k, v in src.state_dict().items()}
                stagnation = 0
                flag = '✓'
            else:
                stagnation += 5
                flag = ' '

            hard_idx = np.argsort(val_recall)[:3]
            hard_str = '  hard: ' + '  '.join(
                f'{classes[i]}({val_recall[i]*100:.0f}%)' for i in hard_idx)

            print(f'  Ep {epoch:4d}  lr={lr:.1e}  loss={tr_loss:.4f}  '
                  f'tr={tr_acc_full:.1f}%  val={val_acc:.1f}%  '
                  f'best={best_val_acc:.1f}% {flag}  {hard_str}')

            if stagnation >= args.patience and not swa_active:
                print(f'\n  Early stopping at epoch {epoch}')
                break

    # ── Final evaluation ──────────────────────────────────────────────────────
    print(f'\n{"═"*60}')
    print('  FINAL EVALUATION')
    print(f'{"═"*60}')

    final_model = ResidualMLP(TOTAL_FEAT, n_classes).to(device)
    final_model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    plain_tr = DataLoader(ds_tr, batch_size=args.batch,
                          shuffle=False, num_workers=args.workers)
    final_tr_acc,  _, _, _                  = evaluate(final_model, plain_tr,   device, n_classes)
    final_val_acc, _, val_preds, val_truths = evaluate(final_model, val_loader, device, n_classes)

    print(f'\n  Train accuracy : {final_tr_acc:.2f}%')
    print(f'  Val   accuracy : {final_val_acc:.2f}%')
    print(f'\n  Validation classification report:')
    print(classification_report(val_truths, val_preds, target_names=classes, digits=3))

    print(f'{"─"*60}')
    print('  Saving artefacts...')
    export_onnx(final_model, out_dir, classes, train_mean, train_std)
    torch.save({'model_state_dict': best_state, 'classes': classes,
                'train_mean': train_mean, 'train_std': train_std,
                'val_acc': final_val_acc, 'args': vars(args),
                'n_features': TOTAL_FEAT},
               os.path.join(out_dir, 'checkpoint.pt'))
    print(f'  Checkpoint saved')

    print(f'\n{"═"*60}')
    print(f'  DONE — Best val accuracy: {final_val_acc:.2f}%')
    print(f'{"═"*60}\n')


if __name__ == '__main__':
    main()
