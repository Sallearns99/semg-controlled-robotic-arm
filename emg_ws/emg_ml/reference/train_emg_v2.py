#!/usr/bin/env python3
"""
train_emg_v2.py — End-to-end raw signal sEMG classifier
"""

import os
import sys
import re
import json
import time
import argparse
import warnings
from pathlib import Path
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt

warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from torch.cuda.amp import GradScaler, autocast
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
FS = 2000
N_CH = 4
WINDOW = 2000
SMOOTH_HZ = 10.0

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
    rect = np.abs(sig)
    out = np.zeros_like(rect)
    for ch in range(rect.shape[1]):
        out[:, ch] = filtfilt(b, a, rect[:, ch])
    return np.clip(out, 0.0, None)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. DATASET
# ═══════════════════════════════════════════════════════════════════════════════


def parse_filename(path):
    m = re.match(r'Sub_(\d+)_Rep_(\d+)_(\w+)\.mat', os.path.basename(path))
    return (int(m.group(1)), int(m.group(2)), m.group(3)) if m else None


def load_signals(data_dir, smooth_hz=SMOOTH_HZ, stride=500, window=WINDOW, verbose=True):
    b, a = butter_lp(smooth_hz, FS)
    files = sorted(Path(os.path.expanduser(data_dir)).glob('*.mat'))
    if not files:
        sys.exit(f'[ERROR] No .mat files in {data_dir}')

    segments, labels_str = [], []
    errors = 0

    it = tqdm(files, desc='  Loading', ncols=70) if verbose else files
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

        if sig.ndim == 1:
            sig = sig.reshape(-1, 1)
        if sig.shape[0] < sig.shape[1]:
            sig = sig.T
        if sig.shape[1] < N_CH:
            sig = np.pad(sig, ((0, 0), (0, N_CH - sig.shape[1])))
        sig = sig[:, :N_CH]
        sig = simulate_sensor(sig, b, a)

        n = sig.shape[0]
        starts = range(0, n - window + 1, stride)
        if not list(starts):
            pad = np.zeros((window - n, N_CH), dtype=np.float32)
            sig = np.vstack([sig, pad])
            starts = [0]

        for s in starts:
            segments.append(sig[s: s + window].astype(np.float32))
            labels_str.append(info[2])

    classes = ALL_GESTURES
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    labels = [cls_to_idx[l] for l in labels_str]

    all_data = np.concatenate(segments, axis=0)
    ch_mean = all_data.mean(axis=0).astype(np.float32)
    ch_std = all_data.std(axis=0).astype(np.float32)

    return segments, labels, classes, ch_mean, ch_std


class EMGRawDataset(Dataset):
    def __init__(self, segments, labels, ch_mean, ch_std):
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.ch_mean = torch.from_numpy(ch_mean)
        self.ch_std = torch.from_numpy(ch_std)
        self.data = []
        for seg in segments:
            x = torch.from_numpy(seg)
            x = (x - self.ch_mean) / (self.ch_std + 1e-8)
            self.data.append(x.T.float())

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.data[idx], self.labels[idx]

# ═══════════════════════════════════════════════════════════════════════════════
# 3. MODEL
# ═══════════════════════════════════════════════════════════════════════════════


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, pool=None, dropout=0.0):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, k, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.pool = nn.MaxPool1d(pool) if pool else None
        self.drop = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        if self.pool:
            x = self.pool(x)
        if self.drop:
            x = self.drop(x)
        return x


class TemporalAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attn = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, x):
        w = torch.softmax(self.attn(x), dim=2)
        return (x * w).sum(dim=2)


class MultiScaleSEMG(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.branch_s = ConvBlock(N_CH, 32, k=7)
        self.branch_m = ConvBlock(N_CH, 32, k=25)
        self.branch_l = ConvBlock(N_CH, 32, k=63)
        self.enc1 = ConvBlock(96, 128, k=7, pool=4, dropout=0.1)
        self.enc2 = ConvBlock(128, 256, k=5, pool=4, dropout=0.2)
        self.enc3 = ConvBlock(256, 256, k=3, pool=5, dropout=0.2)
        self.enc4 = ConvBlock(256, 256, k=3, pool=None, dropout=0.2)
        self.shortcut = nn.Sequential(
            nn.Conv1d(128, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.AvgPool1d(20)
        )
        self.attn_pool = TemporalAttention(256)
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = torch.cat([self.branch_s(x), self.branch_m(x), self.branch_l(x)], dim=1)
        x = self.enc1(x)
        res = self.shortcut(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.attn_pool(self.enc4(x + res))
        return self.head(x)

# ═══════════════════════════════════════════════════════════════════════════════
# 4. UTILS
# ═══════════════════════════════════════════════════════════════════════════════


def get_lr(epoch, warmup=5, base_lr=3e-4, min_lr=1e-5, total=100):
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    t = (epoch - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * t))


def train_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    t_loss = t_corr = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        t_loss += loss.item() * len(y)
        t_corr += (logits.argmax(1) == y).sum().item()
        total += len(y)
    return t_loss / total, 100 * t_corr / total


@torch.no_grad()
def evaluate(model, loader, device, n_classes):
    model.eval()
    preds, truths = [], []
    for x, y in loader:
        with autocast():
            x = x.to(device, non_blocking=True)
            out = model(x)
            preds.extend(out.argmax(1).cpu().numpy())
            truths.extend(y.numpy())
    preds, truths = np.array(preds), np.array(truths)
    return 100 * (preds == truths).mean(), confusion_matrix(truths,
                                                            preds, labels=list(range(n_classes))), preds, truths


def update_weights(prev, tr_cm, val_cm, momentum=0.7):
    def get_rec(cm): return np.diag(cm) / (cm.sum(axis=1) + 1e-8)
    comb = 0.3 * get_rec(tr_cm) + 0.7 * get_rec(val_cm)
    new = (1 - comb) + 0.2
    w = momentum * prev + (1 - momentum) * new
    return np.clip(w / (w.mean() + 1e-8), 0.5, 3.0).astype(np.float32)


def export_onnx(model, out_dir, classes, ch_mean, ch_std):
    model.eval()
    path = os.path.join(out_dir, 'emg_model.onnx')
    try:
        torch.onnx.export(model, torch.zeros(1, N_CH, WINDOW).to(next(model.parameters()).device),
                          path, opset_version=14, input_names=['signal'], output_names=['logits'],
                          dynamic_axes={'signal': {0: 'batch'}, 'logits': {0: 'batch'}})
        print(f'  ONNX saved: {path}')
    except Exception as e:
        print(f'  [WARN] ONNX failed: {e}')

    with open(os.path.join(out_dir, 'emg_norm_stats.json'), 'w') as f:
        json.dump({'ch_mean': ch_mean.tolist(), 'ch_std': ch_std.tolist(),
                  'window': WINDOW, 'n_ch': N_CH}, f, indent=2)

# ═══════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_dir', required=True)
    ap.add_argument('--val_dir', required=True)
    ap.add_argument('--out_dir', default='~/emg_model_output_v2')
    ap.add_argument('--smooth_hz', type=float, default=10.0)
    ap.add_argument('--stride', type=int, default=500)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()

    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Device: {device} | Output: {out_dir}')

    segs_tr, y_tr, classes, ch_mean, ch_std = load_signals(
        args.train_dir, args.smooth_hz, args.stride)
    segs_val, y_val, _, _, _ = load_signals(args.val_dir, args.smooth_hz, WINDOW)

    ds_tr = EMGRawDataset(segs_tr, y_tr, ch_mean, ch_std)
    ds_val = EMGRawDataset(segs_val, y_val, ch_mean, ch_std)

    val_loader = DataLoader(ds_val, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    model = MultiScaleSEMG(len(classes)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler()

    best_acc = 0.0
    stagnation = 0
    cls_weights = np.ones(len(classes), np.float32)

    for epoch in range(1, args.epochs + 1):
        lr = get_lr(epoch - 1, base_lr=args.lr, total=args.epochs)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        if epoch == 1 or epoch % 5 == 0:
            sw = cls_weights[np.array(y_tr)]
            sampler = WeightedRandomSampler(torch.from_numpy(sw).double(), len(y_tr))
            tr_loader = DataLoader(ds_tr, batch_size=args.batch, sampler=sampler,
                                   num_workers=args.workers, pin_memory=True,
                                   persistent_workers=(args.workers > 0))

        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(cls_weights).to(device),
            label_smoothing=0.1)
        loss, acc = train_epoch(model, tr_loader, criterion, optimizer, device, scaler)

        if epoch % 5 == 0 or epoch == args.epochs:
            v_acc, v_cm, v_p, v_t = evaluate(model, val_loader, device, len(classes))
            _, t_cm, _, _ = evaluate(
                model, DataLoader(
                    ds_tr, batch_size=args.batch), device, len(classes))
            cls_weights = update_weights(cls_weights, t_cm, v_cm)

            if v_acc > best_acc:
                best_acc, stagnation = v_acc, 0
                torch.save(model.state_dict(), os.path.join(out_dir, 'best_model.pt'))
                flag = '✓'
            else:
                stagnation += 5
                flag = ''

            print(f'Ep {epoch:3d} | Loss: {loss:.4f} | Tr: {acc:.1f}% | Val: {v_acc:.1f}% {flag}')
            if stagnation >= args.patience:
                break

    model.load_state_dict(torch.load(os.path.join(out_dir, 'best_model.pt')))
    export_onnx(model, out_dir, classes, ch_mean, ch_std)
    print(f'Done! Best Val Acc: {best_acc:.2f}%')


if __name__ == '__main__':
    main()
