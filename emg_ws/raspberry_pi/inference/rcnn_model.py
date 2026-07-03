#!/usr/bin/env python3
"""
rcnn_model.py — Recurrent-CNN (CRNN) over the RAW EMG window + an
inference-compatible wrapper.

This is a *genuine* recurrent CNN: it consumes the raw sliding window
(win_samples x n_channels), runs a 1-D convolutional stack to learn local
temporal/morphological filters, then a bi-GRU to model how those activations
evolve across the window, and finally a linear head. That is the real CRNN the
76-D summary-feature vector cannot support (those are aggregate stats, not a
sequence).

Why a wrapper. inference_rt.py loads ONE object with joblib and asks it for a
prediction + a class-probability vector. The classical path hands it a 76-D
feature row; this CRNN instead consumes the raw window. To keep them
interchangeable, RCNNInference exposes:

    predict_window(window)        -> int gesture id          (cfg id space)
    predict_proba_window(window)  -> np.ndarray (NUM_GESTURES,)  id-indexed

`window` is the SAME array inference_rt._predict already has in hand:
ac_couple(buffer), shape (win_samples, n_channels). The returned probability
vector is full 12-wide and indexed by gesture id (zeros for classes the model
was not trained on), so index 0 is always REST and GESTURE_NAMES[i] lines up —
the REST gate and the --verbose top-2 in inference_rt.py work unchanged.

The wrapper pickles only the state_dict + small config (not the live nn.Module),
so the joblib artifact is portable across torch versions / CPU-GPU.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# Fixed head width (12), matches every trainer. Prefer emg_config as the single
# source of truth, but fall back to the literal so this module can be unpickled
# in minimal environments (e.g. the ROS node's python, which may lack pandas /
# emg_config) — joblib.load(<crnn>.joblib) imports this module.
try:
    import emg_config as cfg
    NUM_GESTURES = cfg.NUM_GESTURES
except Exception:
    NUM_GESTURES = 12


# ── Architecture ────────────────────────────────────────────────────────────
class RCNN(nn.Module):
    """1-D CNN feature extractor -> bi-GRU -> linear head.

    Input : (B, n_channels, win_samples)   (channels-first for Conv1d)
    Output: (B, n_classes) logits
    """

    def __init__(self, n_channels: int = 2, n_classes: int = NUM_GESTURES,
                 conv_ch=(32, 64, 64), gru_hidden: int = 64, gru_layers: int = 1,
                 dropout: float = 0.3):
        super().__init__()
        layers = []
        c_in = n_channels
        for c_out in conv_ch:
            layers += [
                nn.Conv1d(c_in, c_out, kernel_size=5, padding=2),
                nn.BatchNorm1d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),          # halves the time axis each block
            ]
            c_in = c_out
        self.conv = nn.Sequential(*layers)
        self.gru = nn.GRU(c_in, gru_hidden, num_layers=gru_layers,
                          batch_first=True, bidirectional=True,
                          dropout=dropout if gru_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * gru_hidden, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):                 # x: (B, C, T)
        z = self.conv(x)                  # (B, C', T')
        z = z.transpose(1, 2)             # (B, T', C')  -> sequence for the GRU
        seq, _ = self.gru(z)              # (B, T', 2*H)
        pooled = seq.mean(dim=1)          # temporal average pool
        return self.head(pooled)


# ── Inference-compatible wrapper (the joblib artifact) ──────────────────────
class RCNNInference:
    """Drop-in object for inference_rt.py: predict_window / predict_proba_window.

    Holds the trained weights + per-channel standardisation stats (fit on the
    ac_couple'd TRAIN windows, so train and live inputs are normalised
    identically). `classes_present` are the gesture ids actually trained on; the
    softmax is scattered back into a 12-wide id-indexed vector.
    """

    def __init__(self, state_dict, arch: dict, ch_mean, ch_std,
                 win_samples: int, n_channels: int):
        self.arch = arch
        self.ch_mean = np.asarray(ch_mean, dtype=np.float32)   # (n_channels,)
        self.ch_std = np.asarray(ch_std, dtype=np.float32)     # (n_channels,)
        self.win_samples = int(win_samples)
        self.n_channels = int(n_channels)
        self._state_dict = state_dict
        self._model = None                # rebuilt lazily / after unpickle

    # -- (de)serialisation: store weights, not the live module ----------------
    def __getstate__(self):
        d = self.__dict__.copy()
        d["_model"] = None
        return d

    def _ensure_model(self):
        if self._model is None:
            m = RCNN(**self.arch)
            m.load_state_dict(self._state_dict)
            m.eval()
            self._model = m
        return self._model

    # -- core: a single ac_couple'd window -> id-indexed proba ----------------
    def _prep(self, window: np.ndarray) -> np.ndarray:
        """window: (win_samples, n_channels) -> standardised (1, C, T) float32."""
        w = np.asarray(window, dtype=np.float32)
        if w.shape[0] != self.win_samples or w.shape[1] != self.n_channels:
            raise ValueError(
                f"RCNN expected window {(self.win_samples, self.n_channels)}, "
                f"got {w.shape}")
        w = (w - self.ch_mean) / self.ch_std         # per-channel standardise
        return w.T[None, :, :]                        # (1, C, T)

    @torch.no_grad()
    def predict_proba_window(self, window: np.ndarray) -> np.ndarray:
        m = self._ensure_model()
        x = torch.from_numpy(self._prep(window))
        logits = m(x)[0]
        sm = torch.softmax(logits, dim=0).numpy()     # (n_classes == 12)
        # Head is already 12-wide & id-indexed (trained with original gesture ids).
        full = np.zeros(NUM_GESTURES, dtype=np.float32)
        full[: len(sm)] = sm
        return full

    def predict_window(self, window: np.ndarray) -> int:
        return int(np.argmax(self.predict_proba_window(window)))

    # -- batch helper for offline evaluation ----------------------------------
    @torch.no_grad()
    def predict_proba_windows(self, windows: np.ndarray) -> np.ndarray:
        m = self._ensure_model()
        w = (np.asarray(windows, dtype=np.float32) - self.ch_mean) / self.ch_std
        x = torch.from_numpy(np.transpose(w, (0, 2, 1)))   # (N, C, T)
        out = torch.softmax(m(x), dim=1).numpy()
        full = np.zeros((out.shape[0], NUM_GESTURES), dtype=np.float32)
        full[:, : out.shape[1]] = out
        return full
