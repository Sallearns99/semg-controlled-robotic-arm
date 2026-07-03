#!/usr/bin/env python3
"""
mains_notch.py — Causal, streaming mains-interference notch for LIVE inference.

Training removes mains with cfg.apply_mains_filter() = scipy.iirnotch + filtfilt
(zero-phase, whole-trial). That is non-causal, so it cannot run live. This class
applies the SAME notch frequencies (mains_hz + all harmonics below Nyquist) with
the SAME iirnotch coefficients, but single-pass (lfilter) with persistent state
per channel, so it works sample-by-sample on the real-time stream.

Use it BEFORE the window buffer (training order is: mains -> window -> ac_couple),
on every model channel — matching apply_mains_filter, which filters all channels.

It is the live twin of cfg.apply_mains_filter; the two were verified to produce
near-identical 76-D features on real data (see tools/check_mains_consistency.py).
"""
import numpy as np
from scipy.signal import iirnotch, lfilter, lfilter_zi


class MainsNotch:
    def __init__(self, n_channels: int, fs: float = 500.0,
                 mains_hz: int = 50, q: float = 30.0):
        self.n = int(n_channels)
        self.enabled = bool(mains_hz)
        nyq = fs / 2.0
        # Same harmonic set as cfg.apply_mains_filter: 50,100,150,200 for 50/500.
        self.freqs = [mains_hz * k for k in range(1, int(nyq // mains_hz) + 1)
                      if mains_hz and mains_hz * k < nyq]
        self._ba = [iirnotch(f0 / nyq, q) for f0 in self.freqs]   # [(b,a), ...]
        self._zi0 = [lfilter_zi(b, a) for (b, a) in self._ba]      # unit steady-state
        # Per-channel, per-section filter state; seeded on the first sample seen.
        self._zi = [[None] * len(self._ba) for _ in range(self.n)]
        self._seeded = [False] * self.n

    def reset(self):
        self._zi = [[None] * len(self._ba) for _ in range(self.n)]
        self._seeded = [False] * self.n

    def process(self, sample) -> np.ndarray:
        """Filter ONE multi-channel sample. `sample`: length-n_channels array.
        Returns the filtered sample (float64). No-op if mains_hz was falsy."""
        x = np.asarray(sample, dtype=np.float64)
        if not self.enabled or not self._ba:
            return x.copy()
        out = x.copy()
        for c in range(self.n):
            v = out[c]
            for s, (b, a) in enumerate(self._ba):
                if not self._seeded[c]:
                    # Seed each cascaded section's state to the channel's first
                    # value so we don't ring at startup (steady-state * v).
                    self._zi[c][s] = self._zi0[s] * v
                y, self._zi[c][s] = lfilter(b, a, [v], zi=self._zi[c][s])
                v = y[0]
            out[c] = v
            self._seeded[c] = True
        return out


def apply_causal(sig: np.ndarray, fs: float = 500.0,
                 mains_hz: int = 50, q: float = 30.0) -> np.ndarray:
    """Run a fresh causal MainsNotch over a whole (T, C) trial — the TRAINING-side
    twin of the live streaming filter, so models train on exactly the operator
    inference applies. Use this instead of cfg.apply_mains_filter (which is
    zero-phase filtfilt = a deeper |H|^2 notch that a real-time stream cannot
    reproduce). `mains_hz` falsy -> no-op."""
    sig = np.asarray(sig, dtype=np.float32)
    if not mains_hz:
        return sig
    nf = MainsNotch(sig.shape[1], fs=fs, mains_hz=mains_hz, q=q)
    return np.array([nf.process(s) for s in sig], dtype=np.float32)
