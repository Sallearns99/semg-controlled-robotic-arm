#!/usr/bin/env python3
"""
inference_rt.py — Real-time EMG gesture inference from ESP32

Reads a continuous 2-channel EMG stream from the ESP32 over Serial (USB now,
WiFi later), maintains a sliding window buffer, extracts MFCC+TDD features
every hop, and predicts the current gesture using the saved SVM model.

ESP32 must run the streaming firmware (emg_stream.ino), NOT the collection
firmware (emg_collection.ino).  The streaming firmware continuously sends:
    ch1,ch2\\n
at 2000 Hz with no buffering.

Prediction pipeline:
  serial sample → circular buffer (200 ms)
      → every hop (100 ms): extract features → SVM predict
      → majority vote over last 5 predictions → stable gesture label
      → REST gating: if all recent predictions agree on REST → REST
      → print + publish to stdout / ROS2 topic (optional)

Usage:
    python inference_rt.py --port COM3
    python inference_rt.py --port COM3 --model results/classical/best_model.joblib
    python inference_rt.py --list                # show available ports
    python inference_rt.py --port COM3 --verbose # show raw predictions each hop
"""

import argparse
import collections
import json
import os
import queue
import sys
import threading
import time
import warnings
warnings.filterwarnings("ignore", message="Empty filters detected", category=UserWarning)

import numpy as np
import joblib
import librosa
import librosa.filters
import serial
import serial.tools.list_ports

# ── Constants — must match preprocess_v0_windowed.py ────────────────────────
FS              = 2000
N_CHANNELS      = 2
N_MFCC          = 8
MFCC_N_FFT      = 64
MFCC_WIN        = 64
MFCC_HOP        = 32

# Pre-build mel filterbank once so librosa doesn't rebuild it every prediction
_MEL_BASIS = librosa.filters.mel(sr=FS, n_fft=MFCC_N_FFT, n_mels=N_MFCC)

GESTURE_NAMES = [
    "REST", "EXTENSION", "FLEXION", "ULNAR_DEV", "RADIAL_DEV",
    "GRIP", "ABDUCTION", "ADDUCTION", "SUPINATION", "PRONATION",
]
GESTURE_EMOJIS = {
    "REST":       "──",
    "EXTENSION":  "↑↑",
    "FLEXION":    "↓↓",
    "ULNAR_DEV":  "←U",
    "RADIAL_DEV": "→R",
    "GRIP":       "✊",
    "ABDUCTION":  "↔A",
    "ADDUCTION":  "↔D",
    "SUPINATION": "↺ ",
    "PRONATION":  "↻ ",
}

DEFAULT_WIN_MS  = 200
DEFAULT_HOP_MS  = 100
VOTE_WINDOW     = 5    # majority-vote over last N predictions
REST_CONFIDENCE = 0.60 # if SVM probability for REST > threshold, force REST
BAUD_RATE       = 921600


# ── Feature extraction (identical to preprocess_v0_windowed.py) ──────────────

def ac_couple(sig: np.ndarray) -> np.ndarray:
    """Per-window median subtraction — matches preprocess_v0_windowed.py exactly."""
    return sig - np.median(sig, axis=0, keepdims=True)


def mfcc_features(channel: np.ndarray) -> np.ndarray:
    # Use pre-built mel basis to avoid per-call filterbank reconstruction
    stft    = np.abs(librosa.stft(channel.astype(np.float32),
                                   n_fft=MFCC_N_FFT, hop_length=MFCC_HOP,
                                   win_length=MFCC_WIN)) ** 2
    log_mel = librosa.power_to_db(np.dot(_MEL_BASIS, stft))
    dct     = librosa.feature.mfcc(S=log_mel, n_mfcc=N_MFCC)
    return np.concatenate([dct.mean(1), dct.std(1), dct.max(1), dct.min(1)])


def tdd_features(channel: np.ndarray) -> np.ndarray:
    x   = channel.astype(np.float64)
    rms = np.sqrt(np.mean(x ** 2))
    rsq = rms ** 2
    sp  = np.sum(np.abs(x)) / (np.sqrt(len(x)) * np.sqrt(np.sum(x ** 2) + 1e-10) + 1e-10)
    diff = np.diff(x)
    irr  = np.sqrt(np.mean(diff ** 2)) / (rms + 1e-10)
    wl   = np.sum(np.abs(diff))
    thr  = 0.05 * rms
    zc   = float(np.sum((np.abs(diff) > thr) & (x[:-1] * x[1:] < 0)))
    return np.array([rsq, sp, irr, rms, wl, zc], dtype=np.float64)


def extract_features(window: np.ndarray) -> np.ndarray:
    """window: (win_samples, 2) → (116,) float32"""
    parts = []
    for ch in range(N_CHANNELS):
        parts.append(mfcc_features(window[:, ch]))
        parts.append(tdd_features(window[:, ch]))
    return np.concatenate(parts).astype(np.float32)


# ── Serial helpers ────────────────────────────────────────────────────────────

def list_ports():
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("No serial ports found.")
    else:
        for p in ports:
            print(f"  {p.device:<12} {p.description}")


def open_serial(port: str, baud: int = BAUD_RATE) -> serial.Serial:
    s = serial.Serial(port, baud, timeout=2.0)
    time.sleep(0.1)
    s.reset_input_buffer()
    print(f"Opened {port} @ {baud} baud")
    return s


def read_sample(ser: serial.Serial):
    """
    Read one line from the ESP32 streaming firmware.
    Expected format: "ch1,ch2\\n"  (integers, ADC counts 0-4095)
    Returns (ch1, ch2) as floats, or None on parse error.
    """
    try:
        line = ser.readline().decode("ascii", errors="ignore").strip()
        if not line:
            return None
        parts = line.split(",")
        if len(parts) < 2:
            return None
        return float(parts[0]), float(parts[1])
    except Exception:
        return None


# ── Inference engine ──────────────────────────────────────────────────────────

class GesturePredictor:
    def __init__(self, model, win_samples: int, hop_samples: int,
                 vote_window: int = VOTE_WINDOW, verbose: bool = False):
        self.model        = model
        self.win          = win_samples
        self.hop          = hop_samples
        self.verbose      = verbose
        self.buffer       = collections.deque(maxlen=win_samples)
        self.samples_since_hop = 0
        self.recent_preds = collections.deque(maxlen=vote_window)
        self.current_gesture = "REST"
        self.gesture_start   = time.time()
        self.last_print      = ""

    def push(self, ch1: float, ch2: float):
        self.buffer.append([ch1, ch2])
        self.samples_since_hop += 1

        if len(self.buffer) < self.win:
            return   # buffer not full yet

        if self.samples_since_hop >= self.hop:
            self.samples_since_hop = 0
            self._predict()

    def _predict(self):
        window = np.array(self.buffer, dtype=np.float32)  # (win, 2)
        window = ac_couple(window)
        feat   = extract_features(window).reshape(1, -1)

        pred_id = int(self.model.predict(feat)[0])

        # REST gating via probability if model supports it
        if hasattr(self.model, "predict_proba"):
            proba    = self.model.predict_proba(feat)[0]
            rest_p   = proba[0]
            if rest_p > REST_CONFIDENCE:
                pred_id = 0
            if self.verbose:
                top2 = np.argsort(proba)[::-1][:2]
                conf_str = "  ".join(
                    f"{GESTURE_NAMES[i]}={proba[i]:.2f}" for i in top2)
                print(f"  raw: {GESTURE_NAMES[pred_id]:<14}  {conf_str}")

        self.recent_preds.append(pred_id)
        voted = self._majority_vote()

        if voted != self.current_gesture:
            held = time.time() - self.gesture_start
            self.current_gesture = voted
            self.gesture_start   = time.time()
            self._print_gesture(voted, held)

    def _majority_vote(self) -> str:
        counts = collections.Counter(self.recent_preds)
        return GESTURE_NAMES[counts.most_common(1)[0][0]]

    def _print_gesture(self, gesture: str, prev_held_s: float):
        icon = GESTURE_EMOJIS.get(gesture, "  ")
        line = f"  {icon}  {gesture:<14}  (prev held {prev_held_s:.1f}s)"
        if line != self.last_print:
            print(line, flush=True)
            self.last_print = line


# ── Serial reader thread ──────────────────────────────────────────────────────

def _serial_reader(ser: serial.Serial, sample_q: queue.Queue, stop: threading.Event):
    """
    Dedicated thread: reads from serial as fast as possible and enqueues
    (ch1, ch2) tuples.  Never blocks on feature extraction.
    """
    while not stop.is_set():
        sample = read_sample(ser)
        if sample is not None:
            sample_q.put(sample)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Real-time EMG gesture inference")
    parser.add_argument("--port",    default=None,
                        help="Serial port, e.g. COM5 or /dev/ttyUSB0")
    parser.add_argument("--baud",    type=int, default=BAUD_RATE)
    parser.add_argument("--model",   default="results/classical/best_model.joblib",
                        help="Path to saved joblib model from train_classical.py")
    parser.add_argument("--win-ms",  type=int, default=DEFAULT_WIN_MS)
    parser.add_argument("--hop-ms",  type=int, default=DEFAULT_HOP_MS)
    parser.add_argument("--list",    action="store_true",
                        help="List available serial ports and exit")
    parser.add_argument("--verbose", action="store_true",
                        help="Print raw per-hop predictions with probabilities")
    args = parser.parse_args()

    if args.list:
        list_ports(); return

    if args.port is None:
        print("Error: --port required.  Use --list to see available ports.")
        sys.exit(1)

    if not os.path.exists(args.model):
        print(f"Error: model not found at {args.model}")
        print("  Run:  python train_classical.py --data data/processed_v0_windowed")
        sys.exit(1)

    WIN = int(args.win_ms * FS / 1000)
    HOP = int(args.hop_ms * FS / 1000)

    print(f"Loading model from {args.model} …")
    model = joblib.load(args.model)

    predictor  = GesturePredictor(model, WIN, HOP, verbose=args.verbose)
    sample_q   = queue.Queue(maxsize=4000)   # ~2 s buffer headroom
    stop_event = threading.Event()

    print(f"\nWindow {args.win_ms} ms  |  Hop {args.hop_ms} ms")
    print("Connect electrodes and start performing gestures.\n")
    print(f"{'─'*50}")
    print(f"  {'icon':<4}  {'Gesture':<14}  hold time")
    print(f"{'─'*50}")

    ser = open_serial(args.port, args.baud)

    reader = threading.Thread(target=_serial_reader,
                              args=(ser, sample_q, stop_event),
                              daemon=True)
    reader.start()

    sample_count = 0
    dropped      = 0
    t_start      = time.time()

    try:
        while True:
            try:
                sample = sample_q.get(timeout=2.0)
            except queue.Empty:
                print("  [warning: no data for 2 s — check ESP32 connection]")
                continue

            predictor.push(*sample)
            sample_count += 1

            if sample_q.qsize() > 3000:
                # Queue backing up — drop oldest to stay real-time
                n = sample_q.qsize() - 200
                for _ in range(n):
                    try: sample_q.get_nowait()
                    except queue.Empty: break
                dropped += n

            if sample_count % (FS * 5) == 0:
                elapsed = time.time() - t_start
                rate    = sample_count / elapsed
                qsize   = sample_q.qsize()
                print(f"  [throughput: {rate:.0f} Hz  queue: {qsize}  dropped: {dropped}]",
                      flush=True)

    except KeyboardInterrupt:
        elapsed = time.time() - t_start
        print(f"\nStopped.  {sample_count} samples in {elapsed:.1f}s  "
              f"({sample_count/elapsed:.0f} Hz)  dropped: {dropped}")
    finally:
        stop_event.set()
        ser.close()


if __name__ == "__main__":
    main()
