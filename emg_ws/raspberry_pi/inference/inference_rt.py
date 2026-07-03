#!/usr/bin/env python3
"""
inference_rt.py — Real-time EMG gesture inference from ESP32

Reads the EMG stream from the ESP32 over Serial, maintains a sliding-window
buffer of the MODEL_CHANNELS, extracts the SAME MFCC+TDD features as
preprocess_v0_windowed.py every hop, and predicts the current gesture with the
saved classical model.

Stream format: the firmware emits the 11-column collection CSV
    participant_id,cycle,gesture_name,gesture_id,trial,sample_idx,
    v3_env,cheez_raw,cheez_filt,cheez_env,wear
This script ignores the metadata columns and pulls only the MODEL_CHANNELS
(by their column index) from each numeric data row. Header / DATA_START /
status lines are skipped automatically (their channel fields aren't numeric).

To stay perfectly in sync with training, the feature functions and MFCC params
are imported directly from preprocess_v0_windowed.py — there is no second copy.

Usage:
    python inference_rt.py --port COM3
    python inference_rt.py --port COM3 --model results/classical/best_model.joblib
    python inference_rt.py --list
    python inference_rt.py --port COM3 --verbose
"""

import argparse
import collections
import os
import queue
import sys
import threading
import time
import warnings
warnings.filterwarnings("ignore", message="Empty filters detected", category=UserWarning)

import numpy as np
import joblib

# pyserial is OPTIONAL — only the USB path (--port / --list) needs it. WiFi/TCP
# (--host) uses sockets, so a WiFi-only deploy (e.g. the Pi) needs no pyserial.
try:
    import serial
    import serial.tools.list_ports
    _HAVE_SERIAL = True
except Exception:
    serial = None
    _HAVE_SERIAL = False

import emg_config as cfg
# Reuse the EXACT feature pipeline used for training (no duplicate constants).
from preprocess_v0_windowed import ac_couple, extract_features

# ── Constants from the shared config ────────────────────────────────────────
FS             = cfg.FS              # 500
N_CHANNELS     = cfg.N_CHANNELS
MODEL_CHANNELS = cfg.MODEL_CHANNELS
GESTURE_NAMES  = cfg.GESTURE_NAMES
# Column index of each model channel within a full 11-field CSV data row.
CH_INDICES     = [cfg.CSV_COLUMNS.index(c) for c in MODEL_CHANNELS]

GESTURE_ICONS = {
    "REST": "--", "EXTENSION": "Ext", "FLEXION": "Flx",
    "ULNAR_DEV": "Uln", "RADIAL_DEV": "Rad", "GRIP": "Grp",
    "DOWNWARD_ROT_CW": "Dcw", "DOWNWARD_ROT_CCW": "Dcc",
    "FRONT_ROT_CW": "Fcw", "FRONT_ROT_CCW": "Fcc",
    "EXTEND_ELBOW": "ElX", "RETRACT_ELBOW": "ElR",
}

DEFAULT_WIN_MS  = 200
DEFAULT_HOP_MS  = 100
VOTE_WINDOW     = 5     # majority-vote over last N predictions
REST_CONFIDENCE = 0.60  # if model probability for REST > threshold, force REST
BAUD_RATE       = 921600


# ── Serial helpers ────────────────────────────────────────────────────────────

def list_ports():
    if not _HAVE_SERIAL:
        print("pyserial not installed — USB ports unavailable (WiFi via --host needs no serial).")
        return
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("No serial ports found.")
    for p in ports:
        print(f"  {p.device:<12} {p.description}")


def open_serial(port: str, baud: int = BAUD_RATE):
    if not _HAVE_SERIAL:
        print("Error: USB serial needs pyserial (`pip install pyserial`), or use --host for WiFi.")
        sys.exit(1)
    s = serial.Serial(port, baud, timeout=2.0)
    time.sleep(0.1)
    s.reset_input_buffer()
    print(f"Opened {port} @ {baud} baud")
    return s


class _TcpLink:
    """pyserial look-alike over a TCP socket, so the same read path works with the
    WiFi streaming firmware (semg_stream_live_wifi.ino). Mirrors collection_gui's
    _TcpLink: exposes .readline()->bytes and .close(). The ESP is a TCP server
    (STA mode: join the router, connect to its IP / esp32.local on port 5005)."""

    def __init__(self, host: str, port: int = 5005, timeout: float = 2.0):
        import socket
        self.sock = socket.create_connection((host, port), timeout=5.0)
        self.sock.settimeout(timeout)
        self._buf = b""
        print(f"Connected to ESP32 at {host}:{port} (WiFi/TCP)")

    def readline(self) -> bytes:
        import socket
        while b"\n" not in self._buf:
            try:
                chunk = self.sock.recv(4096)
            except socket.timeout:
                return b""                      # mimic pyserial read timeout
            if chunk == b"":
                raise OSError("TCP connection closed by ESP32")
            self._buf += chunk
        line, _, self._buf = self._buf.partition(b"\n")
        return line + b"\n"

    def reset_input_buffer(self):
        self._buf = b""

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass


def open_source(args):
    """Open the gesture-data source: WiFi/TCP if --host given, else USB serial."""
    if args.host:
        return _TcpLink(args.host, args.tcp_port)
    return open_serial(args.port, args.baud)


def read_sample(ser):
    """Read one CSV data row, return the MODEL_CHANNELS values as a list of
    floats, or None for header / status / malformed lines."""
    try:
        line = ser.readline().decode("ascii", errors="ignore").strip()
        if not line:
            return None
        parts = line.split(",")
        # Lean live stream = just the model channels, in order (e.g. "v3_env,cheez_env").
        if len(parts) == N_CHANNELS:
            return [float(p) for p in parts]
        # Full 11-column layout (collection / semg_stream_live STREAM_11COL): by index.
        if len(parts) >= len(cfg.CSV_COLUMNS):
            return [float(parts[i]) for i in CH_INDICES]
        return None
    except Exception:
        return None


# ── Inference engine ──────────────────────────────────────────────────────────

class GesturePredictor:
    def __init__(self, model, win_samples, hop_samples,
                 vote_window=VOTE_WINDOW, verbose=False, mains_notch=None):
        self.model   = model
        self.win     = win_samples
        self.hop     = hop_samples
        self.verbose = verbose
        # Causal mains notch applied to each sample BEFORE it enters the window
        # buffer — matches training order (mains -> window -> ac_couple) and must
        # match the --mains the deployed model was trained with. None = off.
        self.mains   = mains_notch
        self.buffer  = collections.deque(maxlen=win_samples)
        self.samples_since_hop = 0
        self.recent_preds = collections.deque(maxlen=vote_window)
        self.current_gesture = "REST"
        self.gesture_start   = time.time()
        self.last_print      = ""

    def push(self, sample):
        if self.mains is not None:
            sample = self.mains.process(sample)  # causal 50/60 Hz notch + harmonics
        self.buffer.append(list(sample))         # sample: [ch0, ch1, ...]
        self.samples_since_hop += 1
        if len(self.buffer) < self.win:
            return
        if self.samples_since_hop >= self.hop:
            self.samples_since_hop = 0
            self._predict()

    def _predict(self):
        window = ac_couple(np.array(self.buffer, dtype=np.float32))  # (win, N_CH)

        # Two model families share this loop. The CRNN (rcnn_model.RCNNInference)
        # consumes the raw window directly and returns a full 12-wide, gesture-id
        # -indexed probability vector. The classical sklearn pipeline consumes the
        # 76-D MFCC+TDD features; its predict_proba is ordered by model.classes_,
        # so we scatter it into the same 12-wide id-indexed layout. Either way
        # `proba` below is indexed by gesture id (idx 0 == REST), which keeps the
        # REST gate and the GESTURE_NAMES[i] labels correct for any class subset.
        if hasattr(self.model, "predict_window"):
            proba   = self.model.predict_proba_window(window)   # (12,) id-indexed
            pred_id = int(np.argmax(proba))
        else:
            feat    = extract_features(window).reshape(1, -1)
            pred_id = int(self.model.predict(feat)[0])
            proba   = None
            if hasattr(self.model, "predict_proba"):
                raw   = self.model.predict_proba(feat)[0]
                proba = np.zeros(len(GESTURE_NAMES), dtype=float)
                proba[self.model.classes_] = raw               # -> id-indexed

        if proba is not None:
            if proba[0] > REST_CONFIDENCE:
                pred_id = 0
            if self.verbose:
                top2 = np.argsort(proba)[::-1][:2]
                conf = "  ".join(f"{GESTURE_NAMES[i]}={proba[i]:.2f}" for i in top2)
                print(f"  raw: {GESTURE_NAMES[pred_id]:<16}  {conf}")

        self.recent_preds.append(pred_id)
        voted = GESTURE_NAMES[collections.Counter(self.recent_preds).most_common(1)[0][0]]
        if voted != self.current_gesture:
            held = time.time() - self.gesture_start
            self.current_gesture = voted
            self.gesture_start   = time.time()
            self._print_gesture(voted, held)

    def _print_gesture(self, gesture, prev_held_s):
        icon = GESTURE_ICONS.get(gesture, "  ")
        line = f"  {icon:<4} {gesture:<16}  (prev held {prev_held_s:.1f}s)"
        if line != self.last_print:
            print(line, flush=True)
            self.last_print = line


def _serial_reader(ser, sample_q, stop):
    while not stop.is_set():
        sample = read_sample(ser)
        if sample is not None:
            sample_q.put(sample)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Real-time EMG gesture inference")
    parser.add_argument("--port",    default=None, help="Serial port, e.g. COM5 or /dev/ttyUSB0")
    parser.add_argument("--baud",    type=int, default=BAUD_RATE)
    parser.add_argument("--host",    default=None,
                        help="ESP32 IP for WiFi/TCP streaming (e.g. 192.168.1.50 or esp32.local). "
                             "Overrides --port; pairs with semg_stream_live_wifi.ino.")
    parser.add_argument("--tcp-port", type=int, default=5005, help="WiFi firmware TCP port (default 5005)")
    parser.add_argument("--mains",   default="50", choices=["off", "50", "60"],
                        help="Live causal mains notch (Hz) applied while reading. Default 50 (Egypt). "
                             "MUST match the --mains the model was trained with.")
    parser.add_argument("--model",   default="results/classical/best_model.joblib")
    parser.add_argument("--win-ms",  type=int, default=DEFAULT_WIN_MS)
    parser.add_argument("--hop-ms",  type=int, default=DEFAULT_HOP_MS)
    parser.add_argument("--list",    action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.list:
        list_ports(); return
    if args.host is None and args.port is None:
        print("Error: give --port (USB) or --host (WiFi).  Use --list to see serial ports."); sys.exit(1)
    if not os.path.exists(args.model):
        print(f"Error: model not found at {args.model}")
        print("  Run:  python train_classical.py --data data/processed_v0_windowed")
        sys.exit(1)

    WIN = int(args.win_ms * FS / 1000)
    HOP = int(args.hop_ms * FS / 1000)

    print(f"Loading model from {args.model} ...")
    model = joblib.load(args.model)

    mains_hz = 0 if args.mains == "off" else int(args.mains)
    notch = None
    if mains_hz:
        from mains_notch import MainsNotch
        notch = MainsNotch(N_CHANNELS, fs=FS, mains_hz=mains_hz)

    predictor  = GesturePredictor(model, WIN, HOP, verbose=args.verbose, mains_notch=notch)
    sample_q   = queue.Queue(maxsize=4000)
    stop_event = threading.Event()

    src = "WiFi " + args.host if args.host else "serial " + str(args.port)
    print(f"\nChannels {MODEL_CHANNELS}  |  Window {args.win_ms} ms  |  Hop {args.hop_ms} ms  "
          f"|  mains {args.mains} Hz  |  source {src}")
    print("Connect electrodes and start performing gestures.\n")
    print(f"{'-'*50}")

    ser = open_source(args)
    reader = threading.Thread(target=_serial_reader, args=(ser, sample_q, stop_event), daemon=True)
    reader.start()

    sample_count, dropped, t_start = 0, 0, time.time()
    try:
        while True:
            try:
                sample = sample_q.get(timeout=2.0)
            except queue.Empty:
                print("  [warning: no data for 2 s — check ESP32 connection]")
                continue
            predictor.push(sample)
            sample_count += 1
            if sample_q.qsize() > 3000:
                n = sample_q.qsize() - 200
                for _ in range(n):
                    try: sample_q.get_nowait()
                    except queue.Empty: break
                dropped += n
            if sample_count % (FS * 5) == 0:
                elapsed = time.time() - t_start
                print(f"  [throughput: {sample_count/elapsed:.0f} Hz  "
                      f"queue: {sample_q.qsize()}  dropped: {dropped}]", flush=True)
    except KeyboardInterrupt:
        elapsed = time.time() - t_start
        print(f"\nStopped.  {sample_count} samples in {elapsed:.1f}s  "
              f"({sample_count/max(elapsed,1e-9):.0f} Hz)  dropped: {dropped}")
    finally:
        stop_event.set()
        ser.close()


if __name__ == "__main__":
    main()
