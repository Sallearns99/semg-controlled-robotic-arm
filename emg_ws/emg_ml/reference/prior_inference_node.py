#!/usr/bin/env python3
"""
inference_node.py
─────────────────
Accumulates 2000 samples from /emg_raw, applies Muscle Sensor v3
preprocessing, extracts features, runs ONNX inference, publishes gesture.

Subscribes:  /emg_raw           Float32MultiArray  [4 floats per message @ 2000 Hz]
Publishes:   /detected_gesture   String
             /inference_result   String  (JSON: gesture + confidence + all probs)
"""

import os, sys, json

# ── Force venv packages before system packages ───────────────────────────────
# Ubuntu 24.04 system scipy compiled against NumPy 1.x crashes on NumPy 2.x.
# Must happen before ANY scipy/numpy import.
_VENV = os.path.expanduser("~/emg_env/lib/python3.12/site-packages")
if _VENV not in sys.path:
    sys.path.insert(0, _VENV)
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
from collections import deque

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String

try:
    import onnxruntime as ort
except ImportError:
    sys.exit('[FATAL] onnxruntime missing.  pip install onnxruntime')

try:
    from scipy.fft import fft, dct
    from scipy.signal import get_window, butter, filtfilt
except ImportError:
    sys.exit('[FATAL] scipy missing.  pip install scipy')

# ── Constants — must match train_emg_model.py exactly ────────────────────────
WINDOW_SAMPLES = 2000
N_CH           = 4
FS             = 2000
N_MFCC         = 13
N_MELS         = 26
SMOOTH_HZ      = 10.0   # Muscle Sensor v3 RC-smoother simulation


# ── Preprocessing — simulate Muscle Sensor v3 hardware ───────────────────────

def _butter_lp(cutoff, fs, order=2):
    """Pre-compute Butterworth LP coefficients once at startup."""
    b, a = butter(order, cutoff / (fs / 2.0), btype='low')
    return b, a


def simulate_sensor_v3(window, b, a):
    """
    Rectify + low-pass smooth each channel.
    Replicates: AD8226 → full-wave rectifier → RC smoother (Muscle Sensor v3).
    window  : [2000 × 4]  raw differential EMG
    returns : [2000 × 4]  all values >= 0
    """
    rect = np.abs(window)
    out  = np.zeros_like(rect)
    for ch in range(rect.shape[1]):
        out[:, ch] = filtfilt(b, a, rect[:, ch])
    return np.clip(out, 0.0, None)


# ── Feature extraction — matches train_emg_model.py exactly ──────────────────

def _mfcc(sig, fs=FS):
    """
    Mean MFCCs for one channel.
    Window 25 ms Hamming, hop = 50-30 = 20 samples — matches MATLAB mfcc().
    Returns [13] float32.
    """
    win_len  = int(0.025 * fs)            # 50 samples @ 2000 Hz
    hop_len  = win_len - int(0.015 * fs)  # 20 samples
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
    6 TDD features for rectified + smoothed EMG.
    Matches _tdd_rectified() in train_emg_model.py exactly.

      1. MAV — mean amplitude            (replaces Sparseness)
      2. RMS — signal power
      3. VAR — envelope spread
      4. WL  — waveform length
      5. SSC — slope sign changes        (replaces Zero Crossing)
      6. HM  — Hjorth mobility           (replaces Irregularity Rate)

    Returns [6] float32.
    """
    s   = sig.astype(np.float64)
    eps = 1e-8
    d   = np.diff(s)
    return np.array([
        float(np.mean(s)),                                    # MAV
        float(np.sqrt(np.mean(s ** 2))),                      # RMS
        float(np.var(s)),                                     # VAR
        float(np.sum(np.abs(d))),                             # WL
        float(np.sum(np.abs(np.diff(np.sign(d))) > 0)),       # SSC
        float(np.sqrt(np.var(d) / (np.var(s) + eps))),        # HM
    ], dtype=np.float32)


def extract_features(window):
    """
    window : [2000 × 4] rectified+smoothed float32
    returns: [76] float32  =  4 ch × (13 MFCC + 6 TDD)

    NOTE: standalone function, NOT a method — takes only window, no self.
    """
    feats = []
    for ch in range(window.shape[1]):
        feats.append(_mfcc(window[:, ch]))   # [13]
        feats.append(_tdd(window[:, ch]))    # [6]
    return np.concatenate(feats).astype(np.float32)   # [76]


def stable_softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


# ── ROS2 node ─────────────────────────────────────────────────────────────────

class InferenceNode(Node):

    def __init__(self):
        super().__init__('emg_inference_node')

        self.declare_parameter(
            'model_dir',
            os.path.expanduser('~/ros2_emg_ws/src/emg_inference/models'))
        self.declare_parameter('smooth_hz', SMOOTH_HZ)

        model_dir = os.path.expanduser(self.get_parameter('model_dir').value)
        smooth_hz = float(self.get_parameter('smooth_hz').value)

        # Verify model files
        for name in ('emg_model.onnx', 'emg_norm_stats.json', 'emg_class_map.json'):
            p = os.path.join(model_dir, name)
            if not os.path.exists(p):
                self.get_logger().fatal(f'Missing: {p}')
                raise FileNotFoundError(p)

        # ONNX session
        self.session = ort.InferenceSession(os.path.join(model_dir, 'emg_model.onnx'))
        inp = self.session.get_inputs()[0]
        self.get_logger().info(f'ONNX input: {inp.name}  shape={inp.shape}')

        # Normalisation stats  [76]
        with open(os.path.join(model_dir, 'emg_norm_stats.json')) as f:
            stats = json.load(f)
        self.mean = np.array(stats['mean'], dtype=np.float32)
        self.std  = np.array(stats['std'],  dtype=np.float32)

        # Class names
        with open(os.path.join(model_dir, 'emg_class_map.json')) as f:
            self.classes = json.load(f)['class_names']

        # Butterworth LP coefficients — computed once, reused every inference
        self.b_lp, self.a_lp = _butter_lp(smooth_hz, FS)

        # Ring buffer + non-overlapping window counter
        self.buf          = deque(maxlen=WINDOW_SAMPLES)
        self.sample_count = 0

        # Queue depth 100 — must not drop at 2000 Hz
        self.sub = self.create_subscription(
            Float32MultiArray, '/emg_raw', self._emg_cb, 100)
        self.pub_gesture = self.create_publisher(String, '/detected_gesture', 10)
        self.pub_result  = self.create_publisher(String, '/inference_result',  10)

        self.get_logger().info(
            f'Inference node ready\n'
            f'  model_dir : {model_dir}\n'
            f'  smooth_hz : {smooth_hz} Hz\n'
            f'  classes   : {self.classes}\n'
            f'  Waiting for {WINDOW_SAMPLES} samples on /emg_raw ...')

    def _emg_cb(self, msg):
        row = np.array(msg.data, dtype=np.float32)
        if len(row) < N_CH:
            row = np.pad(row, (0, N_CH - len(row)))
        self.buf.append(row[:N_CH])
        self.sample_count += 1

        if self.sample_count >= WINDOW_SAMPLES:
            self.sample_count = 0
            self._run_inference()

    def _run_inference(self):
        # 1. Assemble [2000 × 4]
        window = np.stack(list(self.buf), axis=0)

        # 2. Simulate Muscle Sensor v3 (rectify + smooth)
        window = simulate_sensor_v3(window, self.b_lp, self.a_lp)

        # 3. Extract features → [76]
        features = extract_features(window)

        # 4. Z-score normalise with training statistics
        features = (features - self.mean) / (self.std + 1e-8)

        # 5. Reshape to [1, 76, 1] — Conv1D expects (batch, channels, seq_len)
        features = features.reshape(1, 76, 1).astype(np.float32)

        # 6. ONNX forward pass
        logits  = self.session.run(['logits'], {'features': features})[0]  # [1, 10]
        probs   = stable_softmax(logits[0])
        pred    = int(np.argmax(probs))
        conf    = float(probs[pred])
        gesture = self.classes[pred]

        self.get_logger().info(f'Predicted: {gesture:12s}  conf={conf*100:.1f}%')

        g = String(); g.data = gesture
        self.pub_gesture.publish(g)

        r = String()
        r.data = json.dumps(dict(
            gesture    = gesture,
            confidence = round(conf, 4),
            probs      = {self.classes[i]: round(float(probs[i]), 4)
                          for i in range(len(self.classes))},
        ))
        self.pub_result.publish(r)


def main(args=None):
    rclpy.init(args=args)
    try:
        rclpy.spin(InferenceNode())
    except (KeyboardInterrupt, FileNotFoundError):
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
