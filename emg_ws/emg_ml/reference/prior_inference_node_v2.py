#!/usr/bin/env python3
"""
inference_node.py  —  v2 (raw signal, matches train_emg_v2.py)
───────────────────────────────────────────────────────────────
No feature extraction — raw signal fed directly to CNN.

Pipeline per inference window:
  1. Accumulate 2000 samples from /emg_raw
  2. Rectify + 10 Hz LP  (simulate Muscle Sensor v3)
  3. Per-channel z-score normalise  (training stats)
  4. Reshape [1, 4, 2000]
  5. ONNX forward  →  gesture + confidence

Subscribes : /emg_raw          Float32MultiArray  [4 floats/msg @ 2000 Hz]
Publishes  : /detected_gesture  String
             /inference_result  String  JSON
"""

import os, sys, json
import numpy as np
from collections import deque

# ── Force venv before system packages (Ubuntu 24.04 scipy fix) ───────────────
_VENV = os.path.expanduser('~/emg_env/lib/python3.12/site-packages')
if _VENV not in sys.path:
    sys.path.insert(0, _VENV)
# ─────────────────────────────────────────────────────────────────────────────

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String

try:
    import onnxruntime as ort
except ImportError:
    sys.exit('[FATAL] pip install onnxruntime')

try:
    from scipy.signal import butter, filtfilt
except ImportError:
    sys.exit('[FATAL] pip install scipy')

WINDOW    = 2000
N_CH      = 4
FS        = 2000
SMOOTH_HZ = 10.0


def butter_lp(cutoff, fs, order=2):
    b, a = butter(order, cutoff / (fs / 2.0), btype='low')
    return b, a


def simulate_sensor(window, b, a):
    rect = np.abs(window)
    out  = np.zeros_like(rect)
    for ch in range(rect.shape[1]):
        out[:, ch] = filtfilt(b, a, rect[:, ch])
    return np.clip(out, 0.0, None)


def stable_softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


class InferenceNode(Node):

    def __init__(self):
        super().__init__('emg_inference_node')

        self.declare_parameter(
            'model_dir',
            os.path.expanduser('~/ros2_emg_ws/src/emg_inference/models'))
        self.declare_parameter('smooth_hz', SMOOTH_HZ)

        model_dir = os.path.expanduser(self.get_parameter('model_dir').value)
        smooth_hz = float(self.get_parameter('smooth_hz').value)

        # Verify files
        for name in ('emg_model.onnx', 'emg_norm_stats.json', 'emg_class_map.json'):
            p = os.path.join(model_dir, name)
            if not os.path.exists(p):
                self.get_logger().fatal(f'Missing: {p}')
                raise FileNotFoundError(p)

        self.session = ort.InferenceSession(os.path.join(model_dir, 'emg_model.onnx'))
        inp = self.session.get_inputs()[0]
        self.get_logger().info(f'ONNX input: {inp.name}  shape={inp.shape}')

        # Normalisation — per-channel stats from training
        with open(os.path.join(model_dir, 'emg_norm_stats.json')) as f:
            stats = json.load(f)

        # Support both v1 format (mean/std over 76 features) and
        # v2 format (ch_mean/ch_std per channel)
        if 'ch_mean' in stats:
            self.ch_mean = np.array(stats['ch_mean'], dtype=np.float32)  # [4]
            self.ch_std  = np.array(stats['ch_std'],  dtype=np.float32)  # [4]
            self.norm_mode = 'per_channel'
        else:
            # v1 fallback — feature-level stats; inference node will warn
            self.ch_mean = None
            self.ch_std  = None
            self.norm_mode = 'none'
            self.get_logger().warn(
                'emg_norm_stats.json has no ch_mean/ch_std — '
                'retrain with train_emg_v2.py for best results')

        with open(os.path.join(model_dir, 'emg_class_map.json')) as f:
            self.classes = json.load(f)['class_names']

        self.b_lp, self.a_lp = butter_lp(smooth_hz, FS)
        self.buf          = deque(maxlen=WINDOW)
        self.sample_count = 0

        self.sub = self.create_subscription(
            Float32MultiArray, '/emg_raw', self._cb, 100)
        self.pub_g = self.create_publisher(String, '/detected_gesture', 10)
        self.pub_r = self.create_publisher(String, '/inference_result',  10)

        self.get_logger().info(
            f'Inference node v2 ready\n'
            f'  model_dir  : {model_dir}\n'
            f'  smooth_hz  : {smooth_hz} Hz\n'
            f'  norm_mode  : {self.norm_mode}\n'
            f'  classes    : {self.classes}\n'
            f'  Waiting for {WINDOW} samples on /emg_raw ...')

    def _cb(self, msg):
        row = np.array(msg.data, dtype=np.float32)
        if len(row) < N_CH:
            row = np.pad(row, (0, N_CH - len(row)))
        self.buf.append(row[:N_CH])
        self.sample_count += 1
        if self.sample_count >= WINDOW:
            self.sample_count = 0
            self._infer()

    def _infer(self):
        # [2000, 4]
        window = np.stack(list(self.buf), axis=0)

        # Step 1: simulate Muscle Sensor v3
        window = simulate_sensor(window, self.b_lp, self.a_lp)

        # Step 2: per-channel z-score normalise
        if self.norm_mode == 'per_channel':
            window = (window - self.ch_mean) / (self.ch_std + 1e-8)

        # Step 3: channels-first [1, 4, 2000]
        x = window.T[np.newaxis, :, :].astype(np.float32)

        # Step 4: ONNX
        inp_name = self.session.get_inputs()[0].name
        logits   = self.session.run(['logits'], {inp_name: x})[0]
        probs    = stable_softmax(logits[0])
        pred     = int(np.argmax(probs))
        conf     = float(probs[pred])
        gesture  = self.classes[pred]

        self.get_logger().info(f'Predicted: {gesture:12s}  conf={conf*100:.1f}%')

        g = String(); g.data = gesture
        self.pub_g.publish(g)

        r = String()
        r.data = json.dumps(dict(
            gesture=gesture, confidence=round(conf, 4),
            probs={self.classes[i]: round(float(probs[i]), 4)
                   for i in range(len(self.classes))}))
        self.pub_r.publish(r)


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
