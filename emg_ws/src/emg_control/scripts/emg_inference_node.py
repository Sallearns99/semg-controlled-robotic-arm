#!/usr/bin/env python3
"""emg_inference_node.py — real-time EMG gesture inference for ROS 2.

Ports the proven pipeline from inference_rt.py (the 2026-06 "moving to ubuntu"
overhaul: 500 Hz, 12 gestures, v3_env + cheez_env channels):
  ESP32 serial (11-col collection CSV @ 500 Hz, 921600 baud)
    -> pull v3_env (col 6) + cheez_env (col 9)
    -> 200 ms sliding window / 100 ms hop
    -> per-window AC-couple (median-subtract) -> MFCC+TDD = 76 features
    -> sklearn model.predict (+ predict_proba)
    -> majority vote over last N -> REST gate
Publishes:
  /detected_gesture  std_msgs/String   stable voted gesture name
  /inference_result  std_msgs/String   JSON {gesture, confidence, probs}

The arm bridge only needs /detected_gesture, so the arm can also be tested by
hand-publishing to that topic without running this node.
"""
import collections
import json
import threading
import time
import warnings

warnings.filterwarnings("ignore", message="Empty filters detected", category=UserWarning)

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# ── Feature-extraction constants — MUST match preprocess_v0_windowed.py ──
# 2026-06 overhaul: collection is now 500 Hz with the CheezsEMG-paced firmware.
FS, N_CHANNELS = 500, 2
# Windowed MFCC params for 500 Hz / 100-sample (200 ms) windows. NOTE: the new
# preprocess_v0_windowed.py passes NO n_mels -> librosa default (128). We must
# replicate that exactly (no n_mels kwarg below), or the features won't match.
N_MFCC, MFCC_N_FFT, MFCC_WIN, MFCC_HOP = 8, 32, 32, 16

# 12-gesture canonical set — matches emg_config.py / semg_data_collection.ino.
# voted_id is clamped to this list, so a model trained on a subset still runs.
GESTURE_NAMES = ["REST", "EXTENSION", "FLEXION", "ULNAR_DEV", "RADIAL_DEV",
                 "GRIP", "DOWNWARD_ROT_CW", "DOWNWARD_ROT_CCW",
                 "FRONT_ROT_CW", "FRONT_ROT_CCW", "EXTEND_ELBOW", "RETRACT_ELBOW"]

# Stream layout: the firmware emits the 11-column collection CSV. The model uses
# two of the stored waves (emg_config.MODEL_CHANNELS = v3_env + cheez_env).
V3_ENV_COL, CHEEZ_ENV_COL = 6, 9   # column indices within an 11-field data row

# librosa is imported lazily inside mfcc_features() — only the CLASSICAL feature
# path needs it. The CRNN runs on the raw window (predict_window) and never calls
# extract_features, so a CRNN deploy needs neither librosa nor sklearn.

# ─────────────────────────────────────────────────────────────────────────────
# CHANNELS (2026-06 overhaul): both model channels are now ENVELOPES resting
# near 0 — v3_env (Muscle Sensor v3, on-board rectified) and cheez_env (the
# CHEEZ.semg library's on-DEVICE envelope of its biphasic signal). The old
# software "AC-couple + rectify CH2" step is gone: the ESP now produces the
# envelope, so a single per-window median-subtract on each channel is the whole
# transform — identical to preprocess_v0_windowed.ac_couple.
# ─────────────────────────────────────────────────────────────────────────────


def ac_couple(sig):
    return sig - np.median(sig, axis=0, keepdims=True)


def preprocess_window(window):
    """Per-window AC-couple — identical to preprocess_v0_windowed.ac_couple."""
    return ac_couple(window.astype(np.float32))


def mfcc_features(channel):
    # MUST match preprocess_v0_windowed.py EXACTLY: n_mfcc=8, n_fft=32, win=32,
    # hop=16, and NO n_mels kwarg (librosa default 128 — most filters are empty
    # for n_fft=32, but that is what the model was trained on). Retrain if changed.
    import librosa  # lazy: only the classical feature path needs it (see header)
    m = librosa.feature.mfcc(y=channel.astype(np.float32), sr=FS, n_mfcc=N_MFCC,
                             n_fft=MFCC_N_FFT, hop_length=MFCC_HOP, win_length=MFCC_WIN)
    return np.concatenate([m.mean(1), m.std(1), m.max(1), m.min(1)])


def tdd_features(channel):
    x = channel.astype(np.float64)
    rms = np.sqrt(np.mean(x ** 2))
    rsq = rms ** 2
    sp = np.sum(np.abs(x)) / (np.sqrt(len(x)) * np.sqrt(np.sum(x ** 2)) + 1e-10)
    diff = np.diff(x)
    irr = np.sqrt(np.mean(diff ** 2)) / (rms + 1e-10)
    wl = np.sum(np.abs(diff))
    thr = 0.05 * rms
    zc = float(np.sum((np.abs(diff) > thr) & (x[:-1] * x[1:] < 0)))
    return np.array([rsq, sp, irr, rms, wl, zc], dtype=np.float64)


def extract_features(window):
    parts = []
    for ch in range(N_CHANNELS):
        parts.append(mfcc_features(window[:, ch]))
        parts.append(tdd_features(window[:, ch]))
    return np.concatenate(parts).astype(np.float32)


class EmgInferenceNode(Node):
    def __init__(self):
        super().__init__("emg_inference_node")
        import os
        self.declare_parameter("port", "/dev/ttyACM0")
        self.declare_parameter("baud", 921600)
        # TWO model slots so the pipeline is runnable now and upgradable later:
        #   * model_path           = the current ENVELOPE classifier (works today,
        #                            but trained on ~1 participant -> ~chance acc).
        #   * retrained_model_path = drop your good model here once you have one.
        #   * use_retrained        = flip to True (or `use_retrained:=true`) to
        #                            switch over. Both must be 12-class / 76-feature
        #                            models trained with preprocess_v0_windowed.py.
        self.declare_parameter("model_path",
                               os.path.expanduser("~/emg_ws/emg_ml/models/classical_v0w_12g.joblib"))
        self.declare_parameter("retrained_model_path", "")
        self.declare_parameter("use_retrained", False)
        self.declare_parameter("win_ms", 200)
        self.declare_parameter("hop_ms", 100)
        self.declare_parameter("vote_window", 5)
        self.declare_parameter("rest_confidence", 0.60)
        # Live causal mains notch (Hz). MUST match the --mains the model was
        # trained with. The CRNN + classical_6g models are trained with 50.
        self.declare_parameter("mains", 50)
        # playback: if replay_csv is set, stream a recorded CSV instead of opening serial
        self.declare_parameter("replay_csv", "")
        self.declare_parameter("replay_speed", 1.0)
        self.declare_parameter("replay_gesture", -1)   # gesture_id filter, -1 = all
        self.declare_parameter("replay_cycle", -1)     # cycle filter, -1 = all
        self.declare_parameter("replay_loop", False)

        self.port = self.get_parameter("port").value
        self.baud = int(self.get_parameter("baud").value)
        # pick the active model slot
        use_retrained = bool(self.get_parameter("use_retrained").value)
        retrained = str(self.get_parameter("retrained_model_path").value).strip()
        envelope_model = self.get_parameter("model_path").value
        if use_retrained and retrained and os.path.exists(os.path.expanduser(retrained)):
            model_path = os.path.expanduser(retrained)
            self._model_slot = "retrained"
        else:
            if use_retrained:
                self.get_logger().warning(
                    f"use_retrained=True but retrained_model_path "
                    f"('{retrained}') is empty/missing — falling back to the "
                    f"envelope classifier.")
            model_path = os.path.expanduser(envelope_model)
            self._model_slot = "envelope"
        self.win = int(self.get_parameter("win_ms").value * FS / 1000)
        self.hop = int(self.get_parameter("hop_ms").value * FS / 1000)
        self.vote_window = int(self.get_parameter("vote_window").value)
        self.rest_conf = float(self.get_parameter("rest_confidence").value)

        self.gesture_pub = self.create_publisher(String, "/detected_gesture", 10)
        self.result_pub = self.create_publisher(String, "/inference_result", 10)

        import joblib
        self.get_logger().info(f"Loading [{self._model_slot}] model: {model_path}")
        self.model = joblib.load(model_path)
        # CRNN (rcnn_model.RCNNInference) consumes the raw window via
        # predict_window; classical sklearn pipelines consume 76-D features.
        self.is_crnn = hasattr(self.model, "predict_window")
        self.has_proba = hasattr(self.model, "predict_proba")
        self.get_logger().info(
            f"Model type: {'CRNN (raw-window)' if self.is_crnn else 'classical (features)'}")

        # Live causal mains notch — applied per sample BEFORE the window buffer,
        # matching training order (mains -> window -> ac_couple).
        mains_hz = int(self.get_parameter("mains").value)
        self.notch = None
        if mains_hz:
            try:
                from mains_notch import MainsNotch
                self.notch = MainsNotch(N_CHANNELS, fs=FS, mains_hz=mains_hz)
                self.get_logger().info(f"Live mains notch ON at {mains_hz} Hz (+harmonics).")
            except Exception as e:
                self.get_logger().error(
                    f"mains notch FAILED to load ({e}). The CRNN/classical_6g models were "
                    f"trained WITH the {mains_hz} Hz notch, so running unfiltered will make "
                    f"predictions ~random. This usually means scipy is broken in this python "
                    f"(numpy 2.x vs scipy-for-numpy-1). Run the node from a numpy<2 venv with "
                    f"scipy+torch (see PI/ROS setup), or set mains:=0 only with a mains-off model.")
        if self._model_slot == "envelope":
            self.get_logger().warning(
                "Using the ENVELOPE classifier (v3_env + cheez_env, 500 Hz, 12 "
                "gestures). It is trained on ~1 participant and scores near CHANCE "
                "(~8% on 12 classes), so live predictions will be mostly random. "
                "The whole pipeline still runs end-to-end — once you retrain a good "
                "model, set retrained_model_path:=<path> use_retrained:=true.")
        else:
            self.get_logger().info("Using the RETRAINED model slot.")
        n_cls = getattr(self.model, "classes_", None)
        if n_cls is not None and len(n_cls) != len(GESTURE_NAMES):
            self.get_logger().warning(
                f"Model has {len(n_cls)} classes but GESTURE_NAMES has "
                f"{len(GESTURE_NAMES)} — labels may be offset until retrain.")

        self.buffer = collections.deque(maxlen=self.win)
        self.samples_since_hop = 0
        self.recent = collections.deque(maxlen=self.vote_window)
        self.current = "REST"

        # data source thread: serial (live ESP) or CSV replay (playback)
        self._stop = threading.Event()
        replay = self.get_parameter("replay_csv").value
        if replay:
            self._thread = threading.Thread(target=self._replay_loop, args=(replay,), daemon=True)
        else:
            self._thread = threading.Thread(target=self._serial_loop, daemon=True)
        self._thread.start()
        self.get_logger().info(
            f"emg_inference_node up. port={self.port} baud={self.baud} "
            f"win={self.win} hop={self.hop}. Publishing /detected_gesture + /inference_result.")

    def _serial_loop(self):
        import serial
        try:
            ser = serial.Serial(self.port, self.baud, timeout=2.0)
            time.sleep(0.1)
            ser.reset_input_buffer()
        except Exception as e:
            self.get_logger().error(
                f"Could not open {self.port}: {e}. "
                f"Node stays alive; you can still hand-publish /detected_gesture.")
            return
        self.get_logger().info(f"Serial open {self.port} @ {self.baud}")
        while not self._stop.is_set():
            try:
                line = ser.readline().decode("ascii", errors="ignore").strip()
            except Exception:
                continue
            if not line:
                continue
            parts = line.split(",")
            try:
                if len(parts) >= 11:                  # full 11-col collection CSV row
                    ch1 = float(parts[V3_ENV_COL])
                    ch2 = float(parts[CHEEZ_ENV_COL])
                elif len(parts) >= 2:                 # bare "v3_env,cheez_env" stream
                    ch1, ch2 = float(parts[0]), float(parts[1])
                else:
                    continue
            except (ValueError, IndexError):
                continue
            self._push(ch1, ch2)
        ser.close()

    def _replay_loop(self, path):
        """Stream a recorded raw CSV through the exact same pipeline as live serial.
        11-col CSV: participant,cycle,gesture_name,gesture_id,trial,sample_idx,
                    v3_env,cheez_raw,cheez_filt,cheez_env,wear  (model uses 6 & 9)."""
        import csv
        speed = float(self.get_parameter("replay_speed").value)
        gfilt = int(self.get_parameter("replay_gesture").value)
        cfilt = int(self.get_parameter("replay_cycle").value)
        loop = bool(self.get_parameter("replay_loop").value)
        samples = []
        try:
            with open(path) as f:
                for r in csv.reader(f):
                    if len(r) < 11 or not r[5].isdigit():
                        continue
                    if cfilt >= 0 and r[1] != str(cfilt):
                        continue
                    if gfilt >= 0 and r[3] != str(gfilt):
                        continue
                    try:
                        samples.append((float(r[V3_ENV_COL]), float(r[CHEEZ_ENV_COL])))
                    except ValueError:
                        continue
        except Exception as e:
            self.get_logger().error(f"Replay: cannot read {path}: {e}")
            return
        if not samples:
            self.get_logger().error("Replay: no samples matched filters.")
            return
        self.get_logger().info(
            f"REPLAY {len(samples)} samples from {path} at {speed}x "
            f"(gesture={gfilt} cycle={cfilt}). Streaming as if from the ESP.")
        while not self._stop.is_set():
            t0 = time.time()
            for n, (a, b) in enumerate(samples, 1):
                if self._stop.is_set():
                    break
                self._push(a, b)
                if n % self.hop == 0:                       # pace to ~realtime/speed
                    dt = t0 + n / (FS * speed) - time.time()
                    if dt > 0:
                        time.sleep(dt)
            if not loop:
                break
        self.get_logger().info("Replay finished.")

    def _push(self, ch1, ch2):
        if self.notch is not None:
            ch1, ch2 = self.notch.process([ch1, ch2])   # causal mains notch
        self.buffer.append([ch1, ch2])
        self.samples_since_hop += 1
        if len(self.buffer) < self.win:
            return
        if self.samples_since_hop >= self.hop:
            self.samples_since_hop = 0
            self._predict()

    def _predict(self):
        window = preprocess_window(np.array(self.buffer, dtype=np.float32))

        # CRNN consumes the raw window and returns a 12-wide, gesture-id-indexed
        # probability vector. Classical consumes 76-D features; its predict_proba
        # is ordered by model.classes_, so scatter it into the same 12-wide
        # id-indexed layout. Either way `proba` is indexed by gesture id (idx 0 ==
        # REST), so the REST gate and GESTURE_NAMES labels stay correct.
        if self.is_crnn:
            proba = self.model.predict_proba_window(window)   # (12,) id-indexed
            pred_id = int(np.argmax(proba))
        else:
            feat = extract_features(window).reshape(1, -1)
            pred_id = int(self.model.predict(feat)[0])
            proba = None
            if self.has_proba:
                raw = self.model.predict_proba(feat)[0]
                proba = np.zeros(len(GESTURE_NAMES), dtype=float)
                proba[self.model.classes_] = raw

        confidence = 1.0
        probs = None
        if proba is not None:
            confidence = float(proba[pred_id])
            probs = [round(float(p), 3) for p in proba]
            if proba[0] > self.rest_conf:   # REST gate
                pred_id = 0
                confidence = float(proba[0])

        self.recent.append(pred_id)
        voted_id = collections.Counter(self.recent).most_common(1)[0][0]
        voted = GESTURE_NAMES[voted_id] if voted_id < len(GESTURE_NAMES) else f"class_{voted_id}"

        # always publish the per-hop result (for confidence-gating + debugging)
        self.result_pub.publish(String(data=json.dumps(
            {"gesture": voted, "confidence": round(confidence, 3), "probs": probs})))

        # publish stable gesture only when it changes
        if voted != self.current:
            self.current = voted
            self.gesture_pub.publish(String(data=voted))
            self.get_logger().info(f"gesture -> {voted}  (conf {confidence:.2f})")

    def destroy_node(self):
        self._stop.set()
        super().destroy_node()


def main():
    rclpy.init()
    node = EmgInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
