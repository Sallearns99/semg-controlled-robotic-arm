# EMG ML assets — inventory & integration notes

Source: `~/Downloads/ESPcodes/` (everything generated on the Windows side).
Staged here (`~/emg_ws/emg_ml/`) is the **lean subset we need** — models + code + docs,
**not** the ~900 MB of training data (raw CSVs / npy / npz / spectrograms stayed in Downloads).

## The system in one line
ESP32-S3 streams **2-channel raw EMG @ 2000 Hz** over USB serial → a PC does
**windowed feature extraction + classification** → a gesture label. Inference is on the
**PC, not the ESP**. There is no `inference.ino`; the ESP only streams.

## Signal chain & data format
- **Sensors:** 2× Muscle Sensor v3 (Ch1 = extensor / "Muscle A", Ch2 = flexor / "Muscle B").
- **ESP firmware `emg_stream.ino`** (the one we use for realtime): emits ASCII lines
  `ch1,ch2\n`, raw 12-bit ADC counts (0–4095), at **2000 Hz**, **baud 921600**.
  Pins: Ch1=GPIO3, Ch2=GPIO8 (the `#define`s; header comment says GPIO4/5 — trust the `#define`s).
  Has a commented-out WiFi/TCP path (port 5005) for later wireless.
- **`emg_collection.ino` / `semg_data_collection.ino`:** the *other* firmware — buffers
  12000 samples/trial and dumps; used for dataset collection, NOT realtime.
- **Recording params** (raw_meta_example.json): 2000 Hz, 12000 samples/trial (6 s),
  5 cycles, 10 gestures.

## The 10 gestures (label_map.json — index → name)
`0 REST · 1 EXTENSION · 2 FLEXION · 3 ULNAR_DEV · 4 RADIAL_DEV · 5 GRIP ·
6 ABDUCTION · 7 ADDUCTION · 8 SUPINATION · 9 PRONATION`

## Preprocessing / features (must match between training & inference)
Windowing (windowing_config.json): **200 ms window (400 samples), 100 ms hop (200 samples)**.
Per window: median-subtract (AC-couple) → per channel **MFCC stats (8 mfcc × mean/std/max/min = 32)**
+ **TDD (6: rms², sparsity, irregularity, rms, waveform-length, zero-crossings)** → 38/ch ×2 =
**76 features**. Implemented in `reference/preprocess_v0_windowed.py` and mirrored in `inference_rt.py`.

## Models & accuracy — ⚠️ all at/near chance (10 classes → 10% chance)
| Model | File | Test acc |
|---|---|---|
| Classical RF (windowed) | `models/best_model.joblib` (sklearn Pipeline, 76-feat, has predict_proba) | **9.5%** |
| Deep V4 | `models/v4_best.pt` (620k params) | 20% ungated / 10% gated |
| Deep TB8 | `models/tb8_fold1_best.pt` | 16% val |
V4 history shows train acc ~45% while val stuck ~12% → **severe overfit / no generalization**
(needs more/better data — an ML problem, not a plumbing problem). The user knows accuracy is
low; **right now we build the pipeline, not chase accuracy.** `inference_rt.py` loads the
**joblib** model (verified: loads, 76 feats, predicts) — that's our runtime model for now.

## `inference_rt.py` — the proven realtime pipeline (reference/)
Serial reader thread → circular 200 ms buffer → every 100 ms hop: AC-couple → 76 feats →
`model.predict` → **majority vote over last 5** → **REST gate** (if P(REST) > 0.60 force REST)
→ prints stable gesture. Drops samples if the queue backs up to stay realtime. Baud 921600.
**This is the code we port into a ROS node.**

## Prior ROS nodes (reference/prior_inference_node*.py)
From an earlier 4-channel/ONNX iteration. Useful for **topic naming convention only**:
subscribes `/emg_raw` (Float32MultiArray), publishes `/detected_gesture` (String) +
`/inference_result` (String JSON: gesture+confidence+probs). They hard-code a venv
(`/home/salma/emg_env/...`) and assume 4 channels — outdated, don't run as-is.

## Environment — good news
All heavy deps are already importable in the **ROS Jazzy system python3**: numpy 2.4, scipy,
sklearn 1.8, joblib, **librosa 0.11**, pyserial 3.5, **torch 2.10+cu128**. Only `pandas`
missing (not needed for inference). So **no venv needed** — the ROS nodes can import these directly.

## What we'll build (next)
1. `emg_inference_node` (ROS 2): port `inference_rt.py` — read `/dev/ttyACM0` @921600,
   window+features+joblib predict+vote+REST-gate, publish `/detected_gesture` (String) and
   `/inference_result` (JSON gesture+confidence). Reuse prior topic names.
2. `emg_bridge` (ROS 2): subscribe `/detected_gesture`, debounce/confidence-gate, map each
   gesture → arm/gripper **jog** (reuse teleop_keyboard's incremental-target logic) →
   `/arm_controller/joint_trajectory` + `/gripper_controller/joint_trajectory`.
3. Map 10 gestures → arm actions (TBD with user). REST = deadman/no-op.

## Staged files (this folder)
- `models/` — best_model.joblib, v4_best.pt, tb8_fold1_best.pt
- `reference/` — inference_rt.py, preprocess_v0_windowed.py, train_emg_{model,v2,v4}.py,
  prior_inference_node*.py, `firmware/`{emg_stream,emg_collection,adc_diagnostic}.ino
- `docs/` — label_map.json, requirements.txt, windowing_config.json, raw_meta_example.json,
  results/ (results.json, v4_results.json, a confusion png)
