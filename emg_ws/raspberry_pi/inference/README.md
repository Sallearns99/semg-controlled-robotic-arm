# EMG inference on the Raspberry Pi

This folder is the **classifier half** of the EMG arm: it reads the ESP32 stream
(**USB serial _or_ WiFi/TCP**), classifies the current gesture every hop, and
prints the gesture name. Pipe that into `../arm_control/emg_arm_control.py` to
drive the real arm.

> These files are **copies** of the source pipeline in
> `moving to ubuntu (2)/moving to ubuntu/code/`. If you retrain a model or edit
> the pipeline there, re-copy them. The model `.joblib`, `rcnn_model.py`,
> `mains_notch.py`, `preprocess_v0_windowed.py`, `emg_config.py` must stay
> version-matched (the models were trained **with the 50 Hz notch**, so inference
> must run **`--mains 50`**, which is the default).

## Contents
| File | Role |
|---|---|
| `inference_rt.py` | serial/WiFi → 50 Hz notch → 200 ms window → predict → vote(5) → REST-gate → print |
| `mains_notch.py` | **causal live 50 Hz notch** (needs scipy); matches the filter the models trained with |
| `rcnn_model.py` | **required to load the CRNN** (`joblib.load` rebuilds this class) |
| `preprocess_v0_windowed.py` | `ac_couple` (+ `extract_features` for the classical model) |
| `emg_config.py` | gesture list, channels, CSV layout |
| `models/rcnn_best_model.joblib` | the recurrent-CNN (needs torch, **not** librosa) |
| `models/classical_6g_best_model.joblib` | classical fallback (needs scikit-learn + librosa) |

## Install (once, on the Pi)
```bash
cd ~/…/raspberry_pi/inference
python3 -m pip install -r requirements-inference.txt   # CRNN deps (torch + scipy)
```
`torch` downloads the aarch64 CPU wheel from PyPI (~80 MB). No GPU/CUDA on the Pi.

## Run — USB serial
```bash
python3 inference_rt.py --port /dev/ttyACM0 --model models/rcnn_best_model.joblib --mains 50 \
  | python3 ../arm_control/emg_arm_control.py --stdin --vote-window 3.0
```

## Run — WiFi (ESP on the same router, STA mode)
Flash `semg_stream_live_wifi.ino` (in the source `code/` folder); it prints its IP
over USB on boot. Then on the Pi use `--host` instead of `--port`:
```bash
python3 inference_rt.py --host <esp-ip> --model models/rcnn_best_model.joblib --mains 50 \
  | python3 ../arm_control/emg_arm_control.py --stdin --vote-window 3.0
# --host esp32.local also works if mDNS resolves on your network
```

- `--mains 50` — live causal 50 Hz mains notch (+harmonics). **Leave it on**: the models
  were trained with it, and it measurably improved accuracy. `--mains off` only if you
  retrain the model with mains off too.
- `--vote-window 3.0` debounces the classifier (decide once every 3 s). Drop it for continuous jog.

Inference only, to watch predictions:
```bash
python3 inference_rt.py --host <esp-ip> --model models/rcnn_best_model.joblib --mains 50 --verbose
python3 inference_rt.py --list          # show serial ports
```

> ⚠️ Do **not** point inference at the old `emg_ml/wifi/emg_stream_wifi.ino` — it streams
> 2000 Hz raw ADC, the wrong rate and signal type. Use `semg_stream_live_wifi.ino`.

## CRNN vs classical on the Pi
- **CRNN** (`models/rcnn_best_model.joblib`): needs **torch**, not librosa. Raw-window model.
- **Classical** (`models/classical_6g_best_model.joblib`): needs **scikit-learn + librosa**
  (numba) instead of torch. Same `--model` flag; `inference_rt.py` auto-detects which is loaded.

Both were trained on the 7-class subset (REST + EXTENSION/FLEXION +
DOWNWARD_ROT_CW/CCW + EXTEND/RETRACT_ELBOW); they output the original 12 gesture
ids, so the arm map in `emg_arm_control.py` needs no changes.
