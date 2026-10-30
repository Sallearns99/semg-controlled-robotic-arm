# EMG arm on the Raspberry Pi 4B — copy-paste setup

Everything to get the EMG classifier + arm running on a **Raspberry Pi 4 Model B,
64-bit Ubuntu 24.04**. Commands are meant to be pasted straight into a Pi terminal.

The system is two halves that pipe together:

```
ESP32 (USB serial) ──► inference/inference_rt.py ──gesture name──► arm_control/emg_arm_control.py ──► servos
        gesture CLASSIFIER (this needs torch)              ARM driver (PCA9685)
```

---

## 0. What to copy onto the Pi

Copy the whole `raspberry_pi/` folder. It contains:

```
raspberry_pi/
├── arm_control/        # servo half (PCA9685)
│   ├── emg_arm_control.py   arm_driver.py   servo_config.py
│   ├── servo_calibrate.py   arm_keyboard_teleop.py
│   └── requirements.txt
└── inference/          # classifier half
    ├── inference_rt.py  rcnn_model.py  preprocess_v0_windowed.py  emg_config.py
    ├── models/rcnn_best_model.joblib            # CRNN  (needs torch)
    ├── models/classical_6g_best_model.joblib    # classical (needs sklearn+librosa)
    └── requirements-inference.txt
```

From your **dev PC**, send it over (replace `<pi-ip>`):

```bash
scp -r ~/emg_ws/raspberry_pi <pi-user>@<pi-ip>:~/emg/
```

Then on the **Pi**, everything below assumes:

```bash
cd ~/emg/raspberry_pi
```

> ⚠️ The 5 files in `inference/` are **copies** of the training pipeline. If you
> retrain a model on the PC, re-copy the model `.joblib` **and** `rcnn_model.py`,
> `preprocess_v0_windowed.py`, `emg_config.py` — they must stay version-matched.

---

## 1. System packages (Pi)

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv i2c-tools
```

Enable I2C for the PCA9685 servo board. On **Ubuntu** (not Raspberry Pi OS, so no
`raspi-config`) add the overlay to the firmware config and reboot:

```bash
# append once, only if not already present
grep -q '^dtparam=i2c_arm=on' /boot/firmware/config.txt || \
  echo 'dtparam=i2c_arm=on' | sudo tee -a /boot/firmware/config.txt
echo 'i2c-dev' | sudo tee /etc/modules-load.d/i2c.conf   # load the i2c-dev module at boot
sudo usermod -aG i2c,dialout $USER                        # I2C + serial access
sudo reboot                                               # applies overlay + groups
```

After reboot, confirm the board is seen (address `0x40`) and groups applied:

```bash
groups                  # expect 'i2c' and 'dialout' listed
i2cdetect -y 1          # expect '40' in the grid (the PCA9685)
```

---

## 2. Python deps

Use a venv so you don't fight the system Python.

```bash
cd ~/emg/raspberry_pi
python3 -m venv ~/emg_venv
source ~/emg_venv/bin/activate          # prompt shows (emg_venv)
```

### Classifier (CRNN — recommended on the Pi)
```bash
pip install -r inference/requirements-inference.txt
```
This pulls `torch` (aarch64 CPU wheel from PyPI, ~80 MB) + numpy/pandas/joblib/pyserial.
**No librosa/numba needed for the CRNN.** First import of torch is slow; that's normal.

### Servo half
```bash
pip install -r arm_control/requirements.txt   # adafruit-circuitpython-servokit
```

Quick sanity check that the model loads:
```bash
cd ~/emg/raspberry_pi/inference
python3 -c "import joblib; m=joblib.load('models/rcnn_best_model.joblib'); print('CRNN OK:', type(m).__name__)"
```

---

## 3. Connect to the ESP32 — USB or WiFi

**The models were trained WITH a 50 Hz mains notch, so always run `--mains 50`** (the default).

### USB serial
```bash
cd ~/emg/raspberry_pi/inference
python3 inference_rt.py --list          # lists serial ports
ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null
```
Usually `/dev/ttyACM0`. Flash `semg_stream_live.ino` on the ESP.

### WiFi (recommended for a wireless arm)
Flash **`semg_stream_live_wifi.ino`** (STA mode — edit the SSID/password to your
2.4 GHz router). On boot it prints its IP over USB, e.g. `WiFi IP: 192.168.1.50`.
The Pi must be on the **same router**. Then connect with `--host` instead of `--port`:
```bash
python3 inference_rt.py --host 192.168.1.50 --model models/rcnn_best_model.joblib --mains 50 --verbose
# --host esp32.local also works if mDNS resolves on your network
```
> ⚠️ Do NOT use the old `emg_ml/wifi/emg_stream_wifi.ino` — it streams 2000 Hz raw ADC
> (wrong rate + wrong signal); the models would get garbage. Use `semg_stream_live_wifi.ino`.

---

## 4. Run it

### A) Watch predictions only (no arm) — start here
```bash
cd ~/emg/raspberry_pi/inference
# USB:
python3 inference_rt.py --port /dev/ttyACM0 --model models/rcnn_best_model.joblib --mains 50 --verbose
# WiFi:
python3 inference_rt.py --host 192.168.1.50 --model models/rcnn_best_model.joblib --mains 50 --verbose
```
You should see gesture lines and, with `--verbose`, the top-2 probabilities. Ctrl-C to stop.

### B) Dry-run the arm mapping (no servos move)
```bash
cd ~/emg/raspberry_pi/arm_control
python3 emg_arm_control.py --demo --dry-run     # prints which joint each gesture would drive
```

### C) Full pipeline — classify AND move the real arm
```bash
cd ~/emg/raspberry_pi/inference
# USB:
python3 inference_rt.py --port /dev/ttyACM0 --model models/rcnn_best_model.joblib --mains 50 \
  | python3 ../arm_control/emg_arm_control.py --stdin --vote-window 3.0
# WiFi:
python3 inference_rt.py --host 192.168.1.50 --model models/rcnn_best_model.joblib --mains 50 \
  | python3 ../arm_control/emg_arm_control.py --stdin --vote-window 3.0
```
- `--mains 50` = live causal 50 Hz mains notch (+harmonics). **Keep it on** (matches training).
- `--vote-window 3.0` = decide once every 3 s and move a fixed step (debounces a noisy
  classifier — **recommended**). Drop it for continuous jog while a gesture is held.
- Add `--dry-run` to the `emg_arm_control.py` end to test the wiring without moving servos.

> Both halves must use the **same venv**. If you run them in two terminals, `source
> ~/emg_venv/bin/activate` in each. In the single piped command above, one activation covers both.

---

## 5. Gesture → joint map (built in, no config needed)

| Gesture | Action |
|---|---|
| EXTEND_ELBOW / RETRACT_ELBOW | j2 +/− |
| EXTENSION / FLEXION | j3 +/− |
| DOWNWARD_ROT_CW / CCW | j1 −/+ (base) |
| GRIP | toggle gripper open/closed |
| REST / anything else | HOLD |

The models were trained on a 7-class subset but still output the original 12 gesture
ids, so this map needs no changes. (FRONT_ROT / ULNAR/RADIAL are mapped too but those
classes aren't in the current models, so they won't fire.)

---

## 6. Swap to the classical model (optional)

`inference_rt.py` auto-detects the model type from the `--model` you pass. The classical
model needs **scikit-learn + librosa** instead of torch:

```bash
source ~/emg_venv/bin/activate
pip install scikit-learn librosa        # librosa pulls numba — slower to install on the Pi
python3 inference_rt.py --port /dev/ttyACM0 --model models/classical_6g_best_model.joblib --mains 50 --verbose
```

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'rcnn_model'` | Run `inference_rt.py` from inside `inference/` (so `rcnn_model.py` is importable), or keep the `.py` files together. |
| `ModuleNotFoundError: scipy` | The live notch needs scipy: `pip install scipy` (or `--mains off`, but then use a model trained mains-off). |
| `could not open port /dev/ttyACM0` / permission denied | You're not in `dialout` yet — `sudo usermod -aG dialout $USER`, then log out/in. Check with `--list`. |
| WiFi: `Connection refused` / can't reach `--host` | ESP and Pi must be on the **same 2.4 GHz router**. Use the IP the ESP prints over USB; ping it first. mDNS `esp32.local` may not resolve — use the raw IP. |
| WiFi: connects but `[no data for 2 s]` | Check the ESP serial monitor shows `TCP client connected`; only one client at a time. Re-flash `semg_stream_live_wifi.ino` with your SSID/password. |
| `i2cdetect` shows nothing / no `40` | I2C not enabled or board unpowered. Re-check the `dtparam=i2c_arm=on` line + reboot, wiring, and the **separate 5–6 V servo supply** (NOT the Pi 5 V rail). |
| torch install fails / very slow | Make sure you're on 64-bit Ubuntu (`uname -m` → `aarch64`) and inside the venv. The wheel is ~80 MB; let it finish. |
| Arm jitters / wrong direction | Use `--vote-window 3.0`; flip a joint sign in `arm_control/servo_config.py`; re-run `servo_calibrate.py`. |

---

## 8. One-shot recap (paste after first-time setup is done)

```bash
source ~/emg_venv/bin/activate
cd ~/emg/raspberry_pi/inference
# USB:
python3 inference_rt.py --port /dev/ttyACM0 --model models/rcnn_best_model.joblib --mains 50 \
  | python3 ../arm_control/emg_arm_control.py --stdin --vote-window 3.0
# WiFi (ESP IP from its boot message):
python3 inference_rt.py --host 192.168.1.50 --model models/rcnn_best_model.joblib --mains 50 \
  | python3 ../arm_control/emg_arm_control.py --stdin --vote-window 3.0
```
