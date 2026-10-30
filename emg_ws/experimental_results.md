# Experimental Results & Implementation Record — RP06 sEMG-Controlled Robotic Arm

> **Project:** RP06 — *Design and Implementation of an sEMG-Based Gesture-Controlled
> Robotic Arm Using Deep Learning.*
> **Student:** Salma Farid. **Advisors:** Dr. Nourhan Zayed & Dr. Mostafa
> Abdelaziz. **Institution:** British University in Egypt (BUE), Robotics Engineering —
> final-year capstone.
>
> **Purpose of this document.** A comprehensive, source-of-truth briefing for the
> **Experimental Work / Machine Learning / ROS 2 Integration / Results** chapters,
> drafted from *all* project documentation (control report, project memory, RUNBOOK,
> setup guides) plus the latest implementation work. It is written to be handed to an
> LLM (Claude Sonnet 4.6) to draft the formal report chapters. It deliberately
> **foregrounds material not covered in the existing reports** — the EMG dataset,
> gesture-selection methodology, model training/results, the mains-filtering result,
> wireless inference, and the latest motion-control and deployment engineering — and
> **flags corrections/updates** to facts stated in earlier chapters (see §0).
> Numbers are *measured*; methodology labels are explicit so nothing is overstated.

---

## 0. Updates & corrections to prior reports (read first)

These changed since the earlier chapters / `raspberry_pi/CONTROL_REPORT.md` were written.
The report author should treat the items below as authoritative.

1. **Base joint (`j1`) is now a STEPPER, not a servo.** The base/turntable is driven by
   a **DRV8825 stepper-driver module** (step/dir, bipolar, microstepping), not an MG995
   on PCA9685 channel 0 as the control report states. The motor is unconfirmed but is
   **almost certainly a NEMA 17 bipolar stepper** (1.8°/step, 200 steps/rev) — the
   standard DRV8825 pairing *(confirm on the bench)*. Implication: the base axis uses a
   fundamentally different actuation path (timed GPIO step pulses, no absolute feedback)
   than the PWM servos; in the Gazebo simulation all joints are treated uniformly as
   position-controlled, so bridging the base command to DRV8825 pulses on the real arm
   is an open integration point.
2. **EMG acquisition is now 500 Hz with a CHEEZ.semg second channel** (not 2 kHz / dual
   Muscle-Sensor-v3 as older chapters say). Channels: `v3_env` (Muscle Sensor v3
   hardware envelope, GPIO3 via 56k/22k divider) + `cheez_env` (CHEEZ.semg on-device
   envelope, GPIO7). Corrected electrode placement: **CHEEZ = extensor, v3 = flexor.**
3. **Servo model is MG995** (1.47 N·m stall, 8.05 rad/s @ 6 V) — confirmed; earlier
   "MG996R" mentions are stale.
4. **Software stack is ROS 2 Jazzy / Ubuntu 24.04, Gazebo Harmonic (gz sim 8)** — not
   Humble/22.04.
5. **The deployed gesture model now works end-to-end and at meaningfully above-chance
   accuracy** with the new 6-gesture set + 50 Hz notch (older reports state "classifier
   not yet accurate / awaiting better data"). See §6.
6. **j2 dual-servo synchronization** is *mirrored opposite-sign equal-gain angle
   commands* (open-loop feed-forward), **not** a PWM low-pass filter / feedback loop —
   the old "PWM filtering" account is inaccurate.

---

## 1. System overview

Two-phase capstone:
- **Phase 1:** a custom **6-DOF** arm (5 revolute arm joints + a gripper) driven in
  ROS 2 + Gazebo simulation and on real hardware (Raspberry Pi). Detailed in
  `raspberry_pi/CONTROL_REPORT.md` (control), and the Mechanical/Electronic chapters.
- **Phase 2 (focus of this document):** classify forearm sEMG gestures in real time
  and drive the arm — both the Gazebo simulation and the physical arm — **wirelessly**.

End-to-end loop achieved:
```
forearm muscles → ESP32-S3 (2 EMG channels, 500 Hz)
   → WiFi/TCP (or USB) → feature extraction + classifier (Pi or PC)
   → gesture name → motion bridge (3 s vote / debounce, one-joint-at-a-time)
   → arm joints  (Gazebo simulation AND real servos + stepper base)
```

**ROS packages:** `rotatedauto` (description + Gazebo bring-up + teleop),
`rotatedauto_moveit_config` (MoveIt 2), `emg_control` (EMG inference + gesture→arm bridge).

---

## 2. Mechanical & actuation hardware (summary; full detail in the Mechanical/Control chapters)

- **6-DOF arm**, designed in SolidWorks, FDM-printed in ABS. Mass **2.41 kg**, height
  **670.24 mm**, base Ø **150 mm**, 8 links (`base_link, l1–l5, l6l, l6r`), 6 joints.
- **Actuators:** **7 × MG995** metal-gear servos via a **PCA9685** 16-channel I²C PWM
  driver (addr 0x40, 50 Hz, ~500–2500 µs). The **shoulder `j2` uses two mirrored
  servos** (load sharing; the arm is top-heavy, ~1.37 kg above j2; combined ≈2.94 N·m).
  The **gripper** is one driven finger (`j6l`) + a passive **mimic** follower (`j6r`,
  ratio −2.0885); +0.288 rad = closed, −0.51 rad = open.
- **Base `j1` (UPDATED):** continuous turntable now driven by a **DRV8825 stepper
  driver** (+ stepper motor, likely NEMA 17) instead of a servo — see §0.1.
- **Power:** servos/stepper on a **dedicated 5–6 V ≥ 5 A** supply (never the Pi rail),
  common ground with the Pi/PCA9685. Brownout from an under-rated source is a known
  failure mode (servos "hold but barely move", PCA9685 can reset).

---

## 3. EMG sensing and the dataset

### 3.1 Sensor chain (current design)
- **Channel 1 — `v3_env`:** Muscle Sensor v3 (Advancer Technologies), hardware-rectified
  **envelope**, ADC on GPIO3 (ADC1) via a 56 kΩ/22 kΩ divider (≈0.282× scaling); rests
  near 0.
- **Channel 2 — `cheez_env`:** CHEEZ.semg, on-device bandpass + **envelope** (CheezsEMG
  library) at 500 Hz, signal on GPIO7, digital wear-detect on GPIO6.
- **MCU:** ESP32-S3-N16R8. Streams `v3_env,cheez_env` at **500 Hz**, 921600 baud (USB)
  or WiFi/TCP. ADC: 12-bit, ADC_11db, micros()-paced.

### 3.2 Sensor evolution (context for the report)
The design moved from two Muscle Sensor v3 boards (an earlier 2 kHz, raw-ADC, dual-flexor/
extensor scheme) to **v3 + CHEEZ.semg** to obtain a cleaner on-device envelope and reduce
3× over-weighting of one sensor. Acquisition rate dropped to the CHEEZ-native **500 Hz**.

### 3.3 Dataset
- Recording `P1_20260606_231716`: single participant (P1), **12 gestures × 5 cycles**,
  **5 s/trial = 2500 samples/trial** at 500 Hz. The cleanest session; used for all model
  work here.
- Original 12 gestures (ids): 0 REST, 1 EXTENSION, 2 FLEXION, 3 ULNAR_DEV, 4 RADIAL_DEV,
  5 GRIP, 6 DOWNWARD_ROT_CW, 7 DOWNWARD_ROT_CCW, 8 FRONT_ROT_CW, 9 FRONT_ROT_CCW,
  10 EXTEND_ELBOW, 11 RETRACT_ELBOW.

### 3.4 Two hardware-quality findings that bound the results (important caveats)
- **CHEEZ channel is effectively DEAD in every recording.** `cheez_raw` should rest near
  the ADC mid-rail (~2048); measured REST medians were ~40–120 (pin near ground), so its
  envelope has almost no signal. Diagnostics: `cheez_only` features ≈ chance; `v3_only` ≈
  as good as both channels. **The system is, in practice, a single usable EMG channel
  (v3_env)** — a hardware/wiring fault, not software, and a hard ceiling on accuracy.
- **`v3_env` is contaminated by 50 Hz mains.** The v3 hardware envelope does not suppress
  line interference and the firmware did no filtering, so a 50 Hz peak sits *above* the
  EMG band in v3_env's spectrum. Egypt mains = 50 Hz. (Addressed in §5.)

---

## 4. Feature & model pipeline

- **Windowing:** 200 ms window (100 samples), 100 ms hop (50 samples), 50 ms edge drop.
  Per-window **AC-couple** (per-channel median subtraction) — the only normalization the
  live system can reproduce (it sees one window at a time) and it preserves amplitude
  (the main discriminator). 240 windows/class.
- **Classical features (76-D):** per channel, MFCC statistics (n_mfcc=8, n_fft=32,
  win=32, hop=16 → mean/std/max/min) + 6 time-domain descriptors (RMS², sparseness,
  irregularity, RMS, waveform length, zero-crossings) → `(8×4 + 6) × 2 = 76`.
- **Classical models:** SVM (RBF) / Random Forest / Gradient Boosting (+ optional XGB),
  `StandardScaler` pipeline, `joblib`-serialized; cycle-wise split (train cy 1–3, val 4,
  test 5).
- **Deep model — recurrent CNN (CRNN), new this work:** a *genuine* CRNN on the **raw
  window** (100×2), not the summary features:
  `Conv1d×3 (k=5, BN, ReLU, MaxPool) → bidirectional GRU → temporal mean-pool → FC`,
  **99,500 parameters**; per-channel standardization fit on the training windows;
  cross-entropy + class weights + label smoothing, AdamW + cosine schedule.

### 4.1 Real-time inference conformance (engineering contribution)
The real-time loop loads **one** model and calls `predict`/`predict_proba` on the live
window. To make the torch CRNN a *drop-in* for the sklearn-style loop:
- A wrapper (`RCNNInference`) exposes `predict_window`/`predict_proba_window` and returns
  a **full 12-wide, gesture-id-indexed** probability vector (zeros for absent classes);
  it serializes via `joblib` (state-dict only → portable across machines/torch versions).
- The inference loop auto-detects the model type and, for the classical path, scatters
  `predict_proba` into the same 12-wide id-indexed layout via `model.classes_`, keeping
  the REST gate (`proba[0]`) and gesture labels correct for *any* class subset.

---

## 5. The 50 Hz mains notch (key experimental result)

The user requested mains filtering "while reading the person," which produced one of the
more interesting findings.

- **Diagnosis:** v3_env carries a strong, roughly stationary 50 Hz pedestal (+ harmonics)
  — measured *above* the EMG band in the spectrum.
- **Live implementation (`mains_notch.py`):** a **causal streaming notch** —
  `scipy.signal.iirnotch` + `lfilter` with persistent per-channel state (seeded to the
  first sample to avoid startup ringing) — at 50 Hz **plus all harmonics below Nyquist
  (100/150/200 Hz)**, on both channels, applied **per incoming sample before windowing**
  (training order: mains → window → AC-couple).
- **Quantitative discovery (corrects an earlier assumption):** mains filtering changes
  the 76-D feature vector by **~71 %** (relative L2). It is **NOT** "absorbed" by window
  normalization, so a model trained *without* the notch but run *with* a live notch sees
  ~71 %-shifted inputs and predicts near-randomly → **the live notch forces retraining
  with a matching notch.**
- **Causal-vs-zero-phase consistency:** offline training conventionally uses zero-phase
  `filtfilt` (= |H|², a deeper notch) which a real-time stream cannot reproduce. To
  guarantee *train == inference*, training was switched to the **same causal notch**.
  Verified: causal-vs-filtfilt features differ ~5 % (median); causal-train (per-trial) ≈
  causal-live (continuous).
- **Impact:** the notch **improved** accuracy on the selected set (see §6) — a genuine,
  repeatable result, not just hygiene.

---

## 6. Gesture-subset selection and model results

### 6.1 Selecting 6 gestures + REST — methodology and a corrected statistical pitfall
**Goal:** pick 6 strong gestures — 3 antagonistic pairs + REST — for a usable control set.
Per-class metrics were not stored, so they were computed, going through an important
correction worth highlighting:

1. **First estimate (rejected — leakage):** stratified shuffle 5-fold CV gave RF ≈
   **57.7 %**. This is **optimistically biased**: 200 ms windows overlap 50 % (100 ms
   hop), so a window and its neighbour (sharing half their samples) land in train vs.
   test — the model partly *memorises trials*. It measures within-session separability,
   not generalization.
2. **Honest estimate (used for selection):** **leave-one-cycle-out (LOCO)** — hold out a
   whole recording cycle at a time. Leakage-free. Result **~12–18 %** (chance 8.3 % /
   12 classes). The 7× gap is the signature of **cross-cycle drift**.

**Honest headline:** within a session the gestures are separable; across cycles they
transfer poorly on this single-participant, one-usable-channel data. Under the
leakage-free split, only **REST and the elbow pair** were repeatably above chance.

### 6.2 Selected set (7 classes, original ids kept)
| id | gesture | pair / axis | cross-cycle signal |
|----|---------|-------------|--------------------|
| 0  | REST              | baseline             | strong |
| 10 | EXTEND_ELBOW      | elbow                | **strong — both members reliable** |
| 11 | RETRACT_ELBOW     | elbow                | **strong** |
| 1  | EXTENSION         | wrist                | moderate |
| 2  | FLEXION           | wrist                | weak |
| 6  | DOWNWARD_ROT_CW   | forearm rotation     | weak |
| 7  | DOWNWARD_ROT_CCW  | forearm rotation     | moderate |

Dropped: GRIP (unpaired), ULNAR/RADIAL_DEV, FRONT_ROT_CW/CCW (≈ chance cross-cycle).

> **Design trick:** keeping the **original gesture ids** (no remapping) plus a fixed
> 12-wide model head means dropping classes leaves the entire pipeline (preprocess →
> train → real-time inference → ROS node) index-compatible with **zero code changes**,
> and REST stays the gated index 0.

### 6.3 Results — 6-gesture set (7 classes), mains OFF vs ON
Chance (7 classes) = **14.3 %**. "Test" = deployment-realistic cross-cycle (train cy 1–3,
test cy 5).

| Model | Mains 50 Hz | Val acc | Test acc | Notes |
|-------|-------------|---------|----------|-------|
| Classical (GBM) | off | 16.1 % | 25.0 % | best = GBM |
| **Classical (RF)** | **on (causal)** | **30.7 %** | **38.4 %** | best = RF; **DEPLOYED** |
| CRNN | off | 14.9 % | 31.8 % | val collapsed toward 0 (overfit) |
| CRNN | **on (causal)** | **34.8 %** | 32.7 % | val no longer collapses; REST recall 1.0 |

**Takeaways:**
- The 50 Hz causal notch raised the classical model **25 % → 38 %** test (~2.7× chance)
  and the CRNN's best validation ~15 % → ~35 %.
- The **classical RF is the deployed model** (smoother, higher test accuracy, lighter).
  The CRNN — though architecturally sound and a genuine recurrent CNN — **overfits**
  (train → 100 %, val low) on single-participant, drift-heavy, ~one-channel data,
  consistent with deep EMG models needing many subjects.
- Context numbers (state with labels): intrinsic mixed-cycle (leaky) RF ≈ 57.7 % shows
  the *signal exists*; LOCO ≈ 12–18 % quantifies the drift that limits real-world use.

### 6.4 Interpretation
The dominant limiter is **cross-session drift on a single participant with one functional
EMG channel**, not the model/features. The mains notch is a real improvement on top.

---

## 7. Real-time deployment

### 7.1 Wireless streaming (WiFi)
- **New firmware `semg_stream_live_wifi.ino`:** the exact 500 Hz CheezsEMG sampling of
  the USB streamer, over **WiFi/TCP** (ESP = TCP server on port 5005; batched ~10 ms
  flush so the 500 Hz sampler is never delayed). **AP mode** (ESP is its own hotspot at
  192.168.4.1 — most reliable) or **STA mode** (joins a router). Also mirrors to USB
  serial simultaneously, so a PC (sim) and the Pi (real arm) can both receive — one over
  USB, one over WiFi.
- **`inference_rt.py`** gained `--host`/`--tcp-port` (a pyserial-look-alike TCP adapter)
  and `--mains {off,50,60}` (default 50). pyserial was made **optional** (WiFi needs no
  serial); librosa import made lazy.
- **Wrong-firmware pitfall:** an older `emg_stream_wifi.ino` streams 2000 Hz **raw ADC**
  — wrong rate and signal type; pointing inference at it yields garbage. Use the 500 Hz
  envelope firmware.

### 7.2 Raspberry Pi (real arm)
- Pi 4 Model B, 64-bit Ubuntu 24.04. CRNN deploy needs **torch** (aarch64 CPU wheel);
  classical deploy needs **scikit-learn + librosa**. Inference files bundled under
  `raspberry_pi/inference/` (model + `rcnn_model.py` + `mains_notch.py` + scripts),
  since `joblib.load` of the CRNN requires `rcnn_model.py` importable.
- Servo driver requires the Pi GPIO lib; on kernel 6.x Ubuntu use **`rpi-lgpio`** (drop-in
  for `RPi.GPIO`).

### 7.3 Gazebo simulation (ROS 2)
- The ROS inference node was adapted with the same CRNN branch + live mains notch; the
  config/launch default to the **deployed classical RF model** (CRNN tried first but was
  jerky/over-eager → rolled back; both swappable by parameter).
- The bridge maps gestures to joints: EXTENSION/FLEXION → wrist; EXTEND/RETRACT_ELBOW →
  elbow; DOWNWARD_ROT_CW/CCW → base; GRIP → gripper; REST/other → HOLD.

### 7.4 Motion control — discrete, debounced commands ("one joint per 3 s")
A noisy classifier drove the arm erratically (rapid gesture flips "compiled" into
simultaneous multi-joint motion). Solution = **vote mode**:
- Every **3 s**, take the **majority** gesture over the window and fire **one** move
  (`fire_step` rad), provided the winner is not REST and clears `min_vote`.
- **"One joint at a time" fix:** the bridge tracks the arm's **actual** joint positions
  continuously and, on each fire, **freezes every other joint at its current position**
  and moves only the selected joint (smooth `fire_time` ≈ 0.6 s). Result: exactly one
  deliberate joint motion per decision; anything still moving stops. (The real arm is
  inherently one-joint-per-command since each servo holds its last setpoint.)

---

## 8. Challenges and solutions (consolidated)

### 8.1 EMG / ML (this work — likely missing from other chapters)
| # | Challenge | Root cause | Solution |
|---|-----------|------------|----------|
| 1 | No per-class metrics existed | only aggregate accuracy saved | computed per-class F1 directly |
| 2 | First CV gave optimistic 57.7 % | 50 % window overlap → train/test leakage | leave-one-cycle-out (leakage-free) |
| 3 | Near-chance cross-cycle accuracy | cross-cycle distribution drift | honest reporting; mains notch + subset; flag more data |
| 4 | CRNN overfits (train 100 %, val low) | single participant, ~1 usable channel | deploy classical RF; keep CRNN as scaffold |
| 5 | 50 Hz mains in v3_env | hardware envelope doesn't suppress mains | causal streaming notch (50 Hz + harmonics) |
| 6 | Live notch broke a mains-off model | notch shifts features ~71 % (not "absorbed") | retrain **with** matching notch |
| 7 | filtfilt (offline) ≠ causal (live) | zero-phase \|H\|² can't run real-time | use the **causal** notch in training too |
| 8 | CRNN not loadable by inference | torch model vs sklearn `predict` API | wrapper: `predict_window` + 12-wide id-indexed proba, joblib |
| 9 | CHEEZ channel dead in all recordings | signal not reaching the ADC pin | flagged hardware fix; run v3-only in practice |

### 8.2 Networking / deployment
| # | Challenge | Root cause | Solution |
|---|-----------|------------|----------|
| 10 | ESP wouldn't join phone hotspot | phone served 5 GHz/dual-band; ESP32 is 2.4 GHz only | AP-mode firmware toggle + status-code diagnostics |
| 11 | Devices "on same WiFi" couldn't reach each other | different subnets/bands, or AP client-isolation | consolidate on one 2.4 GHz net; verify with ping/ARP |
| 12 | ESP TCP "connected but no data" | `availableForWrite()` returned 0 on the core → dropped batches; per-sample writes stalled sampler | write unconditionally; ~10 ms batching; ADC1 pins; 921600 baud |
| 13 | ROS python couldn't import scipy/torch | system numpy 2.x vs scipy/sklearn built for numpy 1.x | venv `--system-site-packages` (keeps rclpy) + `numpy<2` stack |
| 14 | torch download timed out on the Pi | 426 MB over weak WiFi | longer pip timeout/retries; resumable wheel / LAN transfer |
| 15 | Servo lib `No module named RPi` | classic RPi.GPIO vs kernel 6.x on Ubuntu | `rpi-lgpio` drop-in |

### 8.3 Arm / simulation (detailed in CONTROL_REPORT.md — summarized here)
| # | Challenge | Resolution |
|---|-----------|------------|
| 16 | SolidWorks→URDF meshes offset to global origin | re-export with a per-link reference frame at each joint pivot |
| 17 | Y-up (CAD) vs Z-up (ROS) | fixed `world→base_footprint→base_link` (rpy −1.5708) to stand it upright |
| 18 | Exporter emitted two joints named `j6` | split into driver `j6l` + passive mimic `j6r` (−2.0885) |
| 19 | Shoulder sag under one servo's torque | dual mirrored servos (2.94 N·m) — verified hold test |
| 20 | 5-DOF arm vs KDL's 6-DOF assumption | rely on joint-space/named-state goals; 5-DOF IK = future work |
| 21 | Arm jerked / many joints at once | vote mode (1 move/3 s) + freeze non-selected joints (this work) |
| 22 | Base axis differs from the rest (UPDATE) | base is a DRV8825 stepper; real-arm step/dir driving is an open item |

---

## 9. Limitations
- **Single participant, single session** for the deployed models; no inter-subject
  validation. Cross-cycle accuracy (~38 % on 7 classes, ~2.7× chance) reflects drift.
- **One functional EMG channel** (CHEEZ dead) caps separability.
- **CRNN overfits** at this data volume; reported as architecture/pipeline contribution.
- **Base-joint actuation gap** between simulation (uniform position control) and the real
  arm (DRV8825 stepper).
- **Open-loop** position control throughout (no joint feedback on the real arm).
- The **3 s vote debounce** trades responsiveness for stability (appropriate for a noisy
  classifier, but a stated limitation).

## 10. Future work
- **More participants and sessions** + session/cycle normalization (e.g. a short REST
  calibration recorded at session start) to fight drift — the highest-leverage fix.
- **Repair the CHEEZ channel** (or a clean second electrode) to restore 2 channels.
- **Integrate the DRV8825 base stepper** (step/dir + homing) into the real-arm driver so
  the base matches the simulation's command space.
- **Re-evaluate the CRNN once data scales**, where deep models should overtake the RF.
- **5-DOF-aware / position-only IK**; closed-loop joint feedback.

---

## 11. Artifacts / file inventory
- **Pipeline:** `moving to ubuntu (2)/moving to ubuntu/code/` — `emg_config.py`,
  `preprocess_v0_windowed.py` (76-D features; lazy librosa; `--mains-causal`),
  `mains_notch.py` (live causal notch), `train_classical.py`, `train_rcnn.py` +
  `rcnn_model.py` (CRNN + inference wrapper), `make_subset.py` (6-gesture filter),
  `inference_rt.py` (USB + WiFi real-time + mains; pyserial optional),
  `tools/fake_esp_tcp.py` (TCP replayer). Firmware: `semg_stream_live.ino` (USB),
  `semg_stream_live_wifi.ino` (WiFi AP/STA).
- **Models:** `emg_ml/models/classical_6g.joblib` (deployed RF, mains-50),
  `emg_ml/models/rcnn_6g.joblib` (CRNN, mains-50).
- **Data:** `…/code/data/raw_good/P1_20260606_231716.csv`; `…/data/raw_6g/` (7 classes).
- **ROS 2:** `src/emg_control/` — `emg_inference_node.py` (CRNN/classical + live notch),
  `emg_bridge_node.py` (vote mode + one-joint-at-a-time), `config/emg_params.yaml`,
  `launch/emg.launch.py`; arm sim `rotatedauto`, MoveIt `rotatedauto_moveit_config`.
- **Pi:** `raspberry_pi/inference/` (inference bundle), `raspberry_pi/arm_control/`
  (PCA9685 servo driver + gesture→joint bridge), `raspberry_pi/PI_SETUP.md`.
- **Existing reports to cross-reference:** `raspberry_pi/CONTROL_REPORT.md` (full control
  + simulation + arm hardware depth, incl. A11–A19 appendices), project chapters
  (Mechanical, Electronic, Control).

---

## 12. Notes for the report author (Sonnet 4.6)
- **Apply §0 corrections** wherever the older chapters conflict (base stepper, 500 Hz +
  CHEEZ, MG995, Jazzy/24.04, mirrored-shoulder account).
- **Keep methodology labels** on every accuracy figure (cross-cycle vs. mixed-cycle/leaky
  vs. LOCO). Do **not** present 57.7 % as a generalization result.
- **Foreground the genuinely novel/defensible contributions:** (a) the **causal live
  50 Hz mains notch with a verified train==inference guarantee**, shown to improve
  accuracy (~25 %→38 %); (b) the **inference-conformant recurrent-CNN wrapper** making a
  torch model a drop-in for an sklearn-style real-time loop; (c) the **leakage-aware
  evaluation** (catching/correcting the windowed-CV optimism); (d) the **wireless,
  debounced, one-joint-at-a-time end-to-end EMG→arm system** across sim and hardware.
- **Be candid** about data-limited accuracy and the dead CHEEZ channel. A fair framing:
  "a working, debounced, wireless end-to-end EMG→arm system whose accuracy is bounded by
  single-subject cross-cycle drift and a one-channel hardware fault."
- **Confirm the base motor** (DRV8825 → likely NEMA 17) on the bench before stating it as
  fact. For arm/sim depth, draw from `CONTROL_REPORT.md` rather than re-deriving.
