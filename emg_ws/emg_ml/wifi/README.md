# EMG over WiFi: ESP32-S3 → Raspberry Pi → dev PC

Streams the 2-channel EMG signal from the ESP32-S3 to the rest of the pipeline
over WiFi instead of (or in addition to) USB serial, keeping the exact same
`ch1,ch2\n` ASCII contract so the existing inference pipeline runs unchanged.

## Files here
| File | What it is |
|---|---|
| `emg_stream_wifi.ino` | ESP32-S3 Arduino sketch: samples 2 EMG channels at 2000 Hz, serves them over **TCP port 5005**, also mirrors to USB Serial, advertises `esp32.local` via mDNS. |
| `tcp_receiver.py` | Raspberry Pi (or PC) TCP client: connects to the ESP, relays the byte stream into a **PTY** (`/dev/ttyEMG`) so the serial-based ROS node reads it unmodified. Also has print / raw modes. Auto-reconnects. |
| `RPI_SETUP.md` | Full Raspberry Pi bring-up runbook (OS, WiFi, find/pin ESP IP, firewall, systemd, Pi-runs-ROS vs forward). |
| `README.md` | This file. |

These are **new** files. The originals are untouched:
`../reference/firmware/emg_stream.ino` (USB-serial firmware) and
`../../src/emg_control/scripts/emg_inference_node.py` (the ROS consumer).

## Architecture

```
  ┌──────────────────────┐   2.4GHz WiFi LAN    ┌──────────────────────┐
  │  ESP32-S3-N16R8      │                       │  Raspberry Pi        │
  │  EMG ADC ch1=GPIO3   │   TCP :5005           │  tcp_receiver.py     │
  │      ch2=GPIO8       │  ch1,ch2\n  ───────►  │  (TCP client)        │
  │  TCP SERVER  ───────────────────────────────►│  relays bytes -> PTY │
  │  + mDNS esp32.local  │   ~2000 lines/s       │  /dev/ttyEMG         │
  │  + USB Serial mirror │   ~24 kB/s            └───────────┬──────────┘
  └──────────────────────┘                                   │
                                                              │  port:=/dev/ttyEMG
                                            ┌─────────────────▼──────────────────┐
                                            │ emg_inference_node.py (ROS 2 Jazzy) │
                                            │ readline() ch1,ch2 -> features ->   │
                                            │ model -> /detected_gesture          │
                                            └─────────────────────────────────────┘
   (ROS node runs on the dev PC OR on the Pi if it's Ubuntu 24.04 — see RPI_SETUP.md)
```

Note: the Pi is the natural place for the receiver because the project's Phase 2
puts the Pi on the arm. But for first bring-up the **dev PC can connect to the
ESP directly** (same LAN) and run `socat`/the receiver itself — the Pi is
optional until you move compute onto the arm.

## Data format (unchanged, end to end)
- ASCII line per sample: `ch1,ch2\n`
- `ch1`, `ch2` = raw 12-bit ADC counts, `0`–`4095` (decimal text).
- Rate: exactly 2000 Hz (2000 lines/s). Baud is irrelevant over WiFi.
- On the wire the ESP **batches ~10 ms of lines per TCP send** for efficiency,
  but the byte stream is still the same newline-delimited lines — the consumer's
  `readline()` is unaffected.

## Design choices (and why)
- **TCP, not UDP.** ~24 kB/s is trivial for WiFi, and the inference window is
  200 ms / 100 ms hop, so a few ms of LAN latency is irrelevant. TCP gives
  ordered, lossless, contiguous bytes → the existing `readline()` parser works
  with zero framing logic. UDP would drop/reorder samples and corrupt the
  feature window for no real latency benefit. (Details in the sketch header.)
- **Batched writes on the ESP.** Per-sample TCP writes at 2000 Hz = 2000
  syscalls/s and risk stalling the 500 µs sample schedule. We accumulate ~10 ms
  of lines and flush, never blocking the sampler (drop a batch if the socket is
  backed up). Sampling cadence always wins.
- **ADC1 pins only.** GPIO3 + GPIO8 are ADC1 (ADC1 = GPIO1–10 on ESP32-S3), so
  they keep working while WiFi is active. ADC2 (GPIO11–20) is unusable with WiFi
  on — do not move the EMG inputs there.
- **PTY relay for zero-touch integration.** The ROS node opens a serial `port`
  and does `readline()`. Relaying the identical bytes into a PTY and pointing
  `port:=/dev/ttyEMG` means the node needs **no changes**.
- **Pi forwards first, can run ROS later.** ROS 2 Jazzy targets Ubuntu 24.04, so
  if the Pi runs ROS it must be Ubuntu 24.04 (Pi 5 recommended). Easiest first
  step: Pi OS + forward to the PC, keep ROS on the PC. See RPI_SETUP.md.

## Bring-up (quick path)
1. **Flash the ESP:** open `emg_stream_wifi.ino` in Arduino IDE, board
   "ESP32S3 Dev Module", set `WIFI_SSID` / `WIFI_PASSWORD` (2.4 GHz), upload.
2. **Find the ESP:** open Serial Monitor @ 921600 — note the `WiFi IP:` line; it
   also answers as `esp32.local`.
3. **Test from the Pi (or PC):** `nc esp32.local 5005 | head` → you should see
   `ch1,ch2` lines.
4. **Relay to a PTY:** `sudo python3 tcp_receiver.py --host esp32.local --mode pty --link /dev/ttyEMG`
5. **Run inference unchanged:**
   `ros2 run emg_control emg_inference_node.py --ros-args -p port:=/dev/ttyEMG`

(For boot persistence on the Pi, use the systemd unit in RPI_SETUP.md.)

## Troubleshooting
| Problem | Likely cause / fix |
|---|---|
| **No data at all** | ESP not connected to WiFi — check USB Serial for `connected.`/`WiFi IP:`. SSID must be **2.4 GHz**. Wrong SSID/pass → sketch falls back to Serial-only and says so. |
| **`nc`/receiver can't connect** | Wrong IP, or ESP already has a client (it serves **one** at a time — close other `nc`/receiver). Confirm same subnet: `ping esp32.local`. |
| **`esp32.local` won't resolve** | Install `avahi-utils` (Pi/PC); or use the raw IP; or set a DHCP reservation (RPI_SETUP.md §4). |
| **Connects then drops** | WiFi power-save or weak signal. Receiver auto-reconnects. On Pi: `sudo iw dev wlan0 set power_save off`. Firmware already sets `WiFi.setSleep(false)`. Move closer / 2.4 GHz channel. |
| **High latency / laggy gestures** | Confirm `setNoDelay(true)` (it's set), short `FLUSH_MS` (10 ms). Check WiFi RSSI; avoid congested channels. The 100 ms hop dominates, so ≤10 ms link latency is fine. |
| **Throughput at 2000 Hz looks low / gaps** | The ESP drops a batch if the socket backs up (by design, to protect sampling). Persistent gaps = weak WiFi or a stalled receiver. Verify with `nc ... | wc -l` over 1 s ≈ 2000. |
| **ROS node opens port but no gestures** | Check the PTY carries data: `cat /dev/ttyEMG | head`. Confirm `port:=/dev/ttyEMG` matches the receiver's `--link`. |
| **ADC reads garbage with WiFi on** | You moved a channel to an ADC2 pin (GPIO11–20). Keep EMG on ADC1 (GPIO1–10); GPIO3/GPIO8 are correct. |

## Verifying line rate
```bash
# ~2000 expected:
timeout 1 nc esp32.local 5005 | wc -l
```
