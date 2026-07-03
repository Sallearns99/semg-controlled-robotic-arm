# Raspberry Pi setup runbook — EMG WiFi receiver

This is the step-by-step bring-up for the Raspberry Pi that receives the EMG
stream from the ESP32-S3 over WiFi.

Data contract end-to-end (do not change): newline-delimited ASCII lines
`ch1,ch2\n`, raw 12-bit ADC counts (0–4095), ~2000 lines/s.

Pipeline: **ESP32-S3 (TCP server :5005) → WiFi LAN → Raspberry Pi (this) → dev PC ROS 2**

---

## 0. First boot with a keyboard + HDMI→USB capture (no spare monitor)

You said you have a keyboard and an **HDMI-to-USB (UVC capture)** adapter — the
kind that turns the Pi's HDMI output into a USB video stream you can view on
another computer. That works, with one caveat: that adapter does **not** make
the Pi use a monitor by itself — you must **view the captured video on your dev
PC**. Two situations:

**A) You have the HDMI→USB *capture* dongle (HDMI male → USB, shows up as a
webcam on the PC):**
1. Pi HDMI OUT → dongle HDMI IN → dongle USB → **dev PC** USB port.
2. On the dev PC, open the capture device as a webcam:
   ```bash
   sudo apt install -y v4l-utils ffmpeg
   v4l2-ctl --list-devices          # find the /dev/videoN of the dongle
   ffplay -f v4l2 -framerate 30 -video_size 1280x720 /dev/videoN
   # (or use 'guvcview' / 'cheese' / OBS "Video Capture Device")
   ```
3. Plug the keyboard into the Pi. Power the Pi. You should see the Pi's boot
   output in the ffplay window and can type on the keyboard.
   - The Pi only outputs HDMI if it detects a sink at boot. Some capture dongles
     don't assert hotplug detect. If the window stays black, force HDMI on:
     power off, edit `config.txt` on the SD card's boot partition (mount it on
     the PC), add `hdmi_force_hotplug=1` (Pi 4) — or for **Pi 5 / Bookworm**
     KMS, add under `[all]`: `video=HDMI-A-1:1280x720@60D`. Reinsert, boot again.

**B) If instead your adapter is HDMI→(something the Pi plugs into):** an "HDMI to
USB-B" cable is unusual; the Pi outputs HDMI, it does not take HDMI in. If the
USB end goes to the PC, treat it as case (A) above. The Pi's own video only
comes out of its HDMI/micro-HDMI port.

> **Strongly recommended shortcut — skip the monitor entirely (headless):**
> Flash the SD card with **Raspberry Pi Imager** and in its settings (gear icon)
> pre-configure: hostname `raspberrypi`, **enable SSH**, set username/password,
> and **enter your WiFi SSID + password**. Then you never need the keyboard or
> capture dongle — just power the Pi and `ssh` in from the dev PC (Step 3).
> This is the most reliable path and is what the rest of this runbook assumes you
> can do. Use the keyboard/capture route only for first-time troubleshooting.

---

## 1. Choose the OS

| Option | When to pick it |
|---|---|
| **Raspberry Pi OS (64-bit, Bookworm)** | You will use the Pi only to **forward** the TCP stream to the dev PC (Pi does NOT run ROS). Lightest, best-supported. **Recommended for first bring-up.** |
| **Ubuntu 24.04 LTS (64-bit) for Pi** | You want the Pi to **run ROS 2 Jazzy itself** (on-arm inference, no PC). ROS 2 Jazzy targets Ubuntu 24.04 — so if the Pi runs ROS, it must be Ubuntu 24.04, not Pi OS. |

Decision guidance:
- **Phase-2 goal is on-arm compute**, but for getting WiFi working *first*, run
  **Pi OS 64-bit + forward to the PC** (keep ROS on the PC where it already
  works). Once the link is solid, optionally reflash to Ubuntu 24.04 and run
  Jazzy on the Pi. Pi 4 (4 GB+) and Pi 5 both run Jazzy; **Pi 5 is recommended**
  for the sklearn + librosa MFCC feature extraction at 10 Hz (every 100 ms hop).

Flash with **Raspberry Pi Imager** (`sudo apt install rpi-imager` on the PC, or
download from raspberrypi.com). In Imager, click the gear → enable SSH, set
user/pass, set WiFi SSID/country.

---

## 2. Put the Pi on the SAME WiFi LAN as the ESP32

The ESP32 and Pi **must be on the same subnet** (same router/SSID, 2.4 GHz —
the ESP32-S3 is 2.4 GHz only).

- If you used Imager's WiFi setting, the Pi joins on boot. Done.
- Otherwise on the Pi:
  - **Pi OS Bookworm (NetworkManager):**
    ```bash
    sudo nmcli device wifi connect "YOUR_SSID" password "YOUR_PASS"
    nmcli device status
    ```
  - **Ubuntu 24.04 (netplan):** edit `/etc/netplan/50-cloud-init.yaml` with your
    SSID/pass under `wifis:`, then `sudo netplan apply`.
- Confirm the Pi's own IP: `hostname -I`.

---

## 3. SSH in from the dev PC

```bash
ssh <user>@raspberrypi.local        # mDNS hostname set in Imager
# or use the IP from your router / `hostname -I` on the Pi
```
If `raspberrypi.local` doesn't resolve from the PC, install mDNS on the PC:
`sudo apt install -y avahi-daemon avahi-utils` (Pi OS has it by default).

---

## 4. Find (and pin) the ESP32's IP

The firmware prints its IP over USB Serial on boot **and** advertises mDNS as
`esp32.local`. Three ways to reach it, best first:

1. **mDNS (zero config):** just use `esp32.local`. Test from the Pi:
   ```bash
   sudo apt install -y avahi-utils
   ping esp32.local
   avahi-browse -rt _emg._tcp        # the firmware advertises this service
   ```
2. **DHCP reservation (most robust):** in your router admin page, bind the
   ESP32's MAC to a fixed IP (e.g. 192.168.1.50). The MAC is printable — add
   `Serial.println(WiFi.macAddress());` if you want it, or read it from the
   router's client list. Then always connect to that IP.
3. **Read it from USB Serial once:** plug the ESP into any computer, open a
   serial monitor at 921600 baud, note the `WiFi IP:` line.

**Static IP on the ESP (optional alternative to DHCP reservation):** add before
`WiFi.begin()` in the sketch:
```cpp
IPAddress ip(192,168,1,50), gw(192,168,1,1), mask(255,255,255,0), dns(192,168,1,1);
WiFi.config(ip, gw, mask, dns);
```

---

## 5. Test connectivity to port 5005

From the Pi (with the ESP powered and a WiFi client slot free):
```bash
ping -c3 esp32.local                       # is it reachable?
nc esp32.local 5005 | head -n 20           # should print ch1,ch2 lines!
# e.g.
#   2013,1987
#   2010,1990
```
If `nc` prints lines, the whole WiFi path works. Ctrl-C to stop (the ESP serves
one client at a time, so free it before running the receiver).

---

## 6. Firewall / port 5005

The Pi is the **client** (it dials out to the ESP), so inbound firewall rules
usually don't matter on the Pi. If you later run something that listens on the
Pi (e.g. re-broadcast), open it:
```bash
# Pi OS / Ubuntu with ufw:
sudo ufw allow 5005/tcp
```
On the **dev PC**, if the Pi forwards/re-serves to it, allow 5005 there too.
By default neither Pi OS nor a stock Ubuntu desktop blocks outbound, so the
basic forward path needs no firewall changes.

---

## 7. Run the receiver

Copy `tcp_receiver.py` to the Pi (e.g. `scp`), then:

```bash
# Smoke test — just print lines:
python3 tcp_receiver.py --host esp32.local --mode print

# Production — relay into a PTY so a serial consumer reads it unchanged:
sudo python3 tcp_receiver.py --host esp32.local --mode pty --link /dev/ttyEMG
# (sudo only needed to create the symlink under /dev; or point --link at a path
#  in your home dir and use that as port:=)
```

### 7a. If the Pi only FORWARDS (ROS stays on the PC)
Run the inference node ON THE PC, but read from the Pi. Two clean options:

- **Simplest: the PC connects to the ESP directly** (the Pi need not be in the
  data path at all for forwarding — both are on the LAN). On the PC:
  ```bash
  socat -d -d pty,link=/dev/ttyEMG,raw,echo=0 tcp:esp32.local:5005 &
  ros2 run emg_control emg_inference_node.py --ros-args -p port:=/dev/ttyEMG
  ```
- **Pi as relay/aggregator:** run `tcp_receiver.py --mode stdout-raw` on the Pi
  piped into a re-server, or just keep the PC↔ESP direct path above and use the
  Pi only when it will host ROS.

### 7b. If the Pi RUNS ROS 2 Jazzy (Ubuntu 24.04 on the Pi)
On the Pi:
```bash
sudo python3 tcp_receiver.py --host esp32.local --mode pty --link /dev/ttyEMG
# then, in the Pi's ROS workspace:
ros2 run emg_control emg_inference_node.py --ros-args -p port:=/dev/ttyEMG
```
The node is UNCHANGED — it just opens `/dev/ttyEMG` as if it were the ESP-over-USB.

> Documented native option (NOT implemented; would require editing the node):
> add a `source:=tcp` + `tcp_host`/`tcp_port` param to `emg_inference_node.py` so
> its data-source thread reads the socket directly and skips the PTY. Cleaner
> long-term, but the PTY path above needs zero code changes today.

---

## 8. Run the receiver on boot (systemd)

Create `/etc/systemd/system/emg-receiver.service`:

```ini
[Unit]
Description=EMG WiFi TCP receiver (ESP32-S3 -> PTY /dev/ttyEMG)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
# Adjust the path to where you copied tcp_receiver.py:
ExecStart=/usr/bin/python3 /home/pi/emg/tcp_receiver.py --host esp32.local --mode pty --link /dev/ttyEMG
Restart=always
RestartSec=3
User=root
# (root so it can create /dev/ttyEMG; or run as your user with --link in $HOME)

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now emg-receiver.service
systemctl status emg-receiver.service
journalctl -u emg-receiver.service -f      # watch logs / reconnects
```

---

## 9. Pi 4 vs Pi 5 for ROS Jazzy (feasibility)

- **Forward-only role:** any Pi (even Pi 3/Zero 2 W) handles 24 kB/s easily.
- **Run ROS 2 Jazzy + inference on the Pi:** needs **Ubuntu 24.04 64-bit**
  (Jazzy's target). The inference does librosa MFCCs + sklearn predict every
  100 ms — light but not free. **Pi 4 (4 GB+) works; Pi 5 recommended** for
  headroom and faster feature extraction. Pi OS cannot run Jazzy from apt
  (different distro), so don't try to apt-install ROS Jazzy on Pi OS.

**Recommendation:** bring up WiFi with **Pi OS + forward to the PC** first
(fewest moving parts). Migrate to **Ubuntu 24.04 on a Pi 5 running ROS** only
once the link and the arm sim→hardware bridge are proven.

---

## 10. Troubleshooting (see README.md for more)

| Symptom | Check |
|---|---|
| `nc esp32.local 5005` hangs | ESP not on WiFi (check its USB Serial), or another client already connected (ESP serves one at a time), or wrong subnet. |
| `esp32.local` won't resolve | install `avahi-utils`; or use the raw IP / DHCP reservation. |
| Lines but they stop | WiFi drop — receiver auto-reconnects; watch `journalctl`. Disable Pi WiFi power-save: `sudo iw dev wlan0 set power_save off`. |
| ROS node opens port but no gestures | confirm `/dev/ttyEMG` exists and carries data: `cat /dev/ttyEMG | head`. |
