#!/usr/bin/env python3
"""emg_monitor.py — live ESP32 EMG signal-health check (standalone, no ROS needed).

Reads the emg_stream.ino serial stream (ch1,ch2 raw 12-bit ADC @ 2000 Hz, 921600 baud)
and shows, per channel, a live bar + stats + a verdict so you can confirm the ESP is
giving REAL EMG (rest = low & steady, flex = rises) and not noise / a dead electrode.

Usage:
    ros2 run emg_control emg_monitor.py                  # default /dev/ttyACM0
    python3 emg_monitor.py --port /dev/ttyACM0
    python3 emg_monitor.py --list                        # list serial ports

Verdicts per channel (over the rolling window):
    NO-DATA  no samples arriving        -> cable / firmware / baud / wrong port
    FLAT     signal perfectly constant  -> electrode unplugged / sensor unpowered
    RAIL     stuck at 0 or 4095         -> saturating / bad gain / bad contact
    NOISY    high RMS while baseline-ish -> 50/60 Hz pickup / loose electrode
    ok       data flowing, plausible    -> flex the muscle: bar & RMS should jump
"""
import argparse
import collections
import sys
import threading
import time

ADC_MAX = 4095
RAIL_LO, RAIL_HI = 30, ADC_MAX - 30   # within 30 counts of a rail = "railed"
FS_NOMINAL = 2000


def list_ports():
    import serial.tools.list_ports
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("No serial ports found.")
    for p in ports:
        print(f"  {p.device:<14} {p.description}")


def bar(value, width=14, vmax=ADC_MAX):
    frac = max(0.0, min(1.0, value / vmax))
    n = int(frac * width)
    return "█" * n + " " * (width - n)


class Channel:
    def __init__(self, maxlen):
        self.buf = collections.deque(maxlen=maxlen)

    def push(self, v):
        self.buf.append(v)

    def stats(self):
        if not self.buf:
            return None
        import numpy as np
        a = np.fromiter(self.buf, dtype=np.float64)
        mean = a.mean()
        rms_ac = a.std()                       # RMS of the AC component (activity/noise)
        lo, hi = a.min(), a.max()
        railed = ((a <= RAIL_LO) | (a >= RAIL_HI)).mean()
        return dict(mean=mean, rms=rms_ac, lo=lo, hi=hi, pp=hi - lo, railed=railed, n=len(a))

    def verdict(self, s, fresh):
        if s is None or not fresh:
            return "NO-DATA"
        if s["pp"] == 0:
            return "FLAT"
        if s["railed"] > 0.5:
            return "RAIL"
        # noisy: lots of AC swing but mean still near baseline (rough heuristic)
        if s["rms"] > 250 and s["mean"] < 400:
            return "NOISY"
        return "ok"


def main():
    ap = argparse.ArgumentParser(description="Live ESP32 EMG signal monitor")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--window-ms", type=int, default=500)
    ap.add_argument("--refresh-hz", type=float, default=10.0)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        list_ports(); return

    import serial
    try:
        ser = serial.Serial(args.port, args.baud, timeout=1.0)
        time.sleep(0.1)
        ser.reset_input_buffer()
    except Exception as e:
        print(f"ERROR opening {args.port} @ {args.baud}: {e}")
        print("  - is the ESP plugged in and running emg_stream.ino?")
        print("  - are you in the 'dialout' group?  (groups | grep dialout)")
        print("  - check the port with:  python3 emg_monitor.py --list")
        sys.exit(1)

    win = int(args.window_ms * FS_NOMINAL / 1000)
    ch1, ch2 = Channel(win), Channel(win)
    stop = threading.Event()
    state = {"count": 0, "errors": 0, "last_rx": 0.0, "t0": time.time(), "n0": 0, "tfs": time.time()}
    lock = threading.Lock()

    def reader():
        while not stop.is_set():
            try:
                line = ser.readline().decode("ascii", errors="ignore").strip()
            except Exception:
                continue
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 2:
                with lock: state["errors"] += 1
                continue
            try:
                a, b = int(parts[0]), int(parts[1])
            except ValueError:
                with lock: state["errors"] += 1
                continue
            with lock:
                ch1.push(a); ch2.push(b)
                state["count"] += 1
                state["last_rx"] = time.time()

    th = threading.Thread(target=reader, daemon=True); th.start()

    print(f"Monitoring {args.port} @ {args.baud}.  Flex a muscle — its bar & RMS should jump.")
    print("Ctrl-C to stop.\n")
    period = 1.0 / args.refresh_hz
    try:
        while True:
            time.sleep(period)
            now = time.time()
            with lock:
                s1, s2 = ch1.stats(), ch2.stats()
                cnt, err, last_rx = state["count"], state["errors"], state["last_rx"]
                dn = cnt - state["n0"]; dt = now - state["tfs"]
                state["n0"], state["tfs"] = cnt, now
            fs = dn / dt if dt > 0 else 0.0
            fresh = (now - last_rx) < 0.5
            v1 = ch1.verdict(s1, fresh); v2 = ch2.verdict(s2, fresh)

            if not fresh or s1 is None:
                sys.stdout.write(f"\r  NO DATA arriving (rx {cnt}, errs {err}) — check cable/firmware/baud/port     ")
                sys.stdout.flush(); continue

            line = (f"\rfs:{fs:4.0f}Hz err:{err:<4d} | "
                    f"CH1 m{s1['mean']:4.0f} rms{s1['rms']:4.0f} [{bar(s1['mean'])}] {v1:<7} | "
                    f"CH2 m{s2['mean']:4.0f} rms{s2['rms']:4.0f} [{bar(s2['mean'])}] {v2:<7}")
            sys.stdout.write(line + "  "); sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set(); ser.close()
        print("\nstopped.")


if __name__ == "__main__":
    main()
