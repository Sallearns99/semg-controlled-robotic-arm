#!/usr/bin/env python3
"""tcp_receiver.py — Raspberry Pi side of the EMG WiFi link.

Connects to the ESP32-S3 TCP server (default esp32.local:5005), reads the
`ch1,ch2\\n` ASCII stream, and exposes it to the EXISTING serial-based pipeline
WITHOUT modifying any existing file.

Modes
-----
  --mode pty   (default, recommended)
      Create a local pseudo-terminal (PTY) and relay every byte from the ESP
      verbatim into it, then symlink a stable device name to the PTY slave.
      The ROS 2 emg_inference_node then runs UNCHANGED with:
          ros2 run emg_control emg_inference_node.py --ros-args -p port:=/dev/ttyEMG
      Because the node just does serial.readline() on `port`, a PTY carrying the
      identical byte stream is indistinguishable from the real ESP-over-USB.

  --mode print
      Just print the lines to stdout (smoke test / debugging).

  --mode stdout-raw
      Write raw bytes to stdout (pipe into another tool).

Notes
-----
  * Pure standard library — no extra pip installs needed on the Pi.
  * Auto-reconnects if the ESP drops or reboots.
  * PTY slave names (/dev/pts/N) are NOT stable across runs, so we always create
    a fixed symlink (--link, default /dev/ttyEMG) pointing at the current slave.

One-liner alternative (no script):
  socat -d -d pty,link=/dev/ttyEMG,raw,echo=0 tcp:esp32.local:5005
  (socat doesn't auto-reconnect on ESP reboot; this script does.)

Integration path for the ROS node (documented, NOT implemented here — do not
modify emg_inference_node.py):
  - Cleanest no-touch: point its `port:=` at the PTY symlink this script makes.
  - A future native option would be to add a `source:=tcp` / `tcp_host` param to
    the node and have its data-source thread read the socket directly instead of
    pyserial. That removes the PTY hop but requires editing the node, so it is
    intentionally left as a documented option only.
"""
import argparse
import os
import pty
import select
import signal
import socket
import sys
import time

DEFAULT_HOST = "esp32.local"   # mDNS name advertised by the firmware
DEFAULT_PORT = 5005
DEFAULT_LINK = "/dev/ttyEMG"   # stable symlink for the ROS node's port:=

_stop = False


def _sig(_signum, _frame):
    global _stop
    _stop = True


def connect(host, port, retry_s):
    """Block until connected; retry forever (handles ESP boot ordering)."""
    while not _stop:
        try:
            s = socket.create_connection((host, port), timeout=5.0)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.settimeout(5.0)
            print(f"[tcp_receiver] connected to {host}:{port}", file=sys.stderr)
            return s
        except OSError as e:
            print(f"[tcp_receiver] connect to {host}:{port} failed: {e}; "
                  f"retrying in {retry_s}s", file=sys.stderr)
            time.sleep(retry_s)
    return None


def run_pty(host, port, link, retry_s):
    master_fd, slave_fd = pty.openpty()
    slave_name = os.ttyname(slave_fd)
    # Maintain a stable symlink to the (unstable) slave device name.
    try:
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(slave_name, link)
        print(f"[tcp_receiver] PTY slave {slave_name} -> symlink {link}", file=sys.stderr)
        print(f"[tcp_receiver] point the ROS node at it: "
              f"... -p port:={link}", file=sys.stderr)
    except OSError as e:
        # If we can't make the symlink (perms), still print the slave name so the
        # user can point port:= at it directly.
        print(f"[tcp_receiver] WARNING could not create {link}: {e}. "
              f"Use port:={slave_name} instead.", file=sys.stderr)

    sock = None
    try:
        while not _stop:
            if sock is None:
                sock = connect(host, port, retry_s)
                if sock is None:
                    break
            # Pump ESP socket -> PTY master. (We ignore PTY->ESP; stream is one-way.)
            try:
                r, _, _ = select.select([sock, master_fd], [], [], 1.0)
                if sock in r:
                    data = sock.recv(4096)
                    if not data:
                        raise ConnectionError("ESP closed the connection")
                    os.write(master_fd, data)
                if master_fd in r:
                    # Drain anything the reader writes back (none expected); prevents
                    # the PTY from blocking if the consumer ever writes.
                    os.read(master_fd, 4096)
            except (socket.timeout, BlockingIOError):
                continue
            except (OSError, ConnectionError) as e:
                print(f"[tcp_receiver] link lost: {e}; reconnecting", file=sys.stderr)
                try:
                    sock.close()
                except OSError:
                    pass
                sock = None
                time.sleep(retry_s)
    finally:
        if sock is not None:
            sock.close()
        try:
            if os.path.islink(link) and os.readlink(link) == slave_name:
                os.remove(link)
        except OSError:
            pass
        os.close(master_fd)
        os.close(slave_fd)


def run_simple(host, port, retry_s, raw):
    sock = None
    buf = b""
    try:
        while not _stop:
            if sock is None:
                sock = connect(host, port, retry_s)
                if sock is None:
                    break
            try:
                data = sock.recv(4096)
                if not data:
                    raise ConnectionError("ESP closed the connection")
            except socket.timeout:
                continue
            except (OSError, ConnectionError) as e:
                print(f"[tcp_receiver] link lost: {e}; reconnecting", file=sys.stderr)
                try:
                    sock.close()
                except OSError:
                    pass
                sock = None
                time.sleep(retry_s)
                continue

            if raw:
                sys.stdout.buffer.write(data)
                sys.stdout.buffer.flush()
            else:
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    print(line.decode("ascii", "ignore"))
    finally:
        if sock is not None:
            sock.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=DEFAULT_HOST,
                    help=f"ESP32 host/IP (default {DEFAULT_HOST})")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT,
                    help=f"ESP32 TCP port (default {DEFAULT_PORT})")
    ap.add_argument("--mode", choices=["pty", "print", "stdout-raw"], default="pty",
                    help="pty: relay to a PTY for the serial ROS node (default); "
                         "print: print lines; stdout-raw: raw bytes to stdout")
    ap.add_argument("--link", default=DEFAULT_LINK,
                    help=f"stable symlink to the PTY slave (default {DEFAULT_LINK})")
    ap.add_argument("--retry", type=float, default=2.0,
                    help="reconnect retry interval seconds (default 2)")
    args = ap.parse_args()

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    if args.mode == "pty":
        run_pty(args.host, args.port, args.link, args.retry)
    else:
        run_simple(args.host, args.port, args.retry, raw=(args.mode == "stdout-raw"))

    print("[tcp_receiver] stopped.", file=sys.stderr)


if __name__ == "__main__":
    main()
