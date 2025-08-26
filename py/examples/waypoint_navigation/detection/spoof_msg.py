#!/usr/bin/env python3

# Sends a fake msg to socket for testing
# 
# python3 spoof_msg.py single --x 1.2 --y 0.5 --conf 0.9

import argparse, csv, json, socket, sys, time
from typing import Iterable, Tuple

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 41234

def send(sock: socket.socket, host: str, port: int, x: float, y: float, conf: float):
    payload = {"type": "cone_goal", "x_fwd_m": float(x), "y_left_m": float(y), "confidence": float(conf)}
    data = json.dumps(payload).encode("utf-8")
    sock.sendto(data, (host, port))
    print(f"sent -> {payload}")

def iter_approach(x0: float, y0: float, x1: float, y1: float, steps: int) -> Iterable[Tuple[float,float]]:
    if steps < 1:
        yield (x1, y1)
        return
    for i in range(steps):
        t = (i + 1) / steps
        yield (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)

def main():
    p = argparse.ArgumentParser(description="Send fake 'cone_goal' UDP messages to the nav script.")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)

    sub = p.add_subparsers(dest="mode", required=False)

    one = sub.add_parser("single", help="Send one or many identical messages (default).")
    one.add_argument("--x", type=float, default=1.2, help="x_fwd_m (forward, meters)")
    one.add_argument("--y", type=float, default=0.0, help="y_left_m (left+, meters)")
    one.add_argument("--conf", type=float, default=0.85, help="confidence (0..1)")
    one.add_argument("--count", type=int, default=1, help="how many times to send")
    one.add_argument("--interval", type=float, default=0.25, help="seconds between sends")

    app = sub.add_parser("approach", help="Linearly move the cone from (x0,y0) to (x1,y1).")
    app.add_argument("--x0", type=float, default=3.0)
    app.add_argument("--y0", type=float, default=0.0)
    app.add_argument("--x1", type=float, default=0.8)
    app.add_argument("--y1", type=float, default=0.0)
    app.add_argument("--steps", type=int, default=8)
    app.add_argument("--conf", type=float, default=0.85)
    app.add_argument("--interval", type=float, default=0.25)

    csvm = sub.add_parser("csv", help="Read x,y,conf[,delay_s] rows from a CSV file (no header by default).")
    csvm.add_argument("file", help="path to CSV")
    csvm.add_argument("--has-header", action="store_true", help="first row is header with x_fwd_m,y_left_m,confidence,delay_s")
    csvm.add_argument("--default-conf", type=float, default=0.85)
    csvm.add_argument("--default-delay", type=float, default=0.25)

    args = p.parse_args()
    if not args.mode:
        args.mode = "single"  # default

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        if args.mode == "single":
            # Defaults (x=1.2,y=0.0) satisfy your listener's CONE_NEAR gate (0.3 <= r <= 2.0)
            for _ in range(args.count):
                send(sock, args.host, args.port, args.x, args.y, args.conf)
                time.sleep(args.interval)

        elif args.mode == "approach":
            for (x, y) in iter_approach(args.x0, args.y0, args.x1, args.y1, args.steps):
                send(sock, args.host, args.port, x, y, args.conf)
                time.sleep(args.interval)

        elif args.mode == "csv":
            with open(args.file, "r", newline="") as f:
                r = csv.reader(f)
                header = None
                if args.has_header:
                    header = next(r, None)
                for row in r:
                    if not row:
                        continue
                    if header:
                        # header names: x_fwd_m,y_left_m,confidence,delay_s
                        rowmap = dict(zip(header, row))
                        x = float(rowmap.get("x_fwd_m"))
                        y = float(rowmap.get("y_left_m"))
                        conf = float(rowmap.get("confidence", args.default-conf))  # fallback if missing
                        delay = float(rowmap.get("delay_s", args.default_delay))
                    else:
                        # positional: x, y, [conf], [delay]
                        x = float(row[0]); y = float(row[1])
                        conf = float(row[2]) if len(row) > 2 and row[2] != "" else args.default_conf
                        delay = float(row[3]) if len(row) > 3 and row[3] != "" else args.default_delay
                    send(sock, args.host, args.port, x, y, conf)
                    time.sleep(delay)
        else:
            print("Unknown mode", file=sys.stderr)
            sys.exit(2)
    finally:
        sock.close()

if __name__ == "__main__":
    main()
