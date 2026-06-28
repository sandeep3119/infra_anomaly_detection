"""
Fire simulated telemetry at the inference service.

- Sends realistic "normal" readings (Gaussian around training-like centers)
  so the drift sample store characterizes a real distribution.
- Spreads traffic across a few devices so each warms past the 61-reading gate.

Usage:
    python scripts/simulate_traffic.py                 # 300 normal requests
    python scripts/simulate_traffic.py --n 500
    python scripts/simulate_traffic.py --drift          # shift readLatencyMs up to test drift
"""

import argparse
import random
import time

import requests

URL = "http://localhost:8000/predict"
DEVICES = ["node-01", "node-02", "node-03"]

# (mean, std) per raw feature — centers loosely match the training data ranges.
# (mean, std) per raw feature — taken from the training data's actual
# distribution (df.describe()) so "normal" traffic matches the reference.
NORMAL = {
    "userDataReadIops":  (5046.0, 718.0),
    "userDataWriteIops": (3029.0, 439.0),
    "readLatencyMs":     (1.99, 0.50),
    "writeLatencyMs":    (3.00, 0.50),
    "cpuPercent":        (40.0, 5.0),
    "memoryPercent":     (60.0, 5.0),
}


def make_reading(drift: bool):
    reading = {}
    for field, (mean, std) in NORMAL.items():
        # Inject drift by shifting readLatencyMs well outside its normal range.
        if drift and field == "readLatencyMs":
            mean = mean + 10.0
        reading[field] = round(random.gauss(mean, std), 3)
    return reading


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300, help="number of requests")
    parser.add_argument("--drift", action="store_true", help="shift readLatencyMs to simulate drift")
    parser.add_argument("--sleep", type=float, default=0.05, help="delay between requests (s)")
    args = parser.parse_args()

    counts = {}
    for i in range(args.n):
        device = random.choice(DEVICES)
        payload = {"deviceID": device, **make_reading(args.drift)}
        try:
            resp = requests.post(URL, json=payload, timeout=5)
            code = resp.status_code
        except requests.RequestException as e:
            code = f"ERR ({e.__class__.__name__})"
        counts[code] = counts.get(code, 0) + 1
        print(f"{i:>4}  {device}  {code}")
        time.sleep(args.sleep)

    print("\nstatus summary:", counts)


if __name__ == "__main__":
    main()
