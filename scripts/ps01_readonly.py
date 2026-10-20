#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""鍙涓嶅啓锛氱湅浼犳劅鍣ㄤ笂鐢靛悗杈撳嚭浠€涔堛€?""
import sys
import time
import serial

PORT = "/dev/ttyS10"
BAUD = 115200
SECONDS = 12.0

s = serial.Serial(PORT, BAUD, timeout=0.2,
                  bytesize=serial.EIGHTBITS,
                  parity=serial.PARITY_NONE,
                  stopbits=serial.STOPBITS_ONE)
s.reset_input_buffer()
print(f"璇诲彇 {SECONDS}s ...", flush=True)

t0 = time.time()
lines = 0
raw_bytes = 0
err_lines = 0
buf = ""
while time.time() - t0 < SECONDS:
    d = s.read(4096)
    if not d:
        continue
    raw_bytes += len(d)
    buf += d.decode("utf-8", "ignore")
    while "\n" in buf:
        line, buf = buf.split("\n", 1)
        line = line.strip()
        if not line:
            continue
        lines += 1
        if "ERROR" in line or "failed" in line:
            err_lines += 1
        if lines <= 60:
            print(f"  {line}")

s.close()
print()
print(f"鍏?{raw_bytes} 瀛楄妭 / {lines} 琛岋紝鍏朵腑閿欒琛?{err_lines}")
