#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""拆解视觉线程的耗时：取帧 vs 检测。"""
import statistics
import sys
import time

sys.path.insert(0, "/home/daiweikai/grape-gripper/src")
sys.path.insert(0, "/home/daiweikai/grape-gripper/scripts")
sys.path.insert(0, "/home/daiweikai/grape-gripper/apps")

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

from k1_camera_stream import K1CameraStream
from grape_state import GrapeStateDetector

print("打开连续视频流…", flush=True)
cam = K1CameraStream(width=1920, height=1080, use_sudo=False, verbose=False)
cam.open()
det = GrapeStateDetector()

read_ms, det_ms = [], []
N = 12
try:
    for i in range(N):
        t = time.perf_counter()
        bgr = cam.read_frame(timeout=5.0)
        t_read = (time.perf_counter() - t) * 1000
        if bgr is None:
            print("  取帧超时")
            continue
        t = time.perf_counter()
        r = det.detect(bgr)
        t_det = (time.perf_counter() - t) * 1000
        read_ms.append(t_read)
        det_ms.append(t_det)
        st = r.get("state") if r.get("ok") else "未检出"
        print("  帧%2d  取帧 %6.1f ms  检测 %6.1f ms  状态 %s"
              % (i + 1, t_read, t_det, st), flush=True)
finally:
    cam.close()

print()
if read_ms:
    print("  取帧中位 %.1f ms，检测中位 %.1f ms，合计 %.1f ms → %.1f fps"
          % (statistics.median(read_ms), statistics.median(det_ms),
             statistics.median(read_ms) + statistics.median(det_ms),
             1000.0 / (statistics.median(read_ms) + statistics.median(det_ms))))
print()
print("  取帧范围 %.0f~%.0f ms   检测范围 %.0f~%.0f ms"
      % (min(read_ms), max(read_ms), min(det_ms), max(det_ms)))