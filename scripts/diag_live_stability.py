#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断实时流下的检测稳定性：同一场景连续取帧，看每帧的检测是否一致。

要区分两种情况：
  A) 同一个葡萄，只是扁平度小幅抖动  → 用时间中值滤波就能解决
  B) 每帧测到的是不同的连通域（时而是葡萄、时而是导线）
                                      → 必须先在检测层面稳定下来

输出每帧的：颜色类、面积、质心、圆度、扁平度、判定的状态。
"""
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

import cv2
import numpy as np
from k1_camera_stream import K1CameraStream
from grape_state import GrapeStateDetector

print("打开连续视频流…", flush=True)
cam = K1CameraStream(width=1920, height=1080, use_sudo=False, verbose=False)
cam.open()
det = GrapeStateDetector(eps=0.015)

rows = []
N = 16
print()
print("  %-3s %-7s %-9s %-16s %-7s %-8s %-7s %s"
      % ("帧", "颜色类", "面积%", "质心", "圆度", "扁平度", "状态", "亮度均值"))
try:
    for i in range(N):
        bgr = cam.read_frame(timeout=5.0)
        if bgr is None:
            print("  取帧超时")
            continue
        t = time.perf_counter()
        r = det.detect(bgr)
        dt = (time.perf_counter() - t) * 1000
        gray = float(bgr.mean())
        if not r.get("ok"):
            print("  %-3d %-7s %-9s %-16s %-7s %-8s %-7s %.1f"
                  % (i + 1, "--", "--", "--", "--", "--", "未检出", gray))
            rows.append(None)
            continue
        m = r["metrics"]
        cx, cy = m["centroid"]
        print("  %-3d %-7s %-9.2f (%5.0f,%5.0f)   %-7.3f %-8.4f %-7s %.1f"
              % (i + 1, m["cls"], m["frac"] * 100, cx, cy, m["circ"],
                 r["flatness"], r["state"], gray), flush=True)
        rows.append(r)
finally:
    cam.close()

ok = [r for r in rows if r]
print()
if ok:
    flats = [r["flatness"] for r in ok]
    cents = [r["metrics"]["centroid"] for r in ok]
    xs = [c[0] for c in cents]
    ys = [c[1] for c in cents]
    print("  有效 %d/%d 帧" % (len(ok), len(rows)))
    print("  扁平度：中位 %.4f  范围 %.4f~%.4f  标准差 %.4f"
          % (statistics.median(flats), min(flats), max(flats),
             statistics.pstdev(flats)))
    print("  质心 x：范围 %.0f~%.0f（跨度 %.0f px）"
          % (min(xs), max(xs), max(xs) - min(xs)))
    print("  质心 y：范围 %.0f~%.0f（跨度 %.0f px）"
          % (min(ys), max(ys), max(ys) - min(ys)))
    states = [r["state"] for r in ok]
    from collections import Counter
    print("  状态分布：%s" % dict(Counter(states)))
    classes = [r["metrics"]["cls"] for r in ok]
    print("  颜色类分布：%s" % dict(Counter(classes)))
    print()
    if max(xs) - min(xs) > 60 or max(ys) - min(ys) > 60:
        print("  ★ 结论：质心跨度很大 → 测到了**不同的连通域**（情况 B）")
        print("    需要在检测层面固定目标（比如锁定首次检测到的位置附近）")
    elif statistics.pstdev(flats) > 0.02:
        print("  ★ 结论：同一个葡萄但扁平度抖动大 → 用**时间中值滤波**（情况 A）")
    else:
        print("  ★ 结论：稳定，可以放心用")