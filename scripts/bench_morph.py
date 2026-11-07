#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测形态学的真实成本，找出能降到 50 ms 的组合。"""
import glob
import sys
import time

sys.path.insert(0, "/home/daiweikai/grape-gripper/src")
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

import cv2
import numpy as np

p = sorted(glob.glob("/tmp/gb2/*.jpg"))[0]
bgr0 = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def bench(fn, n=3):
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    return min(ts)


for scale in (1.0, 0.5, 0.25):
    bgr = bgr0 if scale == 1.0 else cv2.resize(
        bgr0, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = bgr.shape[:2]
    print(f"\n=== {w}x{h}（缩放 {scale}）===")

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0].astype(np.int16)
    S = hsv[:, :, 1].astype(np.int16)
    V = hsv[:, :, 2].astype(np.int16)

    t = bench(lambda: cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV))
    print("  cvtColor(HSV)              %7.1f ms" % t)

    m_green = ((H >= 18) & (H <= 55) & (S >= 55) & (V >= 25)
               & (V <= 250)).astype(np.uint8) * 255
    fg = int((m_green > 0).sum())
    print("  绿色掩码前景                %7d px" % fg)

    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k11e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    k11r = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    k7e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    t = bench(lambda: cv2.morphologyEx(m_green, cv2.MORPH_CLOSE, k5,
                                       iterations=2))
    print("  CLOSE 5x5椭圆 ×2           %7.1f ms" % t)

    t = bench(lambda: cv2.morphologyEx(m_green, cv2.MORPH_OPEN, k11e))
    print("  OPEN 11x11椭圆             %7.1f ms   ← 原方案" % t)

    t = bench(lambda: cv2.morphologyEx(m_green, cv2.MORPH_OPEN, k11r))
    print("  OPEN 11x11矩形             %7.1f ms   ← 可分离，应该快很多" % t)

    t = bench(lambda: cv2.morphologyEx(m_green, cv2.MORPH_OPEN, k7e))
    print("  OPEN 7x7椭圆               %7.1f ms" % t)

    # 组合方案：矩形开 + 椭圆闭
    def combo():
        m = cv2.morphologyEx(m_green, cv2.MORPH_CLOSE, k5, iterations=2)
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, k11r)
    t = bench(combo)
    print("  组合（闭5x5×2 + 开11x11矩形）%7.1f ms" % t)