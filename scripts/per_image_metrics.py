#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐张打印指标，用于判断某一组数据是否被污染。"""
import glob
import os
import sys

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
from shot_web import analyze

for d in sys.argv[1:]:
    files = sorted(glob.glob(os.path.join(d, "*.jpg")), key=os.path.getmtime)
    print(f"\n=== {d}  ({len(files)} 张) ===")
    print("  %-26s %7s %7s %7s %7s %6s %7s" %
          ("文件", "清晰度", "边缘梯", "占比%", "圆度", "实心", "扁平"))
    for f in files:
        buf = np.fromfile(f, dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            print("  %-26s 读不出" % os.path.basename(f)[:26])
            continue
        info, blob = analyze(bgr)
        if blob is None:
            print("  %-26s ❌ 未检出" % os.path.basename(f)[:26])
            continue
        # 扁平度
        cnts, _ = cv2.findContours(blob * 255, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
        fl = float("nan")
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            perim = cv2.arcLength(c, True)
            if perim > 50:
                ap = cv2.approxPolyDP(c, 0.02 * perim, True).reshape(-1, 2)
                if len(ap) >= 3:
                    segs = []
                    n = len(ap)
                    for i in range(n):
                        p1 = ap[i].astype(np.float32)
                        p2 = ap[(i + 1) % n].astype(np.float32)
                        dd = p2 - p1
                        segs.append(float(np.hypot(dd[0], dd[1])))
                    fl = max(segs) / perim
        print("  %-26s %7.1f %7.1f %7.3f %7.3f %6.3f %7.3f" %
              (os.path.basename(f)[:26],
               info.get("focus_grape") or 0,
               info.get("edge_w") or 0,
               (info.get("frac") or 0) * 100,
               info.get("circ") or 0,
               info.get("solidity") or 0,
               fl))