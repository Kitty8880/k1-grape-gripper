#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""拆解 detect() 的各阶段耗时，找出 3 秒花在哪。"""
import glob
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
from grape_state import (color_masks, longest_segment_ratio,
                         MIN_AREA_FRAC, MAX_AREA_FRAC, MIN_ASPECT, MIN_CIRC)

p = sorted(glob.glob("/tmp/gb2/*.jpg"))[0]
bgr = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
h, w = bgr.shape[:2]
print(f"  测试图 {w}x{h}\n")

# ① HSV 转换 + 三张掩码
t = time.perf_counter()
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
t_cvt = (time.perf_counter() - t) * 1000

t = time.perf_counter()
ms = color_masks(bgr)
t_masks = (time.perf_counter() - t) * 1000
print("  %-34s %8.1f ms   （三个颜色类 + 形态学）" % ("color_masks 总计", t_masks))
for k, m in ms.items():
    print("      %-10s 前景 %7d px" % (k, int((m > 0).sum())))

# ② 逐类统计连通域数量
t = time.perf_counter()
total_comp = 0
info = []
for cls, m in ms.items():
    n, labels, stats, cents = cv2.connectedComponentsWithStats(m, 8)
    total_comp += (n - 1)
    passed = 0
    for idx in range(1, n):
        a = int(stats[idx, cv2.CC_STAT_AREA])
        if a < w * h * MIN_AREA_FRAC or a > w * h * MAX_AREA_FRAC:
            continue
        bw = int(stats[idx, cv2.CC_STAT_WIDTH])
        bh = int(stats[idx, cv2.CC_STAT_HEIGHT])
        if bw <= 0 or bh <= 0:
            continue
        if min(bw, bh) / max(bw, bh) < MIN_ASPECT:
            continue
        passed += 1
    info.append((cls, n - 1, passed))
t_cc = (time.perf_counter() - t) * 1000
print("\n  %-34s %8.1f ms" % ("连通域统计（三类）", t_cc))
for cls, n, passed in info:
    print("      %-10s 总 %4d 个连通域，通过面积/长宽比 %3d 个" % (cls, n, passed))

# ③ 关键怀疑点：每个通过筛选的连通域都要做全尺寸 blob*255 + findContours
t = time.perf_counter()
n_calls = 0
for cls, m in ms.items():
    n, labels, stats, cents = cv2.connectedComponentsWithStats(m, 8)
    for idx in range(1, n):
        a = int(stats[idx, cv2.CC_STAT_AREA])
        if a < w * h * MIN_AREA_FRAC or a > w * h * MAX_AREA_FRAC:
            continue
        bw = int(stats[idx, cv2.CC_STAT_WIDTH])
        bh = int(stats[idx, cv2.CC_STAT_HEIGHT])
        if bw <= 0 or bh <= 0:
            continue
        if min(bw, bh) / max(bw, bh) < MIN_ASPECT:
            continue
        blob = (labels == idx).astype(np.uint8)      # ← 全尺寸比较！
        cnts, _ = cv2.findContours(blob * 255, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)   # ← 全尺寸查找！
        n_calls += 1
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            cv2.arcLength(c, True)
t_full = (time.perf_counter() - t) * 1000
print("\n  %-34s %8.1f ms   ← ★ 嫌疑最大（%d 次全尺寸操作）"
      % ("全尺寸 blob+findContours", t_full, n_calls))

# ④ 对照：只用包围盒裁剪
t = time.perf_counter()
for cls, m in ms.items():
    n, labels, stats, cents = cv2.connectedComponentsWithStats(m, 8)
    for idx in range(1, n):
        a = int(stats[idx, cv2.CC_STAT_AREA])
        if a < w * h * MIN_AREA_FRAC or a > w * h * MAX_AREA_FRAC:
            continue
        x, y, bw, bh = (int(stats[idx, cv2.CC_STAT_LEFT]),
                        int(stats[idx, cv2.CC_STAT_TOP]),
                        int(stats[idx, cv2.CC_STAT_WIDTH]),
                        int(stats[idx, cv2.CC_STAT_HEIGHT]))
        if bw <= 0 or bh <= 0:
            continue
        if min(bw, bh) / max(bw, bh) < MIN_ASPECT:
            continue
        sub = (labels[y:y + bh, x:x + bw] == idx).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            cv2.arcLength(c, True)
t_crop = (time.perf_counter() - t) * 1000
print("  %-34s %8.1f ms   ← 改成包围盒裁剪"
      % ("局部 blob+findContours", t_crop))

print()
print("  ★ 预计优化收益：%.1f ms → %.1f ms，快 %.1f 倍"
      % (t_masks + t_full, t_masks + t_crop,
         (t_masks + t_full) / max(t_masks + t_crop, 1e-6)))