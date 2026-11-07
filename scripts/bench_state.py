#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测 grape_state.detect 的真实耗时 —— 决定实时循环能跑多少 fps。

为什么要测
----------
"实时视觉监测"要求每帧都判状态。1080p 全分辨率直接跑肯定慢，
但缩到多少才既够快、又够准？这必须先量出来，不能猜。

同时验证一件事：**降采样会不会破坏扁平度判据**？
如果降采样后三档的扁平度分布明显变宽，那说明细节丢了，不能降太多。

用法::

    sudo python3 scripts/bench_state.py /tmp/ga2 /tmp/gb2 /tmp/gc2
"""
from __future__ import annotations

import glob
import os
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
from grape_state import GrapeStateDetector, STATES, FLAT_FREE_MAX, FLAT_TOUCH_MAX


def imread_u(p):
    b = np.fromfile(p, dtype=np.uint8)
    if b.size == 0:
        return None
    return cv2.imdecode(b, cv2.IMREAD_COLOR)


def main() -> int:
    dirs = sys.argv[1:] or ["/tmp/ga2", "/tmp/gb2", "/tmp/gc2"]
    if len(dirs) != 3:
        print("❌ 需要 3 个目录")
        return 1

    imgs = []          # [(label, bgr)]
    for label, d in zip(STATES, dirs):
        for p in sorted(glob.glob(os.path.join(d, "*.jpg")))[:5]:
            b = imread_u(p)
            if b is not None:
                imgs.append((label, b))
    if not imgs:
        print("❌ 没有图片")
        return 1

    print("=" * 74)
    print("状态识别耗时 vs 降采样比例")
    print("=" * 74)
    print(f"  样本 {len(imgs)} 张，原始 {imgs[0][1].shape[1]}x{imgs[0][1].shape[0]}")
    print()
    print("  %-8s %-12s %-10s %-12s %s"
          % ("缩放", "分辨率", "单帧 ms", "可用 fps", "三档准确率"))

    det = GrapeStateDetector()
    best = None
    for scale in (1.0, 0.75, 0.5, 0.375, 0.25):
        times = []
        correct = 0
        total = 0
        for label, bgr in imgs:
            if scale == 1.0:
                im = bgr
            else:
                im = cv2.resize(bgr, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_AREA)
            t0 = time.perf_counter()
            r = det.detect(im)
            dt = (time.perf_counter() - t0) * 1000
            times.append(dt)
            if r["ok"]:
                total += 1
                if r["state"] == label:
                    correct += 1
        med = statistics.median(times)
        h, w = im.shape[:2]
        acc = 100.0 * correct / total if total else 0.0
        fps = 1000.0 / med if med > 0 else 0
        flag = "  ← 最快可用" if (fps >= 8 and best is None) else ""
        print("  %-8.3f %-12s %-10.1f %-12.1f %.1f%% (%d/%d)%s"
              % (scale, "%dx%d" % (w, h), med, fps, acc, correct, total, flag))
        if fps >= 8 and best is None:
            best = (scale, med, fps, acc)

    print()
    print("=" * 74)
    print("结论")
    print("=" * 74)
    if best:
        print("  ★ 推荐实时用缩放 %.3f（单帧 %.1f ms，约 %.1f fps，准确率 %.1f%%）"
              % best)
    else:
        print("  ⚠️ 所有比例都低于 8 fps，实时循环要降低判定频率（比如每 3 帧判一次）")

    # 降采样对扁平度分布的影响（看判据是否还成立）
    print()
    print("  降采样后三档扁平度中位（看单调性还在不在）：")
    for scale in (1.0, 0.5, 0.25):
        line = "    缩放 %.3f：" % scale
        for label, d in zip(STATES, dirs):
            vals = []
            for p in sorted(glob.glob(os.path.join(d, "*.jpg"))):
                b = imread_u(p)
                if b is None:
                    continue
                im = b if scale == 1.0 else cv2.resize(
                    b, None, fx=scale, fy=scale,
                    interpolation=cv2.INTER_AREA)
                r = det.detect(im)
                if r["ok"]:
                    vals.append(r["flatness"])
            line += "  %s=%.4f" % (label, statistics.median(vals) if vals else 0)
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())