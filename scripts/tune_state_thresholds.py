#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新标定 grape_state 的阈值 —— 用 GrapeStateDetector 自己的扁平度值。

为什么需要这一步
----------------
eval_3class.py 找出的阈值（0.144 / 0.161）是用"另一条路径"算的扁平度，
而 grape_state.py 内部算出来的值有极微小的差异（浮点/轮廓取法），
结果 0.144 正好压在一个自由状态样本上，导致 8/10 变成 7/10。

所以这里直接用**模块本身**的输出重新标定，并采用"**取相邻样本值的中点**"
作为阈值 —— 这样每一侧都留出最大余量，不会卡在某个样本上。

用法::

    sudo python3 scripts/tune_state_thresholds.py /tmp/ga2 /tmp/gb2 /tmp/gc2
"""
from __future__ import annotations

import glob
import os
import statistics
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
from grape_state import GrapeStateDetector, STATES

EPS_LIST = [0.010, 0.012, 0.015, 0.018, 0.020, 0.025]


def imread_u(p):
    b = np.fromfile(p, dtype=np.uint8)
    if b.size == 0:
        return None
    return cv2.imdecode(b, cv2.IMREAD_COLOR)


def flat_values(det, dirs):
    """返回 [(label, flatness, filename)]。"""
    rows = []
    for label, d in zip(STATES, dirs):
        for p in sorted(glob.glob(os.path.join(d, "*.jpg"))):
            bgr = imread_u(p)
            if bgr is None:
                continue
            r = det.detect(bgr)
            if not r["ok"]:
                continue
            rows.append((label, r["flatness"], os.path.basename(p)))
    return rows


def best_thresholds(rows):
    """
    穷举两个切点，最大化准确率；并列时选"最小间隔最大"的那组（更鲁棒）。
    """
    vals = sorted({v for _, v, _ in rows})
    cands = [(vals[i] + vals[i + 1]) / 2.0 for i in range(len(vals) - 1)]
    best = None
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            t1, t2 = cands[i], cands[j]
            ok = 0
            for lb, v, _ in rows:
                pred = 0 if v < t1 else (1 if v < t2 else 2)
                if STATES[pred] == lb:
                    ok += 1
            acc = ok / len(rows)
            margin = min(t1 - min(vals), t2 - t1)
            key = (acc, margin)
            if best is None or key > best[0]:
                best = (key, t1, t2, ok)
    return best


def main() -> int:
    dirs = sys.argv[1:] or ["/tmp/ga2", "/tmp/gb2", "/tmp/gc2"]
    if len(dirs) != 3:
        print("❌ 需要 3 个目录")
        return 1

    print("=" * 74)
    print("grape_state 阈值标定（逐 eps 搜索）")
    print("=" * 74)
    print("  %-8s %-10s %-10s %-10s %-8s %s"
          % ("eps", "自由中位", "贴住中位", "夹紧中位", "阈值1", "阈值2   准确率"))

    best_all = None
    for eps in EPS_LIST:
        det = GrapeStateDetector(eps=eps)
        rows = flat_values(det, dirs)
        if len(rows) < 10:
            continue
        bt = best_thresholds(rows)
        if bt is None:
            continue
        (acc, _margin), t1, t2, ok = bt
        med = {}
        for lb in STATES:
            vs = [v for l, v, _ in rows if l == lb]
            med[lb] = statistics.median(vs) if vs else float("nan")
        print("  %-8.3f %-10.4f %-10.4f %-10.4f %-8.4f %.4f   %5.1f%% (%d/%d)"
              % (eps, med["自由"], med["贴住"], med["夹紧"], t1, t2,
                 acc * 100, ok, len(rows)))
        if best_all is None or acc > best_all[0]:
            best_all = (acc, eps, t1, t2, rows)

    if best_all is None:
        print("没有可用数据")
        return 1

    acc, eps, t1, t2, rows = best_all
    print()
    print("=" * 74)
    print("推荐参数")
    print("=" * 74)
    print(f"  FLAT_EPS       = {eps:.3f}")
    print(f"  FLAT_FREE_MAX  = {t1:.4f}    （小于此值 → 自由）")
    print(f"  FLAT_TOUCH_MAX = {t2:.4f}    （小于此值 → 贴住，否则 → 夹紧）")
    print(f"  预期准确率      = {acc*100:.1f}%")

    # 用推荐参数复算混淆矩阵
    det = GrapeStateDetector(free_max=t1, touch_max=t2, eps=eps)
    cm = {a: {b: 0 for b in STATES} for a in STATES}
    for lb, _v, _n in rows:
        pass
    for f in [r[2] for r in rows]:
        pass
    # 直接按已算出的 flatness 判定（避免重复读图）
    for lb, v, _n in rows:
        pred = "自由" if v < t1 else ("贴住" if v < t2 else "夹紧")
        cm[lb][pred] += 1
    print()
    print("  混淆矩阵：")
    print("    真实\\预测   " + "  ".join("%6s" % s for s in STATES) + "   正确率")
    for lb in STATES:
        row = cm[lb]
        tot = sum(row.values())
        print("    %-10s  %s   %5.1f%%"
              % (lb, "  ".join("%6d" % row[s] for s in STATES),
                 100.0 * row[lb] / tot if tot else 0))

    # 误判样本明细，便于判断是不是检测抖动
    print()
    print("  误判明细：")
    for lb, v, n in rows:
        pred = "自由" if v < t1 else ("贴住" if v < t2 else "夹紧")
        if pred != lb:
            print("    %-34s 真实 %s  判成 %s  扁平度 %.4f"
                  % (n[:34], lb, pred, v))
    return 0


if __name__ == "__main__":
    sys.exit(main())