#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
指标重复性检查 —— 判断"同状态内的抖动"会不会淹没"状态之间的差异"。

为什么这是关键
--------------
新方案的全部判据都建立在这几个几何量上：

    圆度 roundness = 4πA/P²
    实心度 solidity = 轮廓面积 / 凸包面积
    扁平度 flatness = 最长直边 / 周长
    边缘梯度 edge_grad = 葡萄边界处的平均梯度

**如果同一个状态拍 10 张，这些量每次跳 ±0.15，
而"自由 / 贴住 / 夹紧"三种状态之间只差 0.1，
那数据就没法用 —— 差异被噪声淹没。**

所以拿一组 10 张，先算**同状态内的标准差**，
再和预期的"状态间差异"比较。

判据：变异系数（σ/μ）< 5% 才算可重复。

用法::

    # 单组
    sudo python3 scripts/metrics_std.py --dir debug/set_a

    # 多组对比（组间差异 vs 组内抖动）
    sudo python3 scripts/metrics_std.py --dir debug/set_a --dir debug/set_b \\
        --dir debug/set_c
"""
from __future__ import annotations

import argparse
import glob
import os
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "apps"))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

import cv2
import numpy as np

# 直接复用 shot_web 的检测逻辑，保证"界面看到的"和"这里算的"完全一致
from shot_web import build_mask, analyze   # noqa: E402


def flatness_of(blob, eps_ratio: float = 0.02, min_seg_ratio: float = 0.06):
    """
    扁平度 = 最长直边 / 周长。

    物理依据：球被平板压住时，接触处变成一段**直线**。
    自由状态 flatness 小；被夹住时接触区变直，flatness 变大。

    ⚠️ 之前用"极坐标角度断口"是错的 —— 任何闭合轮廓
       自己的边界点必然覆盖 360°，这个量恒等于满。
    """
    cnts, _ = cv2.findContours(blob * 255, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 20:
        return None
    perim = cv2.arcLength(c, True)
    if perim < 50:
        return None
    approx = cv2.approxPolyDP(c, eps_ratio * perim, True).reshape(-1, 2)
    if len(approx) < 3:
        return None
    segs = []
    n = len(approx)
    for i in range(n):
        p1 = approx[i].astype(np.float32)
        p2 = approx[(i + 1) % n].astype(np.float32)
        d = p2 - p1
        segs.append(float(np.hypot(d[0], d[1])))
    return max(segs) / perim


def metrics_of(path):
    buf = np.fromfile(path, dtype=np.uint8)
    if buf.size == 0:
        return None
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    info, blob = analyze(bgr)
    if blob is None or info.get("frac") is None:
        return None
    fl = flatness_of(blob)
    return {
        "focus": info.get("focus"),
        "focus_grape": info.get("focus_grape"),
        "edge_grad": info.get("edge_w"),
        "frac": info.get("frac"),
        "roundness": info.get("circ"),
        "solidity": info.get("solidity"),
        "flatness": fl,
    }


KEYS = [
    ("focus_grape", "清晰度(葡萄)", 1.0),
    ("edge_grad", "边缘梯度", 1.0),
    ("frac", "占比%", 100.0),
    ("roundness", "圆度", 1.0),
    ("solidity", "实心度", 1.0),
    ("flatness", "扁平度", 1.0),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="指标重复性检查")
    ap.add_argument("--dir", action="append", required=True,
                    help="图片目录，可重复传多个做组间对比")
    args = ap.parse_args()

    groups = {}
    print("=" * 76)
    print("指标重复性检查")
    print("=" * 76)

    for d in args.dir:
        full = d if os.path.isabs(d) else os.path.join(_ROOT, d)
        files = sorted(glob.glob(os.path.join(full, "*.jpg")))
        files = [f for f in files if "_gap" not in f]
        if not files:
            print(f"  {d}: 没有图片")
            continue
        rows = []
        for f in files:
            m = metrics_of(f)
            if m:
                rows.append(m)
        groups[os.path.basename(full)] = rows
        print(f"  {d}: {len(rows)}/{len(files)} 张可用")
        if rows:
            for k, label, scale in KEYS:
                vals = [r[k] for r in rows if r[k] is not None]
                if not vals:
                    print(f"     {label:<12} --")
                    continue
                mean = statistics.mean(vals) * scale
                sd = (statistics.pstdev(vals) * scale) if len(vals) > 1 else 0.0
                cv = (sd / abs(mean) * 100) if mean else 0.0
                print("     %-12s 均值 %8.3f  标准差 %7.4f  变异系数 %5.1f%% %s"
                      % (label, mean, sd, cv,
                         "✅" if cv < 5 else ("🔶" if cv < 10 else "❌")))
        print()

    # 组间差异 vs 组内抖动
    names = [n for n, r in groups.items() if len(r) >= 2]
    if len(names) >= 2:
        print("=" * 76)
        print("组间差异 / 组内抖动（比值 > 3 才有区分力）")
        print("=" * 76)
        for k, label, scale in KEYS:
            stats = []
            for n in names:
                vals = [r[k] for r in groups[n] if r[k] is not None]
                if len(vals) >= 2:
                    stats.append((n, statistics.mean(vals) * scale,
                                  statistics.pstdev(vals) * scale))
            if len(stats) < 2:
                continue
            means = [s[1] for s in stats]
            pooled_sd = statistics.mean([s[2] for s in stats]) or 1e-9
            spread = max(means) - min(means)
            ratio = spread / pooled_sd
            print("  %-12s 组间跨度 %8.3f   组内σ %7.4f   比值 %5.1f  %s"
                  % (label, spread, pooled_sd, ratio,
                     "✅ 有区分力" if ratio > 3 else "❌ 分不开"))
        print()
        print("  各组均值：")
        for k, label, scale in KEYS:
            line = "    %-12s" % label
            for n in names:
                vals = [r[k] for r in groups[n] if r[k] is not None]
                line += "  %s=%7.3f" % (n.replace("set_", ""),
                                        statistics.mean(vals) * scale
                                        if vals else float("nan"))
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())