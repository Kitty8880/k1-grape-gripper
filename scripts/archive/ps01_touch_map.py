#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PS01 触摸图 —— 用平滑后的数值直观显示"按在哪里、用力多大"。

为什么要这个工具
----------------
PS01 的原始噪声 σ ≈ 4,500 计数（0.66%），而 0.8N 的力响应只有约 1,250 计数。
如果直接显示瞬时值，数字会一直乱跳，肉眼根本看不出按压效果。

本工具的做法：
    ① 先在若干秒内测出基线（平均）
    ② 之后按固定时间窗（默认 0.3s）对读数做平均，再显示
    ③ 用横条直观显示偏离基线的幅度

这样按压效果一眼就能看出来。

用法::

    sudo python3 scripts/ps01_touch_map.py
    sudo python3 scripts/ps01_touch_map.py --baseline 3 --window 0.3
    sudo python3 scripts/ps01_touch_map.py --channel 1
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

try:
    from grape_gripper.ps01 import Ps01Error, Ps01Reader
except ImportError as exc:
    print(f"❌ 导入失败：{exc}")
    sys.exit(1)


def main() -> int:
    ap = argparse.ArgumentParser(description="PS01 触摸图")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--channel", type=int, default=1, help="用哪个 PD 通道（默认 PD1）")
    ap.add_argument("--baseline", type=float, default=3.0, help="基线采集时长（秒）")
    ap.add_argument("--window", type=float, default=0.3, help="平滑窗口（秒）")
    ap.add_argument("--bar-scale", type=float, default=400.0,
                    help="横条每格代表多少计数")
    ap.add_argument("--sensor-hz", type=float, default=160.0)
    args = ap.parse_args()

    ch = args.channel
    try:
        r = Ps01Reader(args.port)
        r.open()
    except Ps01Error as exc:
        print(f"❌ {exc}")
        return 1
    if not r.wait_first_sample(timeout=3.0):
        print(f"❌ 无数据：{r.stats()}")
        r.close()
        return 1

    print("=" * 74)
    print(f"PS01 触摸图   PD{ch}")
    print("=" * 74)

    # ---------- 基线 ----------
    print(f"采集基线（{args.baseline:.0f} 秒）—— 现在不要碰传感器")
    vals = []
    t0 = time.time()
    while time.time() - t0 < args.baseline:
        v = r.channel(ch)
        if v is not None:
            vals.append(float(v))
        time.sleep(0.002)
    if not vals:
        print("❌ 基线采集失败")
        r.close()
        return 1
    base = statistics.mean(vals)
    base_sd = statistics.pstdev(vals)
    print(f"  基线 PD{ch} = {base:,.0f}   原始噪声 σ = {base_sd:,.0f} "
          f"（{base_sd/base*100:.2f}%）")
    print(f"  平滑窗口 {args.window:.2f}s（约 {int(args.sensor_hz*args.window)} 个独立样本）")
    print(f"  → 平滑后噪声 σ ≈ {base_sd/max(1,(args.sensor_hz*args.window))**0.5:,.0f} 计数")
    print()
    print("现在开始用手指按压不同位置，观察下面的横条")
    print("Ctrl+C 停止")
    print()
    print(f"{'时间':>7} {'平滑值':>12} {'Δ':>10}  {'强度':<34}")
    print("-" * 74)

    t0 = time.time()
    peak = (0.0, 0.0)
    try:
        while True:
            # 在一个窗口内密集采样后取平均
            acc = []
            tw = time.time()
            while time.time() - tw < args.window:
                v = r.channel(ch)
                if v is not None:
                    acc.append(float(v))
                time.sleep(0.002)
            if not acc:
                continue
            smooth = statistics.mean(acc)
            dev = smooth - base

            if abs(dev) > abs(peak[0]):
                peak = (dev, time.time() - t0)

            bar_len = int(min(34, abs(dev) / args.bar_scale))
            bar = ("█" * bar_len) if dev >= 0 else ("▒" * bar_len)
            mark = ""
            if abs(dev) > 3 * base_sd / max(1, (args.sensor_hz * args.window)) ** 0.5:
                mark = "  ← 明显"

            print(f"{time.time()-t0:6.1f}s {smooth:>12,.0f} {dev:>+10,.0f}  "
                  f"{bar:<34}{mark}")

    except KeyboardInterrupt:
        print("\n中断")
    finally:
        r.close()

    print()
    print("=" * 74)
    print("结果")
    print("=" * 74)
    print(f"  基线 PD{ch}     = {base:,.0f}")
    print(f"  最大偏离        = {peak[0]:>+,.0f} 计数"
          f"（出现在 {peak[1]:.1f}s）")
    print()
    if abs(peak[0]) < 1:
        print("  ❌ 全程没有明显变化 —— 按压没有传到传感器受力面")
    else:
        k = 1564.0     # PD1 的灵敏度（计数/牛顿），来自牛奶盒标定
        print(f"  按灵敏度 {k:,.0f} 计数/N 折算，这个变化相当于 "
              f"{abs(peak[0])/k:.2f} N")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())