#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PS01 按压响应测试 —— 直接告诉你"按压有没有引起可检测的变化"。

流程：
    ① 建立基线（默认 3 秒，期间不要碰传感器）
    ② 记录按压响应（默认 6 秒，期间持续按住）
    ③ 对比，输出每个通道的最大偏差、出现时刻、是否显著

这样不用翻几百行原始数据，一眼就能判断传感器是否真的感知到了力。

用法::

    sudo python3 scripts/ps01_press_test.py
    sudo python3 scripts/ps01_press_test.py --baseline 3 --record 6
    sudo python3 scripts/ps01_press_test.py --all      # 顺便显示 PD3
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

# 把项目的 src/ 加进搜索路径（无论从哪个目录调用都成立）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

try:
    from grape_gripper.ps01 import Ps01Error, Ps01Reader
except ImportError as exc:
    print(f"❌ 导入失败：{exc}")
    print("   请从项目根目录运行： cd ~/grape-gripper && sudo python3 scripts/ps01_press_test.py")
    sys.exit(1)


def main() -> int:
    ap = argparse.ArgumentParser(description="PS01 按压响应测试")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baseline", type=float, default=3.0, help="基线时长（秒）")
    ap.add_argument("--record", type=float, default=6.0, help="记录时长（秒）")
    ap.add_argument("--bucket", type=float, default=0.25, help="统计分桶时长（秒）")
    ap.add_argument("--all", action="store_true", help="同时显示 PD3")
    args = ap.parse_args()

    print("=" * 74)
    print("PS01 按压响应测试")
    print("=" * 74)

    try:
        reader = Ps01Reader(args.port)
        reader.open()
    except Ps01Error as exc:
        print(f"❌ {exc}")
        return 1

    if not reader.wait_first_sample(timeout=3.0):
        print(f"❌ 没有数据。统计：{reader.stats()}")
        reader.close()
        return 1

    print(f"✅ {reader.describe()}")
    print()

    def sample_bucket(duration: float) -> dict:
        """在 duration 时间内按 bucket 分桶，返回每桶各通道均值。"""
        buckets = []
        t0 = time.time()
        cur = {1: [], 2: [], 3: []}
        bucket_start = t0
        while time.time() - t0 < duration:
            for ch in (1, 2, 3):
                v = reader.channel(ch)
                if v is not None:
                    cur[ch].append(v)
            if time.time() - bucket_start >= args.bucket:
                bucket = {"t": time.time() - t0}
                for ch in (1, 2, 3):
                    bucket[ch] = statistics.mean(cur[ch]) if cur[ch] else None
                buckets.append(bucket)
                cur = {1: [], 2: [], 3: []}
                bucket_start = time.time()
            time.sleep(0.002)
        return {"buckets": buckets}

    # ---------- 阶段一：基线 ----------
    print(f"【阶段 1】建立基线（{args.baseline:.0f} 秒）")
    print("          ⚠️ 现在不要碰传感器")
    for i in range(3, 0, -1):
        print(f"          {i}...", end="\r", flush=True)
        time.sleep(1)
    print("          开始采集基线...        ")

    base = sample_bucket(args.baseline)
    base_vals = {1: [], 2: [], 3: []}
    for b in base["buckets"]:
        for ch in (1, 2, 3):
            if b[ch] is not None:
                base_vals[ch].append(b[ch])

    base_mean, base_std = {}, {}
    for ch in (1, 2, 3):
        if base_vals[ch]:
            base_mean[ch] = statistics.mean(base_vals[ch])
            base_std[ch] = statistics.pstdev(base_vals[ch]) or 1.0
        else:
            base_mean[ch] = base_std[ch] = None

    print()
    print("  基线：")
    for ch in (1, 2, 3):
        if base_mean[ch] is None:
            print(f"    PD{ch}  无数据")
        else:
            print(f"    PD{ch}  均值 = {base_mean[ch]:>10,.0f}   "
                  f"标准差 = {base_std[ch]:>8,.0f}   "
                  f"（噪声 {base_std[ch]/base_mean[ch]*100:.2f}%）")

    # ---------- 阶段二：按压 ----------
    print()
    print(f"【阶段 2】记录按压响应（{args.record:.0f} 秒）")
    print("          ⚠️ 现在持续按住传感器，不要松开！")
    for i in range(3, 0, -1):
        print(f"          {i}...", end="\r", flush=True)
        time.sleep(1)
    print("          开始记录...              ")

    rec = sample_bucket(args.record)

    # ---------- 对比 ----------
    print()
    header = f"{'时间':>7}  {'PD1':>12} {'ΔPD1':>9}  {'PD2':>12} {'ΔPD2':>9}"
    if args.all:
        header += f"  {'PD3':>12} {'ΔPD3':>9}"
    print(header)
    print("-" * len(header))

    max_dev = {1: (0.0, 0.0), 2: (0.0, 0.0), 3: (0.0, 0.0)}   # ch -> (偏差, 时刻)
    for b in rec["buckets"]:
        row = f"{b['t']:6.2f}s"
        for ch in (1, 2) if not args.all else (1, 2, 3):
            v = b[ch]
            if v is None or base_mean[ch] is None:
                row += f"  {'--':>12} {'--':>9}"
                continue
            dev = v - base_mean[ch]
            row += f"  {v:>12,.0f} {dev:>+9,.0f}"
            if abs(dev) > abs(max_dev[ch][0]):
                max_dev[ch] = (dev, b["t"])
        print(row)

    # ---------- 结论 ----------
    print()
    print("=" * 74)
    print("结论")
    print("=" * 74)
    any_signal = False
    for ch in (1, 2, 3):
        if base_mean[ch] is None:
            continue
        dev, t = max_dev[ch]
        if base_std[ch]:
            sigma = abs(dev) / base_std[ch]
        else:
            sigma = 0.0
        pct = dev / base_mean[ch] * 100
        verdict = "✅ 有响应" if sigma >= 5 else ("⚠️ 微弱" if sigma >= 3 else "❌ 无响应")
        if sigma >= 5:
            any_signal = True
        print(f"  PD{ch}：最大偏差 {dev:>+10,.0f}  "
              f"({pct:+.2f}%)  出现在 {t:.2f}s  "
              f"= {sigma:.1f} 倍基线噪声  → {verdict}")

    print()
    if any_signal:
        print("  ★ 传感器能感知到按压，可以做力闭环。")
        print("    下一步：把偏差方向/幅度换算成力，写入 config.yaml 的标定项。")
    else:
        print("  传感器在按压下没有明显响应。可能原因：")
        print("    1. 按的位置不对（PS01 是光学式，力要传到内部弹性体）")
        print("    2. 按压力度太小")
        print("    3. 传感器没有和被测物刚性连接")

    reader.close()
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())