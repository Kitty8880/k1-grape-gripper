#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PS01 力标定 —— 用已知质量求「读数变化 → 牛顿」的对应关系。

方法
----
把已知质量的物体（推荐：1 元硬币，1 枚 ≈ 6.1 克 ≈ 0.06 N）依次放到传感器上，
每加一次记录一组读数。最后拟合出标定曲线。

输出可直接填进 config.yaml 的：
    sensor.calib.left.raw_zero / raw_span / force_max_n

用法::

    # 默认台阶：0 / 0.30 / 0.60 / 0.90 / 1.20 N（5/10/15/20 枚硬币）
    sudo python3 scripts/ps01_calib.py

    # 自定义台阶（单位：克）
    sudo python3 scripts/ps01_calib.py --steps 0,30.5,61,91.5,122

    # 每档采集时长
    sudo python3 scripts/ps01_calib.py --hold 4

    # 只用某个通道
    sudo python3 scripts/ps01_calib.py --channel 1
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

G = 9.80665


def main() -> int:
    ap = argparse.ArgumentParser(description="PS01 力标定")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--steps", default="0,30.5,61,91.5,122",
                    help="依次加载的质量（克），逗号分隔")
    ap.add_argument("--hold", type=float, default=3.5, help="每档采集时长（秒）")
    ap.add_argument("--settle", type=float, default=1.5, help="加载后等待稳定时间（秒）")
    ap.add_argument("--channels", default="1,2", help="使用哪些 PD 通道")
    args = ap.parse_args()

    masses = [float(x) for x in args.steps.split(",") if x.strip()]
    channels = [int(x) for x in args.channels.split(",") if x.strip()]

    print("=" * 78)
    print("PS01 力标定")
    print("=" * 78)

    try:
        reader = Ps01Reader(args.port)
        reader.open()
    except Ps01Error as exc:
        print(f"❌ {exc}")
        return 1

    if not reader.wait_first_sample(timeout=3.0):
        print(f"❌ 没有数据：{reader.stats()}")
        reader.close()
        return 1

    print(f"✅ {reader.describe()}")
    print()
    print("⚠️ 操作说明：")
    print("   · 用 1 元硬币（每枚约 6.1 克）或其它已知质量的小物件")
    print("   · 每档会先提示你放多少，放好后回车，脚本自动采集")
    print("   · 加载后请等它稳定再回车；不要用手压，手的力不可控")
    print()

    results = []   # [(mass_g, {ch: (mean, std)})]

    def collect(duration: float) -> dict:
        data = {ch: [] for ch in channels}
        t0 = time.time()
        while time.time() - t0 < duration:
            for ch in channels:
                v = reader.channel(ch)
                if v is not None:
                    data[ch].append(v)
            time.sleep(0.002)
        return {
            ch: (statistics.mean(data[ch]) if data[ch] else None,
                 statistics.pstdev(data[ch]) if len(data[ch]) > 1 else 0.0)
            for ch in channels
        }

    # ---------- 采集 ----------
    for mass in masses:
        force_n = mass / 1000.0 * G
        print("-" * 78)
        print(f"【{mass:.1f} g  =  {force_n:.3f} N】")
        if mass == 0:
            print("  请确保传感器上没有任何东西")
        else:
            print(f"  请放上 {mass:.1f} 克（约 {mass/6.1:.0f} 枚 1 元硬币）")
        input("  放好后按回车开始采集...")

        if mass > 0 and args.settle > 0:
            print(f"  等待 {args.settle:.1f}s 稳定...")
            time.sleep(args.settle)

        print(f"  采集中（{args.hold:.1f}s）...", end="", flush=True)
        r = collect(args.hold)
        print(" 完成")
        for ch in channels:
            m, s = r[ch]
            if m is not None:
                print(f"    PD{ch}  均值 = {m:>10,.0f}   标准差 = {s:>8,.0f}")

        # ★ 自检：和上一档比，读数有没有真的变化？
        if results:
            prev_mass, prev_r = results[-1]
            added_g = mass - prev_mass
            if added_g > 1.0:
                print(f"    ── 相比上一档（+{added_g:.1f}g）的变化 ──")
                suspicious = True
                for ch in channels:
                    pm = prev_r.get(ch, (None, 0))[0]
                    cm = r[ch][0]
                    if pm is None or cm is None:
                        continue
                    delta = cm - pm
                    noise = max(r[ch][1], prev_r.get(ch, (0, 0))[1])
                    ratio = abs(delta) / noise if noise else 0.0
                    flag = ""
                    if ratio >= 1.0:
                        flag = "  ✅ 有变化"
                        suspicious = False
                    elif ratio >= 0.5:
                        flag = "  ⚠️ 变化接近噪声"
                    else:
                        flag = "  ❌ 读不出变化"
                    print(f"       PD{ch}: Δ = {delta:>+9,.0f}   "
                          f"（{ratio:.2f} 倍噪声）{flag}")
                if suspicious:
                    print()
                    print("    ★★ 警告：加了重量但读数几乎没变！")
                    print("       可能原因：")
                    print("         1. 你放的是同一个物体（质量没变化）")
                    print("         2. 重量没有真正压在传感器的受力面上")
                    print("         3. 加载物不稳，晃动抵消了信号")
                    print("       本次标定结果不可信，请检查后重做。")
        results.append((mass, r))

    reader.close()

    # ---------- 分析 ----------
    print()
    print("=" * 78)
    print("标定结果")
    print("=" * 78)

    zero = results[0][1] if results else {}
    for ch in channels:
        z = zero.get(ch, (None, 0))[0]
        if z is None:
            continue
        print()
        print(f"── PD{ch} ──")
        print(f"{'质量(g)':>9} {'力(N)':>9} {'读数':>12} {'Δ读数':>11} "
              f"{'每牛顿变化':>12}")
        print("-" * 60)

        pairs = []
        for mass, r in results:
            m = r[ch][0]
            if m is None:
                continue
            f = mass / 1000.0 * G
            d = m - z
            per_n = d / f if f > 0 else 0.0
            print(f"{mass:>9.1f} {f:>9.3f} {m:>12,.0f} {d:>+11,.0f} "
                  f"{per_n:>12,.0f}")
            if f > 0:
                pairs.append((f, d))

        if len(pairs) >= 2:
            # 线性拟合（过原点约束的最小二乘）
            num = sum(f * d for f, d in pairs)
            den = sum(f * f for f, d in pairs)
            k = num / den if den else 0.0
            print()
            print(f"  线性拟合（过原点）： Δ读数 = {k:,.0f} × 力(N)")
            print(f"  即灵敏度约 {k:,.0f} 计数/牛顿")
            print()
            print("  填入 config.yaml：")
            print(f"    sensor.calib.left.raw_zero:  {z:,.0f}     ← 空载读数")
            print(f"    sensor.calib.left.raw_span:  {k * 10.0:,.0f}"
                  f"     ← 对应 force_max_n 的读数变化")
            print(f"    sensor.calib.left.force_max_n: 10.0")
            print(f"    sensor.calib.left.gamma: 1.0")
            print()
            print(f"  ★ 关键判断：目标力 0.8N 对应读数变化约 "
                  f"{k * 0.8:,.0f} 计数")
            noise = zero.get(ch, (None, 0))[1] or 1.0
            snr = (k * 0.8) / noise if noise else 0
            print(f"     基线噪声约 {noise:,.0f} 计数 → 单次采样信噪比 {snr:.1f}")
            if snr >= 3:
                print(f"     ✅ 可用（配合平均滤波能到 {snr*6:.0f} 倍以上）")
            else:
                print(f"     ⚠️ 偏低，需要加强滤波或改进受力传递")

    print()
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())