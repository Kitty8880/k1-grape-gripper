#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PS01 噪声分析 —— 找出信噪比最好的信号组合。

做三件事：
    ① 测各通道的原始噪声（标准差）
    ② 测通道间的相关性（噪声是共模还是独立）
    ③ 评估各种组合的信噪比（PD1 / PD2 / PD1-PD2 / PD1-PD3 / 加权 ...）

理论依据
--------
PD3 对夹持力没有响应（实测），但它和 PD1/PD2 共用同一个 AFE 和 LED 驱动。
如果噪声主要来自共模源（LED 强度波动、供电纹波、温漂），
那么做差就能抵消噪声、保留力信号，信噪比可能大幅提升。

用法::

    sudo python3 scripts/ps01_noise_analysis.py --seconds 8
    sudo python3 scripts/ps01_noise_analysis.py --seconds 8 --known-n 2.53 --known-d 3956
"""
from __future__ import annotations

import argparse
import math
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


def pearson(a, b) -> float:
    n = min(len(a), len(b))
    if n < 3:
        return 0.0
    a, b = a[:n], b[:n]
    ma, mb = statistics.mean(a), statistics.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da and db else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description="PS01 噪声分析")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--known-n", type=float, default=2.53,
                    help="标定用的已知力（牛顿）")
    ap.add_argument("--known-d", type=float, default=3956.0,
                    help="该已知力下 PD1 的读数变化")
    ap.add_argument("--target-n", type=float, default=0.8,
                    help="关心的目标力（牛顿）")
    ap.add_argument("--sensor-hz", type=float, default=160.0,
                    help="传感器实际更新率（不是读取循环的轮询率！）")
    args = ap.parse_args()

    print("=" * 76)
    print("PS01 噪声分析")
    print("=" * 76)

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
    print(f"✅ {r.describe()}")
    print()
    print(f"采集中（{args.seconds:.0f} 秒）—— 请不要触碰传感器")
    print("  提示：可以先在传感器下面垫海绵再跑一次，对比振动影响")

    # 尽量同时读取三通道（相邻微秒内），保证时间对齐
    p1, p2, p3 = [], [], []
    t0 = time.time()
    while time.time() - t0 < args.seconds:
        v1, v2, v3 = r.channel(1), r.channel(2), r.channel(3)
        if v1 is not None and v2 is not None and v3 is not None:
            p1.append(float(v1))
            p2.append(float(v2))
            p3.append(float(v3))
        time.sleep(0.001)
    r.close()

    n = len(p1)
    print(f"  有效样本 {n} 个（约 {n/max(0.1,args.seconds):.0f} Hz）")
    if n < 50:
        print("❌ 样本太少")
        return 1

    m1, m2, m3 = (statistics.mean(p1), statistics.mean(p2), statistics.mean(p3))
    s1, s2, s3 = (statistics.pstdev(p1), statistics.pstdev(p2),
                  statistics.pstdev(p3))

    print()
    print("─" * 76)
    print("① 各通道噪声")
    print("─" * 76)
    print(f"{'通道':>6} {'均值':>12} {'标准差':>10} {'相对噪声':>10}")
    for name, m, s in (("PD1", m1, s1), ("PD2", m2, s2), ("PD3", m3, s3)):
        print(f"{name:>6} {m:>12,.0f} {s:>10,.0f} {s/m*100:>9.3f}%")

    print()
    print("─" * 76)
    print("② 通道间相关性（噪声是共模还是独立）")
    print("─" * 76)
    c12 = pearson(p1, p2)
    c13 = pearson(p1, p3)
    c23 = pearson(p2, p3)
    print(f"  PD1 ↔ PD2 : {c12:+.3f}")
    print(f"  PD1 ↔ PD3 : {c13:+.3f}")
    print(f"  PD2 ↔ PD3 : {c23:+.3f}")
    print()
    print("  解读：")
    print("   接近 +1 → 噪声高度共模 → 做差能抵消，信噪比大幅提升")
    print("   接近  0 → 噪声独立 → 做差没用，只能靠平均")
    print("   接近 -1 → 反相共模 → 求和能抵消")

    # 灵敏度（用 PD1 的标定值推算其它组合）
    k1 = args.known_d / args.known_n if args.known_n else 0.0

    # 各组合的噪声
    diff12 = [a - b for a, b in zip(p1, p2)]
    diff13 = [a - b for a, b in zip(p1, p3)]
    diff23 = [a - b for a, b in zip(p2, p3)]
    s_d12 = statistics.pstdev(diff12)
    s_d13 = statistics.pstdev(diff13)
    s_d23 = statistics.pstdev(diff23)

    print()
    print("─" * 76)
    print("③ 各信号组合的噪声（假设力响应只体现在 PD1/PD2）")
    print("─" * 76)
    print(f"{'信号':>12} {'噪声 σ':>10} {'vs PD1':>10}")
    print(f"{'PD1':>12} {s1:>10,.0f} {'1.00x':>10}")
    print(f"{'PD2':>12} {s2:>10,.0f} {s2/s1:>9.2f}x")
    print(f"{'PD1-PD2':>12} {s_d12:>10,.0f} {s_d12/s1:>9.2f}x")
    print(f"{'PD1-PD3':>12} {s_d13:>10,.0f} {s_d13/s1:>9.2f}x")
    print(f"{'PD2-PD3':>12} {s_d23:>10,.0f} {s_d23/s1:>9.2f}x")

    print()
    print("─" * 76)
    print(f"④ 目标力 {args.target_n:.2f}N 的可分辨性（按 PD1 灵敏度 {k1:,.0f} 计数/N）")
    print("─" * 76)
    sig = k1 * args.target_n
    print(f"  目标力对应的读数变化： {sig:,.0f} 计数")
    print()
    print(f"{'平均窗口':>10} {'独立样本':>8} {'噪声 σ':>10} {'信噪比':>8}  判定")
    print("-" * 60)
    print("  ⚠️ 注意：独立样本按传感器实际更新率 "
          f"{args.sensor_hz:.0f}Hz 计算，")
    print("     不是读取循环的轮询率（重复读同一个值不会降噪）")
    print()
    for win in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0):
        ns = max(1, int(args.sensor_hz * win)) if win > 0 else 1
        s_eff = s1 / math.sqrt(ns)
        snr = sig / s_eff if s_eff else 0
        label = "单次采样" if win == 0 else f"{win:.2f}s"
        verdict = ("✅ 好" if snr >= 5 else
                   "🔶 可用" if snr >= 3 else
                   "⚠️ 勉强" if snr >= 2 else "❌ 不够")
        print(f"{label:>10} {ns:>8} {s_eff:>10,.0f} {snr:>8.2f}  {verdict}")

    # ---- 差分组合的收益 ----
    print()
    print("─" * 76)
    print("⑤ 用 PDi - PD3 作为力信号能改善多少")
    print("─" * 76)
    print("  （PD3 对力无响应，但和 PD1/PD2 共享噪声 → 做差可抵消）")
    print()
    print(f"{'信号':>12} {'噪声 σ':>10} {'0.8N 信噪比':>12} {'相对 PD1':>10}")
    for label, s_comb in (("PD1", s1), ("PD2", s2),
                          ("PD1-PD3", s_d13), ("PD2-PD3", s_d23)):
        snr0 = sig / s_comb if s_comb else 0
        print(f"{label:>12} {s_comb:>10,.0f} {snr0:>12.2f} {s1/s_comb:>9.2f}x")

    print()
    print("=" * 76)
    return 0


if __name__ == "__main__":
    sys.exit(main())