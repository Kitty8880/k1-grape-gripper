#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PS01 双传感器测试 —— 验证第二只传感器是否可用，并评估差分降噪效果。

为什么要专门测这个
------------------
1. **验证第二只传感器能否在 K1 上正常工作**
   第一只的经验是"两步启动"（先等初始化日志打完，再发启动命令），
   第二只必须重新验证这一套是否成立。

2. **验证两个 CH340 的设备名是否会互换**
   插入顺序不同，/dev/ttyUSB0 和 /dev/ttyUSB1 可能对调。
   用 /dev/serial/by-id/ 下的稳定名字可以避免这个问题。

3. **评估差分降噪的可行性**（这是装第二只传感器最大的技术价值）
   之前单只的实测发现：PD1 与 PD3 的噪声相关系数高达 +0.74，
   而 PD1-PD3 的噪声比 PD1 低约 21%。
   如果**两只传感器**的噪声也高度相关（比如共同受供电/温度影响），
   那么 A-B 的噪声会明显低于单只 —— 直接提升力控精度。
   本工具会算出相关系数和差分噪声，给出结论。

用法::

    # 自动探测两个串口
    sudo python3 scripts/ps01_dual_test.py

    # 手动指定（推荐用 by-id 的稳定名字）
    sudo python3 scripts/ps01_dual_test.py --port-a /dev/ttyUSB0 --port-b /dev/ttyUSB1

    # 只测一只（验证第二只是否可用）
    sudo python3 scripts/ps01_dual_test.py --port-a /dev/ttyUSB1
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import statistics
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

from grape_gripper.ps01 import Ps01Error, Ps01Reader   # noqa: E402


def list_ports() -> None:
    print("── 可用串口 ──")
    ports = sorted(glob.glob("/dev/ttyUSB*")) + sorted(glob.glob("/dev/ttyACM*"))
    if not ports:
        print("  （没有找到 ttyUSB* 设备）")
    for p in ports:
        try:
            target = os.path.realpath(p)
        except OSError:
            target = "?"
        print(f"  {p}  →  {target}")

    byid = "/dev/serial/by-id"
    if os.path.isdir(byid):
        print()
        print("── 稳定名字（/dev/serial/by-id）──")
        entries = sorted(os.listdir(byid))
        if not entries:
            print("  （空）")
        for e in entries:
            link = os.path.join(byid, e)
            try:
                tgt = os.path.basename(os.path.realpath(link))
            except OSError:
                tgt = "?"
            print(f"  {e}")
            print(f"      → {tgt}")
        print()
        print("  ★ 建议用这里的名字，插入顺序变了也不会错")
    else:
        print()
        print("  （没有 /dev/serial/by-id 目录）")


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
    ap = argparse.ArgumentParser(description="PS01 双传感器测试")
    ap.add_argument("--port-a", default=None, help="传感器 A 的串口")
    ap.add_argument("--port-b", default=None, help="传感器 B 的串口（不填则只测 A）")
    ap.add_argument("--channel", type=int, default=1, help="用哪个 PD 通道比较")
    ap.add_argument("--seconds", type=float, default=8.0, help="同时读取时长")
    ap.add_argument("--list", action="store_true", help="只列串口")
    args = ap.parse_args()

    print("=" * 74)
    print("PS01 双传感器测试")
    print("=" * 74)
    print()
    list_ports()
    print()

    if args.list:
        return 0

    port_a = args.port_a
    port_b = args.port_b
    ports = sorted(glob.glob("/dev/ttyUSB*"))
    if port_a is None:
        if not ports:
            print("❌ 没找到串口设备")
            return 1
        port_a = ports[0]
        if port_b is None and len(ports) > 1:
            port_b = ports[1]
    elif port_b is None and len(ports) > 1:
        other = [p for p in ports if p != port_a]
        if other:
            port_b = other[0]

    print("-" * 74)
    print(f"  传感器 A：{port_a}")
    print(f"  传感器 B：{port_b or '（不测）'}")
    print("-" * 74)
    print()

    readers = {}
    try:
        for name, port in (("A", port_a), ("B", port_b)):
            if not port:
                continue
            print(f"[{name}] 打开 {port} ...")
            try:
                r = Ps01Reader(port)
                r.open()
            except Ps01Error as exc:
                print(f"      ❌ {exc}")
                if name == "A":
                    return 1
                continue
            if not r.wait_first_sample(timeout=3.0):
                print(f"      ❌ 没有数据：{r.stats()}")
                try:
                    r.close()
                except Exception:  # noqa: BLE001
                    pass
                continue
            print(f"      ✅ {r.describe()}")
            readers[name] = r
            time.sleep(0.3)

        if "A" not in readers:
            print("\n❌ 传感器 A 不可用")
            return 1

        if "B" not in readers:
            print()
            print("=" * 74)
            print("结论")
            print("=" * 74)
            print("  只测到 1 只传感器。")
            print("  如果第二只插着但没读到，检查：")
            print("    · ls /dev/ttyUSB*  有没有两个设备")
            print("    · 是否被别的程序占用")
            print("    · 用 --port-b 手动指定")
            print("=" * 74)
            return 0

        # ---- 同时读取 ----
        print()
        print(f"[同时读取] {args.seconds:.0f} 秒 —— 请不要触碰传感器")
        print()
        print(f"{'时间':>7} {'A':>12} {'B':>12} {'A-B':>10}")
        print("-" * 46)

        va, vb = [], []
        distinct_a, distinct_b = 0, 0
        last_a = last_b = None
        t0 = time.time()
        next_print = 0.5
        while time.time() - t0 < args.seconds:
            a = readers["A"].channel(args.channel)
            b = readers["B"].channel(args.channel)
            if a is not None and b is not None:
                va.append(float(a))
                vb.append(float(b))
                # 统计"独立样本"（值发生变化的次数），而不是轮询次数
                if a != last_a:
                    distinct_a += 1
                    last_a = a
                if b != last_b:
                    distinct_b += 1
                    last_b = b
            el = time.time() - t0
            if el >= next_print:
                next_print += 0.5
                if a is not None and b is not None:
                    print(f"{el:6.1f}s {a:>12,.0f} {b:>12,.0f} "
                          f"{a-b:>+10,.0f}")
            time.sleep(0.002)

        if len(va) < 30:
            print("❌ 样本太少")
            return 1

        # ---- 统计 ----
        ma, mb = statistics.mean(va), statistics.mean(vb)
        sa, sb = statistics.pstdev(va), statistics.pstdev(vb)
        diff = [x - y for x, y in zip(va, vb)]
        md, sd = statistics.mean(diff), statistics.pstdev(diff)
        corr = pearson(va, vb)

        print()
        print("-" * 46)
        print(f"轮询 {len(va)} 次；独立样本 A {distinct_a} 个 / "
              f"B {distinct_b} 个（{args.seconds:.0f}s）")
        print(f"  → 有效更新率约 A {distinct_a/args.seconds:.0f}Hz / "
              f"B {distinct_b/args.seconds:.0f}Hz")
        print("  （注意：轮询频率远高于真实更新率，噪声统计用的是实际采到的值）")
        print()

        # ---- ★ 关键：平均两只的降噪效果 ----
        print("① 各传感器噪声")
        print(f"{'通道':>8} {'均值':>12} {'标准差':>10} {'相对噪声':>10}")
        for nm, m, s in (("A", ma, sa), ("B", mb, sb)):
            print(f"{nm:>8} {m:>12,.0f} {s:>10,.0f} {s/m*100:>9.3f}%")
        print(f"{'A-B':>8} {md:>12,.0f} {sd:>10,.0f}")

        import math as _m
        s_avg = 0.5 * _m.sqrt(sa**2 + sb**2 + 2 * corr * sa * sb)
        s_indep = _m.sqrt(sa**2 + sb**2)
        print(f"{'(A+B)/2':>8} {(ma+mb)/2:>12,.0f} {s_avg:>10,.0f}")

        print()
        print("② 噪声相关性")
        print(f"  A ↔ B : {corr:+.3f}")
        print("  （接近 +1 = 共模，做差能抵消；接近 0 = 独立，做差无效但平均有效）")

        print()
        print("③ 差分 vs 平均")
        print(f"{'方案':>14} {'噪声 σ':>10} {'相对单只A':>12}")
        print(f"{'单只 A':>14} {sa:>10,.0f} {'1.00x':>12}")
        print(f"{'单只 B':>14} {sb:>10,.0f} {sb/sa:>11.2f}x")
        print(f"{'差分 A-B':>14} {sd:>10,.0f} {sd/sa:>11.2f}x")
        print(f"{'平均 (A+B)/2':>14} {s_avg:>10,.0f} {s_avg/sa:>11.2f}x"
              f"{'  ✅ 更优' if s_avg < sa * 0.95 else ''}")
        print(f"  （若完全独立，平均的理论噪声是 {s_indep/2:,.0f}）")

        # ---- 结论 ----
        print()
        print("=" * 74)
        print("结论")
        print("=" * 74)
        print(f"  ✅ 第二只传感器可正常工作（设备 {port_b}）")
        print(f"     基线 A {ma:,.0f}   B {mb:,.0f}   相差 {md:+,.0f}")
        if abs(md) > 2000:
            print("     ⚠️ 两只基线差得较多，各自需要独立标定")
        print()
        if s_avg < sa * 0.95:
            print(f"  ★ 两只平均能降噪 {(1-s_avg/sa)*100:.0f}%（σ {sa:,.0f} → "
                  f"{s_avg:,.0f}）")
            print("    因为对称夹持时两侧法向力相等，平均两只是合理的 ——")
            print("    这直接提升力控的分辨率，比差分方案实际得多。")
        else:
            print("  两只平均没有明显降噪效果，")
            print("  更适合用于【左右夹指各自测力 + 偏心检测】。")
        print()
        print("  建议用法：")
        print("    · 夹持力 = (A+B)/2   —— 降噪")
        print("    · 偏心度 = |A-B|     —— 物体没夹正时会明显偏大")
        print("=" * 74)
        return 0

    finally:
        for r in readers.values():
            try:
                r.close()
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    sys.exit(main())