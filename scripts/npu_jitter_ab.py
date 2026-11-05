#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NPU 是否真的会让 PS01 力读数变抖？—— 受控 A/B 实测

背景
----
2026-11-05 用户反馈"力读数抖动变厉害"。当时我把原因归给
"NPU 监控占 4 个线程推理，挤掉串口读取线程"，
但那只是**推断，没有实测对比**。

这个脚本补上那个对比：同样的传感器、同样的时长、同样的姿势，
唯一差别是 NPU 推理**开 / 关**，比较两者的噪声标准差 σ。

做法
----
    阶段 A：只读传感器 8 秒            → σ_off
    阶段 B：后台跑 NPU 推理 + 读传感器 8 秒 → σ_on

用同一个 PS01 读取器、同样的采样次数。为了排除"插拔/换脚本"带来的偏差，
两个阶段在同一进程里顺序执行（NPU 用后台线程跑，不走子进程）。

用法::

    sudo python3 scripts/npu_jitter_ab.py
    sudo python3 scripts/npu_jitter_ab.py --seconds 12

⚠️ 运行前先停掉服务，否则服务会占着串口：
    sudo systemctl stop grape-gripper
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _HERE)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass


def collect(seconds: float, tag: str):
    """
    读 PS01 若干秒，返回 (原始计数列表, 采样率)。
    会先把串口上的旧进程情况打印出来，读不到就明确报错。
    """
    from grape_gripper.ps01 import Ps01Reader

    dev = "/dev/ttyUSB0"
    print(f"  [{tag}] 打开 {dev} ...", flush=True)
    # 用法与 grasp_with_verify.py 完全一致：
    #   open() → wait_first_sample() → channel(1) 轮询 → close()
    # ⚠️ 不传 channel 参数（构造函数没有这个参数）；
    #    PD1 由 channel(1) 取得（索引 1 → PD2，2 → PD1，
    #    这里和力控用的是同一个通道，保证可比）
    r = Ps01Reader(dev)
    r.open()
    if not r.wait_first_sample(timeout=3.0):
        raise RuntimeError(f"传感器无数据：{r.stats()}")
    samples = []
    t0 = time.time()
    try:
        while time.time() - t0 < seconds:
            v = r.channel(1)
            if v is not None:
                samples.append(float(v))
            time.sleep(0.002)      # 和力控的采样节奏一致
    finally:
        try:
            r.close()
        except Exception:  # noqa: BLE001
            pass
    rate = len(samples) / max(1e-6, time.time() - t0)
    print(f"  [{tag}] 采到 {len(samples)} 个样本，{rate:.1f} Hz", flush=True)
    return samples, rate


def stats(x):
    if len(x) < 5:
        return None
    m = statistics.median(x)
    # 用中位数去均值，抗离群点（PS01 每 2 秒有尖峰，占 20~25%）
    mad = [abs(v - m) for v in x]
    sigma = statistics.median(mad) * 1.4826      # MAD → σ 的稳健估计
    return {"n": len(x), "median": m,
            "mean": statistics.mean(x),
            "sigma_robust": sigma,
            "sigma_plain": statistics.pstdev(x),
            "min": min(x), "max": max(x),
            "span": max(x) - min(x)}


def main() -> int:
    ap = argparse.ArgumentParser(description="NPU 对手感读数抖动的影响（A/B）")
    ap.add_argument("--seconds", type=float, default=8.0,
                    help="每个阶段测多少秒（默认 8）")
    args = ap.parse_args()

    print("=" * 74)
    print("NPU 对力读数抖动的影响 —— 受控 A/B")
    print("=" * 74)
    print(f"  每个阶段 {args.seconds:.0f} 秒")
    print()

    # ---------- 阶段 A：关闭 NPU ----------
    print("── 阶段 A：NPU 关闭 ──")
    try:
        a_samples, a_rate = collect(args.seconds, "A")
    except Exception as exc:  # noqa: BLE001
        print(f"  ❌ 阶段 A 失败：{exc}")
        print("     排查：sudo pkill -f product_mode; sudo pkill -f grasp_with_verify")
        return 1
    a = stats(a_samples)
    if a is None:
        print("  ❌ 样本太少，检查串口是否被占用：")
        print("     sudo pkill -f product_mode; sudo pkill -f grasp_with_verify")
        return 1

    time.sleep(1.0)

    # ---------- 阶段 B：开启 NPU ----------
    print()
    print("── 阶段 B：NPU 推理运行中 ──")
    mon = None
    try:
        sys.path.insert(0, os.path.join(_ROOT, "apps"))
        from npu_monitor import NpuMonitor
        mon = NpuMonitor(verbose=False)
        mon.start()
        print(f"  已启动：{mon.det.describe()}", flush=True)
        time.sleep(2.0)      # 等第一帧、进入稳定推理
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠️ NPU 启动失败：{exc}")

    try:
        b_samples, b_rate = collect(args.seconds, "B")
    except Exception as exc:  # noqa: BLE001
        print(f"  ❌ 阶段 B 失败：{exc}")
        return 1
    b = stats(b_samples)
    if mon is not None:
        try:
            mon.stop()
        except Exception:  # noqa: BLE001
            pass
    if b is None:
        print("  ❌ 阶段 B 样本太少")
        return 1

    # ---------- 对比 ----------
    print()
    print("=" * 74)
    print("结果")
    print("=" * 74)
    print(f"  {'指标':<22}{'A（NPU 关）':>16}{'B（NPU 开）':>16}{'变化':>12}")
    rows = [
        ("样本数", "n", "{:.0f}"),
        ("采样率 (Hz)", None, "{:.1f}"),
        ("中位数 (counts)", "median", "{:,.0f}"),
        ("σ 稳健 (counts)", "sigma_robust", "{:,.0f}"),
        ("σ 普通 (counts)", "sigma_plain", "{:,.0f}"),
        ("峰峰值 (counts)", "span", "{:,.0f}"),
    ]
    for label, key, fmt in rows:
        if key is None:
            va, vb = a_rate, b_rate
        else:
            va, vb = a[key], b[key]
        if va and vb and key not in (None, "median"):
            chg = f"{(vb-va)/va*100:+.0f}%"
        else:
            chg = "—"
        print(f"  {label:<22}{fmt.format(va):>16}{fmt.format(vb):>16}{chg:>12}")

    print()
    ratio = b["sigma_robust"] / a["sigma_robust"] if a["sigma_robust"] else 0
    if ratio >= 1.25:
        print(f"  ❌ NPU 开启后噪声增大 {ratio:.2f}× —— 确实有干扰，"
              f"默认关闭是对的")
    elif ratio >= 1.10:
        print(f"  ⚠️ NPU 开启后噪声增大 {ratio:.2f}× —— 轻微干扰，"
              f"可能可以接受")
    else:
        print(f"  ✅ NPU 开启后噪声变化 {ratio:.2f}× —— **没有明显干扰**，"
              f"之前的判断错了")
        print("     建议：默认开启 --npu（AI 核 91% 到手）")
    print()
    print("  解读说明：")
    print("    · σ 稳健 = MAD×1.4826，抗离群点 —— PS01 每 2 秒有尖峰，")
    print("      占 20~25% 的样本，用普通 σ 会被尖峰放大，不适合比较")
    print("    · 力 ≈ counts / 1564 N/count（标定值）")
    print(f"    · {a['sigma_robust']:.0f} counts ≈ "
          f"{a['sigma_robust']/1564:.2f} N")
    print(f"      {b['sigma_robust']:.0f} counts ≈ "
          f"{b['sigma_robust']/1564:.2f} N")
    return 0


if __name__ == "__main__":
    sys.exit(main())