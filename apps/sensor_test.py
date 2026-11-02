#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
走配置文件读取力传感器 —— 验证 config.yaml → sensor → PS01 整条链路。

用法::

    # 连续显示读数（用手按压传感器，看数值变化）
    sudo python3 apps/sensor_test.py

    # 只读 10 秒
    sudo python3 apps/sensor_test.py --seconds 10

    # 记录空载基准（标定第一步）
    sudo python3 apps/sensor_test.py --tare

    # 采样统计（检查噪声水平）
    sudo python3 apps/sensor_test.py --stats
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

from grape_gripper.config import Config, default_config_path   # noqa: E402
from grape_gripper.sensor import create_adc, SensorError        # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="按配置文件读力传感器")
    ap.add_argument("--config", default=None)
    ap.add_argument("--seconds", type=float, default=0,
                    help="采样时长（0 = 直到 Ctrl+C）")
    ap.add_argument("--tare", action="store_true", help="记录空载基准")
    ap.add_argument("--stats", action="store_true", help="统计噪声水平")
    ap.add_argument("--hz", type=float, default=10.0, help="显示刷新率")
    args = ap.parse_args()

    cfg_path = args.config or default_config_path()
    print("=" * 70)
    print("力传感器读取（按配置文件）")
    print("=" * 70)
    print(f"配置文件： {cfg_path}")

    cfg = Config.load(cfg_path)
    adc_type = cfg.get_str("sensor.adc", "?")
    port = cfg.get_str("sensor.ps01.port", "-")
    ch_l = cfg.get_int("sensor.channel_left", 0)
    ch_r = cfg.get_int("sensor.channel_right", 1)
    cmap = cfg.get("sensor.ps01.channel_map", {}) or {}

    print()
    print("传感器参数：")
    print(f"  类型          {adc_type}")
    print(f"  端口          {port}")
    print(f"  通道映射      {cmap}")
    print(f"  左/右通道     {ch_l} / {ch_r}")
    print(f"  hardware.simulate = {cfg.get_bool('hardware.simulate', True)}")

    print()
    print("正在初始化传感器...")
    try:
        adc = create_adc(cfg)
    except SensorError as exc:
        print(f"❌ {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"❌ {type(exc).__name__}: {exc}")
        return 1

    print(f"✅ {adc.describe()}")

    # ---------------- 记录空载基准 ----------------
    if args.tare:
        print()
        print("记录空载基准（请确保传感器没有受力）...")
        time.sleep(1.0)
        samples_l, samples_r = [], []
        for _ in range(100):
            samples_l.append(adc.read_raw(ch_l))
            samples_r.append(adc.read_raw(ch_r))
            time.sleep(0.01)
        ml = statistics.mean(samples_l)
        mr = statistics.mean(samples_r)
        print()
        print(f"  左通道 空载均值 = {ml:,.0f}   标准差 = {statistics.pstdev(samples_l):,.0f}")
        print(f"  右通道 空载均值 = {mr:,.0f}   标准差 = {statistics.pstdev(samples_r):,.0f}")
        print()
        print("  把这两个值填进 config.yaml：")
        print(f"    sensor.calib.left.raw_zero:  {ml:.0f}")
        print(f"    sensor.calib.right.raw_zero: {mr:.0f}")
        adc.close()
        return 0

    # ---------------- 统计模式 ----------------
    if args.stats:
        print()
        print("采 3 秒统计噪声...")
        sl, sr = [], []
        t0 = time.time()
        while time.time() - t0 < 3.0:
            sl.append(adc.read_raw(ch_l))
            sr.append(adc.read_raw(ch_r))
            time.sleep(0.005)
        print()
        for name, s in (("左", sl), ("右", sr)):
            if not s:
                continue
            print(f"  {name}通道： n={len(s)}  均值={statistics.mean(s):,.0f}  "
                  f"标准差={statistics.pstdev(s):,.0f}  "
                  f"极差={max(s)-min(s):,.0f}")
        print()
        print("  标准差小 → 读数稳定；很大 → 检查供电/接线/是否在受力")
        adc.close()
        return 0

    # ---------------- 连续显示 ----------------
    print()
    print("连续显示（用手按压夹指，观察数值变化）")
    print("Ctrl+C 停止")
    print()
    print(f"{'时间':>7}  {'左通道':>12}  {'右通道':>12}  {'左-右':>10}")
    print("-" * 52)

    t_start = time.time()
    interval = 1.0 / max(0.5, args.hz)
    base_l = base_r = None
    try:
        while True:
            now = time.time()
            if args.seconds and (now - t_start) > args.seconds:
                break

            vl = adc.read_raw(ch_l)
            vr = adc.read_raw(ch_r)
            if base_l is None:
                base_l, base_r = vl, vr

            print(f"{now - t_start:6.1f}s  {vl:>12,.0f}  {vr:>12,.0f}  "
                  f"{vl - vr:>+10,.0f}")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n中断")
    finally:
        # 打印底层统计
        stats = getattr(adc, "stats", None)
        if callable(stats):
            st = stats()
            print()
            print("底层统计：")
            for k, v in st.items():
                print(f"  {k:<18} {v}")
        adc.close()

    print()
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())