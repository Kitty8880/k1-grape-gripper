#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
舵机安全闭合限位自动标定 —— 装上传感器后必做。

背景
----
装上传感器（和对称垫块）后，夹爪的几何变了：
原来"600µs = 两指接触"，现在 600µs 会把传感器直接压爆
（MG996R 的夹持力可达 20N 以上，而传感器设计量程只有几牛）。

本脚本用**传感器读数**当判据，从小脉宽开始逐步闭合，
一旦检测到接触（力开始上升）就立刻停止并回退，
从而找到一个安全的闭合限位。

安全措施
--------
  · 每步只减小 --step 微秒（默认 40µs）
  · 每步之间读力，超过 --threshold 立即停止
  · 到达 --min-pulse 硬下限立刻停止
  · Ctrl+C 随时中断，退出时自动卸力

用法::

    sudo python3 scripts/servo_force_limit_finder.py
    sudo python3 scripts/servo_force_limit_finder.py --start 1500 --step 40
    sudo python3 scripts/servo_force_limit_finder.py --threshold 0.3
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
    from grape_gripper.config import Config, default_config_path
    from grape_gripper.ps01 import Ps01Error, Ps01Reader
except ImportError as exc:
    print(f"❌ 导入失败：{exc}")
    sys.exit(1)

PERIOD_US = 20000.0


def main() -> int:
    ap = argparse.ArgumentParser(description="舵机安全闭合限位标定")
    ap.add_argument("--config", default=None)
    ap.add_argument("--pin", type=int, default=None, help="GPIO（默认取配置）")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--channel", type=int, default=1, help="用哪个 PD 通道")
    ap.add_argument("--start", type=float, default=1500.0,
                    help="起始脉宽（应该是完全张开的位置）")
    ap.add_argument("--step", type=float, default=25.0,
                    help="每步减小多少微秒（越小越安全，但越慢）")
    ap.add_argument("--min-pulse", type=float, default=650.0,
                    help="硬下限，到这里一定停")
    ap.add_argument("--threshold", type=float, default=0.3,
                    help="接触判定力（牛顿）。实际取 max(本值, 4×噪声)")
    ap.add_argument("--baseline", type=float, default=3.0,
                    help="空载噪声测量时长（秒）")
    ap.add_argument("--margin", type=float, default=100.0,
                    help="找到接触后回退多少微秒作为安全限位")
    ap.add_argument("--settle", type=float, default=0.6, help="每步稳定时间")
    ap.add_argument("--measure", type=float, default=0.8, help="每步测力时长")
    ap.add_argument("--sensor-hz", type=float, default=160.0,
                    help="传感器有效更新率（用于把单次噪声折算到平均值）")
    args = ap.parse_args()

    cfg = Config.load(args.config or default_config_path())
    pin = args.pin if args.pin is not None else cfg.get_int("servo.gpio_pin", 49)
    raw_zero = cfg.get_float("sensor.calib.left.raw_zero", 683457.0)
    raw_span = cfg.get_float("sensor.calib.left.raw_span", 15640.0)
    f_max = cfg.get_float("sensor.calib.left.force_max_n", 10.0)
    k_counts_per_n = raw_span / f_max if f_max else 1564.0
    # 脉宽范围从配置读（装上传感器后可能调整过）
    pmin_cfg = cfg.get_float("servo.pulse_min_us", 650.0)
    pmax_cfg = cfg.get_float("servo.pulse_max_us", 1450.0)

    print("=" * 74)
    print("舵机安全闭合限位标定")
    print("=" * 74)
    print(f"  GPIO{pin}   传感器 {args.port} PD{args.channel}")
    print(f"  灵敏度 {k_counts_per_n:,.0f} 计数/N   零位 {raw_zero:,.0f}")
    print(f"  脉宽范围 {pmin_cfg:.0f}~{pmax_cfg:.0f}us（来自 config.yaml）")
    print(f"  起始脉宽 {args.start:.0f}us   步长 -{args.step:.0f}us   "
          f"硬下限 {args.min_pulse:.0f}us")
    print(f"  接触判定 {args.threshold:.2f}N")
    print()
    print("⚠️ 上机前确认：")
    print("   1. 传感器已固定在夹指上（受力面朝内）")
    print("   2. 夹爪前方没有任何障碍物")
    print("   3. 手放在电源开关上")
    print()
    print("★ 本脚本的作用：装上传感器后，原来的'两指接触脉宽'失效了。")
    print("  如果还用旧值闭合，会把传感器压在另一侧夹指上 —— 可能压坏。")
    print("  本脚本用传感器读数当判据，找到'刚刚接触'的脉宽并回退。")
    print()
    input("   确认后按回车开始...")

    # ---- 传感器 ----
    try:
        reader = Ps01Reader(args.port)
        reader.open()
    except Ps01Error as exc:
        print(f"❌ {exc}")
        return 1
    if not reader.wait_first_sample(timeout=3.0):
        print(f"❌ 传感器无数据：{reader.stats()}")
        reader.close()
        return 1

    # ---- 舵机 ----
    try:
        from gpiozero import Device, Servo
        from gpiozero.pins.lgpio import LGPIOFactory
        Device.pin_factory = LGPIOFactory(chip=cfg.get_int("servo.gpio_chip", 0))
        servo = Servo(pin,
                      min_pulse_width=650 / 1e6,
                      max_pulse_width=1450 / 1e6,
                      frame_width=PERIOD_US / 1e6,
                      initial_value=None)
    except Exception as exc:  # noqa: BLE001
        print(f"❌ 舵机初始化失败：{exc}")
        reader.close()
        return 1

    def set_pulse(pulse_us: float) -> None:
        span = pmax_cfg - pmin_cfg
        ratio = (pulse_us - pmin_cfg) / span if span else 0.0
        ratio = max(0.0, min(1.0, ratio))
        servo.value = ratio * 2.0 - 1.0

    def measure_force(duration: float) -> tuple[float, float]:
        vals = []
        t0 = time.time()
        while time.time() - t0 < duration:
            v = reader.channel(args.channel)
            if v is not None:
                vals.append(float(v))
            time.sleep(0.002)
        if not vals:
            return 0.0, 0.0
        mean = statistics.mean(vals)
        force = (mean - raw_zero) / k_counts_per_n
        return force, mean

    contact_pulse = None
    try:
        print()
        print(f"{'脉宽(us)':>10} {'读数':>12} {'力(N)':>10}  状态")
        print("-" * 60)

        # ---- 先张开，并测出"空载噪声"作为判定基准 ----
        set_pulse(args.start)
        time.sleep(1.5)
        print(f"正在测空载噪声（{args.baseline:.1f}s，夹爪张开、无接触）...")
        bl_vals = []
        t0 = time.time()
        while time.time() - t0 < args.baseline:
            v = reader.channel(args.channel)
            if v is not None:
                bl_vals.append(float(v))
            time.sleep(0.002)

        if len(bl_vals) < 20:
            print("❌ 基线采集失败")
            return 1
        bl_mean = statistics.mean(bl_vals)
        bl_std = statistics.pstdev(bl_vals)
        noise_n = bl_std / k_counts_per_n
        f0 = (bl_mean - raw_zero) / k_counts_per_n

        # ★ 关键：noise_n 是"单次采样"的噪声；而判定用的是 measure 秒的平均值，
        #   平均后噪声按 √N 下降。直接拿原始噪声算阈值会严重偏高
        #   （实测踩过这个坑：阈值被抬到 10.75N，整轮扫描都不触发）。
        n_samples = max(1.0, args.measure * args.sensor_hz)
        noise_n_avg = noise_n / math.sqrt(n_samples)
        eff_threshold = max(args.threshold, 4.0 * noise_n_avg)
        print(f"  空载读数 {bl_mean:,.0f}   噪声 σ {bl_std:,.0f} 计数 "
              f"= {noise_n:.2f} N（单次采样）")
        print(f"  起始力 {f0:+.3f} N")
        print(f"  平均 {args.measure:.1f}s（≈{n_samples:.0f} 个样本）后 "
              f"噪声 σ = {noise_n_avg:.2f} N")
        print(f"  ★ 判定阈值：{eff_threshold:.2f} N "
              f"（设定 {args.threshold:.2f} ，噪声下限 {4.0*noise_n_avg:.2f}）")
        if noise_n_avg > 0.15:
            print(f"  ⚠️ 平均后噪声仍偏大（{noise_n_avg:.2f}N），"
                  f"可加大 --measure")
        print()

        pulse = args.start
        consecutive = 0
        while pulse > args.min_pulse:
            pulse -= args.step
            set_pulse(pulse)
            time.sleep(args.settle)
            force, raw = measure_force(args.measure)

            status = ""
            # ★ 要求连续两步超阈值才算接触（噪声是随机的，接触是持续的）
            if force >= eff_threshold:
                consecutive += 1
                status = (f"超出阈值（第 {consecutive}/2 次）" if consecutive < 2
                          else "★ 检测到接触，停止")
            else:
                consecutive = 0

            print(f"{pulse:>10.0f} {raw:>12,.0f} {force:>+10.3f}  {status}")

            if consecutive >= 2:
                contact_pulse = pulse
                break
            if force > eff_threshold * 4:
                print("      ⚠️ 力远超阈值，立即停止")
                contact_pulse = pulse
                break

        # 回退到安全位置
        safe_pulse = (contact_pulse + args.margin) if contact_pulse else (pulse + args.margin)
        safe_pulse = min(safe_pulse, args.start)
        print()
        print(f"→ 回退到安全脉宽 {safe_pulse:.0f}us")
        set_pulse(safe_pulse)
        time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n中断")
    finally:
        try:
            servo.detach()
        except Exception:  # noqa: BLE001
            pass
        reader.close()

    print()
    print("=" * 74)
    print("结论")
    print("=" * 74)
    if contact_pulse is None:
        print(f"  在 {args.min_pulse:.0f}us 之前没有检测到接触。")
        print("  可能：夹爪行程不够，或者传感器没装好（读数不响应）")
    else:
        print(f"  两指接触发生在： {contact_pulse:.0f} us")
        print(f"  建议安全闭合限位： {safe_pulse:.0f} us（留 {args.margin:.0f}us 余量）")
        print()
        print("  换算成角度（配置里的坐标系，0°=全闭、180°=全开）：")
        # 角度映射：650us→0°，1450us→180°
        def to_deg(p):
            return (p - 650.0) / (1450.0 - 650.0) * 180.0
        print(f"    接触点 {contact_pulse:.0f}us  → {to_deg(contact_pulse):.1f}°")
        print(f"    安全限位 {safe_pulse:.0f}us → {to_deg(safe_pulse):.1f}°")
        print()
        print("  把安全限位填进 config.yaml：")
        print(f"    servo:")
        print(f"      angle_close_deg: {to_deg(safe_pulse):.1f}"
              f"     # 原为 0.0（危险！会让两指对压传感器）")
        print()
        print("  ⚠️ 注意：这样夹爪最小开口会略大于 0，")
        print("     但不影响夹葡萄（葡萄直径 22mm）。")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())