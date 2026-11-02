#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
夹爪抓取测试（不带传感器）—— 验证机械结构和舵机控制。

目的
----
在装传感器之前，先确认：
    ① 夹爪能正常张合
    ② 闭合到不同位置时开口有多大
    ③ 能夹住多大尺寸的物体、夹得牢不牢

这样后面加上传感器，就只剩"测力"这一件事要验证了。

⚠️ 安全须知
-----------
下限 650us 是舵机的机械端点。真正要避免的是**两指之间没有东西时闭合到底** ——
那会让舵机堵转（持续大电流、发热），3D 打印件也可能被压裂。

所以：
    · 脚本默认下限 650us
    · **空夹时不要低于 750us**（两指会硬碰硬）
    · **夹着物体时可以到 650us**（力被物体承担，不会顶死）
    · 听到持续嗡嗡声请立刻 Ctrl+C

用法::

    # 张开 → 闭合到 900us → 保持 5 秒 → 张开
    sudo python3 apps/grasp_demo.py --close 900 --hold 5

    # 交互模式：手动一步步调，最安全
    sudo python3 apps/grasp_demo.py --interactive

    # 慢速扫描，观察开口随脉宽的变化
    sudo python3 apps/grasp_demo.py --sweep

    # 反复抓放（模拟连续作业）
    sudo python3 apps/grasp_demo.py --close 900 --hold 3 --repeat 10
"""
from __future__ import annotations

import argparse
import os
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


def set_timer_slack(ns: int = 1) -> bool:
    """
    把本进程的 timer slack 设为 ns 纳秒。

    为什么需要
    ----------
    Linux 默认允许内核把定时器唤醒"松弛"最多 50µs（省电，合并唤醒）。
    而软件 PWM 靠 time.sleep() 定时，50µs 的误差摊到 20ms 周期、
    1.5ms 脉宽上就是约 3° 的舵机抖动 —— 肉眼可见。

    把 slack 设成 1ns 后，sleep 的唤醒精度会大幅提升。
    """
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        PR_SET_TIMERSLACK = 29
        return libc.prctl(PR_SET_TIMERSLACK, ctypes.c_ulong(ns), 0, 0, 0) == 0
    except Exception:  # noqa: BLE001
        return False


def read_timer_slack() -> str:
    try:
        with open("/proc/self/timerslack_ns", encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return "?"


def make_servo(pin: int, pmin: float, pmax: float, period_us: float):
    """
    建舵机 —— 优先用库里的忙等 PWM 驱动（抖动明显小于 gpiozero）。

    忙等 PWM 是实测结论：gpiozero 的软件 PWM 有肉眼可见抖动，
    chrt / timer slack 都改不掉，而忙等 PWM 明显更稳。
    """
    try:
        from grape_gripper.actuator import BusyPwmServo
        s = BusyPwmServo(gpio_pin=pin, period_us=period_us,
                         pulse_min_us=pmin, pulse_max_us=pmax,
                         angle_min=0.0, angle_max=180.0)
        s.set_pulse_us(1500.0)
        print(f"  驱动：忙等 PWM（库实现）")
        return s
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠️ 忙等 PWM 不可用（{exc}），退回 gpiozero")

    from gpiozero import Device, Servo
    from gpiozero.pins.lgpio import LGPIOFactory

    Device.pin_factory = LGPIOFactory(chip=0)
    print(f"  驱动：gpiozero 软件 PWM（抖动较大）")
    return Servo(pin,
                 min_pulse_width=pmin / 1e6,
                 max_pulse_width=pmax / 1e6,
                 frame_width=period_us / 1e6,
                 initial_value=None)


def main() -> int:
    ap = argparse.ArgumentParser(description="夹爪抓取测试（不含传感器）")
    ap.add_argument("--config", default=None)
    ap.add_argument("--pin", type=int, default=None)
    ap.add_argument("--open", dest="open_pulse", type=float, default=1450.0,
                    help="张开脉宽（默认 1450us）")
    ap.add_argument("--close", dest="close_pulse", type=float, default=900.0,
                    help="闭合脉宽（默认 900us）")
    ap.add_argument("--min-pulse", type=float, default=650.0,
                    help="安全下限（默认 650us = 舵机机械端点）。"
                         "两指硬碰硬顶死会堵转发热，所以别低于 650")
    ap.add_argument("--hold", type=float, default=5.0, help="保持时间（秒）")
    ap.add_argument("--repeat", type=int, default=1, help="重复次数")
    ap.add_argument("--interactive", action="store_true", help="交互模式")
    ap.add_argument("--sweep", action="store_true", help="慢速扫描开口")
    ap.add_argument("--timer-slack", type=int, default=1,
                    help="定时器松弛量（纳秒）。1=最小抖动；0=不改（对照用）")
    args = ap.parse_args()

    cfg = Config.load(args.config or default_config_path())
    if cfg.simulate:
        print("⚠️ config.yaml 里 hardware.simulate = true，请改成 false")
        return 1

    pin = args.pin if args.pin is not None else cfg.get_int("servo.gpio_pin", 49)
    pmin = cfg.get_float("servo.pulse_min_us", 650.0)
    pmax = cfg.get_float("servo.pulse_max_us", 1450.0)
    period = cfg.get_float("servo.period_us", 20000.0)

    print("=" * 70)
    print("夹爪抓取测试（不含传感器）")
    print("=" * 70)
    print(f"  GPIO{pin}   脉宽范围 {pmin:.0f}~{pmax:.0f}us")

    # ★ 关键：减小 timer slack，提升软件 PWM 的定时精度
    slack_before = read_timer_slack()
    if args.timer_slack > 0:
        ok = set_timer_slack(args.timer_slack)
        print(f"  timer slack: {slack_before} → {read_timer_slack()} ns"
              f"  {'✅ 已优化' if ok else '❌ 设置失败'}")
    else:
        print(f"  timer slack: {slack_before} ns（未修改，对照用）")
    print(f"  张开 {args.open_pulse:.0f}us   闭合 {args.close_pulse:.0f}us   "
          f"安全下限 {args.min_pulse:.0f}us")
    print()
    print("⚠️ 听到持续嗡嗡声（顶死）请立刻 Ctrl+C")

    # ★ 允许 --min-pulse 突破配置里的脉宽下限：
    #   配置里 pulse_min_us=650 是"留了余量的安全值"，
    #   但排查机械端点时需要能往下试（比如 500us）。
    eff_pmin = min(pmin, args.min_pulse)
    if eff_pmin < 600:
        print(f"  ⚠️ 脉宽下限放宽到 {eff_pmin:.0f}us（低于 600 会压到机械端点，")
        print(f"     听到持续嗡嗡声请立刻 Ctrl+C）")

    try:
        servo = make_servo(pin, eff_pmin, pmax, period)
    except Exception as exc:  # noqa: BLE001
        print(f"❌ 舵机初始化失败：{exc}")
        return 1

    def set_pulse(p: float) -> None:
        p = max(args.min_pulse, min(pmax, p))
        if hasattr(servo, "set_pulse_us"):
            servo.set_pulse_us(p)              # 忙等 PWM（库实现）
            return
        ratio = (p - pmin) / (pmax - pmin) if pmax > pmin else 0.0
        ratio = max(0.0, min(1.0, ratio))
        servo.value = ratio * 2.0 - 1.0        # gpiozero

    try:
        input("\n按回车开始（先张开）...")
        set_pulse(args.open_pulse)
        print(f"  → 张开到 {args.open_pulse:.0f}us")
        time.sleep(1.5)

        # ---------------- 扫描模式 ----------------
        if args.sweep:
            print()
            print("慢速扫描：观察开口随脉宽的变化，记录每个脉宽下的开口宽度")
            print()
            print(f"{'脉宽(us)':>10}   说明")
            print("-" * 40)
            p = args.open_pulse
            while p >= args.min_pulse:
                set_pulse(p)
                print(f"{p:>10.0f}   ← 请用尺量 / 目测开口")
                time.sleep(2.0)
                p -= 50
            set_pulse(args.open_pulse)
            print(f"\n已回到张开位 {args.open_pulse:.0f}us")
            return 0

        # ---------------- 交互模式 ----------------
        if args.interactive:
            print()
            print("交互模式：输入脉宽数值（回车=闭合，q=退出）")
            print(f"  当前 {args.open_pulse:.0f}us（张开）")
            cur = args.open_pulse
            while True:
                try:
                    line = input(f"  脉宽 [{args.min_pulse:.0f}~{pmax:.0f}] "
                                 f"（当前 {cur:.0f}）: ").strip()
                except EOFError:
                    break
                if line.lower() in ("q", "quit", "exit"):
                    break
                if not line:
                    line = str(args.close_pulse)
                try:
                    cur = float(line)
                except ValueError:
                    print("    ❌ 请输入数字")
                    continue
                set_pulse(cur)
                actual = max(args.min_pulse, min(pmax, cur))
                note = "" if abs(actual - cur) < 0.5 else \
                    f"   ⚠️ 已限幅到 {actual:.0f}us"
                print(f"    → 已设为 {actual:.0f}us{note}")
            set_pulse(args.open_pulse)
            return 0

        # ---------------- 抓取循环 ----------------
        for i in range(args.repeat):
            print()
            print(f"── 第 {i+1}/{args.repeat} 次 ──")

            set_pulse(args.open_pulse)
            print(f"  [{time.strftime('%H:%M:%S')}] 张开 {args.open_pulse:.0f}us")
            time.sleep(1.0)

            if args.repeat > 1:
                print("         （现在把物体放进夹爪）")
                time.sleep(1.5)

            set_pulse(args.close_pulse)
            print(f"  [{time.strftime('%H:%M:%S')}] 闭合 {args.close_pulse:.0f}us"
                  f"  保持 {args.hold:.0f}s")

            # 保持期间，用户可以试着拽物体，看夹得牢不牢
            t0 = time.time()
            while time.time() - t0 < args.hold:
                left = args.hold - (time.time() - t0)
                print(f"         保持中... 剩 {left:4.1f}s", end="\r")
                time.sleep(0.2)
            print(" " * 40, end="\r")

            set_pulse(args.open_pulse)
            print(f"  [{time.strftime('%H:%M:%S')}] 松开")
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n中断")
    finally:
        try:
            set_pulse(args.open_pulse)
            time.sleep(0.5)
        except Exception:  # noqa: BLE001
            pass
        # 卸力 + 释放 GPIO（两种驱动接口不同，逐个尝试）
        for meth in ("disable", "detach", "close"):
            fn = getattr(servo, meth, None)
            if callable(fn):
                try:
                    fn()
                except Exception:  # noqa: BLE001
                    pass

    print()
    print("=" * 70)
    print("完成。请记录：")
    print("  · 能夹住的物体尺寸范围")
    print("  · 闭合到多少 us 时刚好夹住（不松不顶）")
    print("  · 保持期间是否听到嗡嗡声（顶死）")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())