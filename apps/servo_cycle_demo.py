#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
舵机循环演示 —— 持续做「张开 → 保持 → 闭合 → 保持」，用于观察运动和平滑度。

用法::

    # 默认跑 10 秒
    sudo python3 apps/servo_cycle_demo.py

    # 跑 30 秒，每端停 2 秒
    sudo python3 apps/servo_cycle_demo.py --seconds 30 --hold 2.0

    # 只走半程（90°↔180°），观察小范围运动是否平滑
    sudo python3 apps/servo_cycle_demo.py --seconds 20 --span 90

    # 慢速运动（更容易看出抖动）
    sudo python3 apps/servo_cycle_demo.py --seconds 20 --rate 40
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

from grape_gripper.actuator import ActuatorError, create_servo   # noqa: E402
from grape_gripper.config import Config, default_config_path      # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="舵机循环演示")
    ap.add_argument("--config", default=None)
    ap.add_argument("--seconds", type=float, default=10.0, help="总时长（秒）")
    ap.add_argument("--hold", type=float, default=1.0, help="每端保持时间（秒）")
    ap.add_argument("--span", type=float, default=0.0,
                    help="行程（度）。0 = 用配置里的全行程；90 = 只走半程")
    ap.add_argument("--rate", type=float, default=0.0,
                    help="运动速率（度/秒）。0 = 用配置值")
    args = ap.parse_args()

    cfg_path = args.config or default_config_path()
    cfg = Config.load(cfg_path)

    if cfg.simulate:
        print("=" * 68)
        print("⚠️  config.yaml 里 hardware.simulate = true")
        print("    这样跑的是模拟舵机，不会动。")
        print("    请把它改成 false，或确认上传的是最新配置。")
        print("=" * 68)
        return 1

    if args.rate > 0:
        cfg_dict = getattr(cfg, "data", None)
        if isinstance(cfg_dict, dict):
            cfg_dict.setdefault("servo", {})["max_rate_deg_per_s"] = args.rate

    try:
        servo = create_servo(cfg)
    except ActuatorError as exc:
        print(f"❌ {exc}")
        return 1

    name = type(servo).__name__
    print("=" * 68)
    print("舵机循环演示")
    print("=" * 68)
    print(f"  {servo.describe()}")
    if name == "SimulatedServo":
        print("  ⚠️ 这是模拟舵机，不会真的动！")
        return 1

    a_open = cfg.get_float("servo.angle_open_deg", 180.0)
    a_close = cfg.get_float("servo.angle_close_deg", 0.0)
    if args.span > 0:
        a_close = max(0.0, a_open - args.span)

    rate = args.rate if args.rate > 0 else cfg.get_float("servo.max_rate_deg_per_s", 200.0)
    move_s = abs(a_open - a_close) / max(1.0, rate)

    print(f"  张开角度   {a_open:.0f}°")
    print(f"  闭合角度   {a_close:.0f}°")
    print(f"  速率       {rate:.0f} °/s   （单程约 {move_s:.2f}s）")
    print(f"  总时长     {args.seconds:.0f}s   每端保持 {args.hold:.1f}s")
    print()
    print("开始，Ctrl+C 停止")
    print()

    t_start = time.monotonic()
    cycles = 0

    def move_to(target: float, label: str) -> int:
        servo.set_angle(target)
        n = 0
        t0 = time.monotonic()
        while time.monotonic() - t0 < move_s + 0.5:
            servo.update()
            n += 1
            time.sleep(0.01)
            if servo.at_target:
                break
        return n

    try:
        while time.monotonic() - t_start < args.seconds:
            el = time.monotonic() - t_start
            n_open = move_to(a_open, "张开")
            print(f"  [{el:5.1f}s] 张开  角度={servo.angle:6.1f}°  "
                  f"update×{n_open}")
            time.sleep(args.hold)

            el = time.monotonic() - t_start
            n_close = move_to(a_close, "闭合")
            print(f"  [{el:5.1f}s] 闭合  角度={servo.angle:6.1f}°  "
                  f"update×{n_close}")
            time.sleep(args.hold)

            cycles += 1
    except KeyboardInterrupt:
        print("\n中断")
    finally:
        # 结束时回到张开位置，避免夹爪一直咬着
        try:
            servo.set_angle(a_open)
            t0 = time.monotonic()
            while time.monotonic() - t0 < 2.0 and not servo.at_target:
                servo.update()
                time.sleep(0.01)
        except Exception:  # noqa: BLE001
            pass
        servo.disable()

    elapsed = time.monotonic() - t_start
    print()
    print("=" * 68)
    print(f"完成： {cycles} 个完整循环，用时 {elapsed:.1f}s")
    print(f"       平均每循环 {elapsed/max(1,cycles):.2f}s")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())