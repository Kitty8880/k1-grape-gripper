#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
走配置文件驱动舵机 —— 验证 config.yaml → actuator → lgpio → 舵机 整条链路。

与 mg996r_test.py 的区别：
    mg996r_test.py  直接指定 GPIO 和脉宽，用于排查
    本脚本          完全按 config.yaml 的参数工作，
                    和 gripper.py 用的是同一条代码路径

用法::

    # 读配置，显示参数
    python3 apps/servo_config_test.py --show

    # 依次走到 全开 / 全闭 / 半开
    sudo python3 apps/servo_config_test.py --cycle

    # 走到指定角度（配置坐标系里的角度）
    sudo python3 apps/servo_config_test.py --angle 90

    # 平滑扫一遍，观察有没有卡顿/顶死
    sudo python3 apps/servo_config_test.py --sweep
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

from grape_gripper.actuator import ActuatorError, create_servo  # noqa: E402
from grape_gripper.config import Config, default_config_path     # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="按配置文件驱动舵机")
    ap.add_argument("--config", default=None, help="配置文件路径")
    ap.add_argument("--show", action="store_true", help="只显示参数")
    ap.add_argument("--cycle", action="store_true", help="走到 开 / 闭 / 半开")
    ap.add_argument("--sweep", action="store_true", help="平滑扫一遍")
    ap.add_argument("--angle", type=float, default=None, help="走到指定角度")
    args = ap.parse_args()

    cfg_path = args.config or default_config_path()
    print("=" * 66)
    print("按配置文件驱动舵机")
    print("=" * 66)
    print(f"配置文件： {cfg_path}")

    cfg = Config.load(cfg_path)

    # ---- 显示关键参数 ----
    driver = cfg.get_str("servo.driver", "?")
    pin = cfg.get_int("servo.gpio_pin", -1)
    pmin = cfg.get_float("servo.pulse_min_us", 0)
    pmax = cfg.get_float("servo.pulse_max_us", 0)
    period = cfg.get_float("servo.period_us", 20000)
    a_open = cfg.get_float("servo.angle_open_deg", 0)
    a_close = cfg.get_float("servo.angle_close_deg", 0)
    a_home = cfg.get_float("servo.angle_home_deg", 0)

    print()
    print("舵机参数：")
    print(f"  驱动后端      {driver}")
    print(f"  GPIO          {pin}   (gpiochip{cfg.get_int('servo.gpio_chip', 0)})")
    print(f"  周期          {period:.0f} us  ({1e6/period:.0f} Hz)")
    print(f"  脉宽范围      {pmin:.0f} ~ {pmax:.0f} us")
    print(f"  全开角度      {a_open:.1f}°")
    print(f"  全闭角度      {a_close:.1f}°")
    print(f"  初始角度      {a_home:.1f}°")
    print()
    print(f"  simulate      {cfg.get_bool('hardware.simulate', True)}")

    if args.show:
        return 0

    if cfg.get_bool("hardware.simulate", True):
        print()
        print("⚠️  config.yaml 里 hardware.simulate = true，会走模拟模式。")
        print("    要驱动真舵机，把它改成 false，或者用 --force-real 参数。")

    # ---- 创建舵机 ----
    print()
    print("正在创建舵机对象...")
    try:
        servo = create_servo(cfg)
    except ActuatorError as exc:
        print(f"❌ 创建失败：{exc}")
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"❌ 创建失败：{type(exc).__name__}: {exc}")
        return 1

    print(f"✅ {servo.describe()}")

    def go(angle: float, label: str, settle: float = 1.2) -> None:
        servo.set_angle(angle)
        # 等舵机走完（max_rate 限制下）
        t0 = time.time()
        while time.time() - t0 < settle:
            servo.update()
            time.sleep(0.02)
        pulse = pmin + (angle - 0.0) / 180.0 * (pmax - pmin)
        print(f"  → {label:<10} 角度 {angle:6.1f}°  脉宽 {pulse:6.0f}us")

    try:
        if args.angle is not None:
            print()
            go(args.angle, "指定角度", settle=1.5)

        elif args.sweep:
            print()
            print("平滑扫描（观察有没有卡顿或顶死）")
            seq = [a_home, a_open, (a_open + a_close) / 2, a_close, a_home]
            for a in seq:
                go(a, "扫描", settle=1.5)

        elif args.cycle:
            print()
            go(a_home, "初始位")
            time.sleep(0.3)
            go(a_open, "全开")
            time.sleep(0.3)
            go((a_open + a_close) / 2, "半开")
            time.sleep(0.3)
            go(a_close, "全闭")
            time.sleep(0.3)
            go(a_home, "回到初始位")

        else:
            print()
            print("（没有指定动作。用 --show / --cycle / --sweep / --angle N）")

    except KeyboardInterrupt:
        print("\n中断")
    finally:
        try:
            servo.disable()
            print()
            print("已卸力（舵机不再输出力矩）")
        except Exception:  # noqa: BLE001
            pass
        try:
            close = getattr(servo, "close", None)
            if callable(close):
                close()
        except Exception:  # noqa: BLE001
            pass

    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())