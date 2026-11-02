#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
舵机最小对照测试 —— 定位"配置路径不动，mg996r_test.py 能动"的差异。

分三段对照，每段都打印详细信息：

  A 段：完全绕开我们的代码，直接用 gpiozero.Servo（和 mg996r_test.py 一样的写法）
  B 段：用 create_servo() 建对象，但**绕开速率限幅**直接写角度
  C 段：走正常的 set_angle() + update() 流程

哪一段不动，问题就在哪一层。

用法::

    sudo python3 apps/servo_minimal.py
"""
from __future__ import annotations

import os
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass


PIN = 49
PULSE_MIN = 650.0
PULSE_MAX = 1450.0
PERIOD = 20000.0


def banner(t: str) -> None:
    print()
    print("=" * 66)
    print(t)
    print("=" * 66)


def wait(sec: float = 1.5, note: str = "") -> None:
    if note:
        print(f"    （观察舵机：{note}）")
    time.sleep(sec)


def seg_a() -> bool:
    """A 段：直接用 gpiozero.Servo，完全照抄 mg996r_test.py 的写法。"""
    banner("A 段：直接用 gpiozero.Servo（对照组）")
    try:
        from gpiozero import Device, Servo
        from gpiozero.pins.lgpio import LGPIOFactory

        Device.pin_factory = LGPIOFactory(chip=0)
        print(f"  pin_factory = {Device.pin_factory!r}")

        servo = Servo(PIN,
                      min_pulse_width=PULSE_MIN / 1e6,
                      max_pulse_width=PULSE_MAX / 1e6,
                      frame_width=PERIOD / 1e6,
                      initial_value=None)
        print(f"  Servo 已创建： {servo!r}")

        for label, value in (("最大脉宽(张开)", 1.0),
                             ("中位", 0.0),
                             ("最小脉宽(闭合)", -1.0),
                             ("最大脉宽(张开)", 1.0)):
            servo.value = value
            print(f"    set value={value:+.1f}  →  {label}")
            wait(1.5)

        servo.detach()
        servo.close()
        print("  A 段结束")
        return True
    except Exception:  # noqa: BLE001
        print("  ❌ A 段异常：")
        traceback.print_exc()
        return False


def seg_b() -> bool:
    """B 段：用 create_servo()，但绕开速率限幅直接写角度。"""
    banner("B 段：create_servo() + 绕开速率限幅")
    try:
        from grape_gripper.actuator import create_servo
        from grape_gripper.config import Config, default_config_path

        cfg = Config.load(default_config_path())
        print(f"  simulate      = {cfg.simulate}")
        print(f"  servo.driver  = {cfg.get_str('servo.driver', '?')}")
        print(f"  servo.gpio_pin= {cfg.get_int('servo.gpio_pin', -1)}")
        print(f"  pulse range   = {cfg.get_float('servo.pulse_min_us', 0)}"
              f" ~ {cfg.get_float('servo.pulse_max_us', 0)} us")

        servo = create_servo(cfg)
        print(f"  实例类型      = {type(servo).__name__}")
        print(f"  describe      = {servo.describe()}")

        if type(servo).__name__ == "SimulatedServo":
            print("  ⚠️ 走的是模拟舵机！config.yaml 里 hardware.simulate 还是 true")
            print("     或者 sensor.adc / driver 配置没生效")
            return False

        for label, ang in (("180°(张开)", 180.0),
                           ("90°(半开)", 90.0),
                           ("0°(闭合)", 0.0),
                           ("180°(张开)", 180.0)):
            # 直接设内部状态并调用底层写硬件，绕开 update() 的速率限幅
            servo._angle = ang
            servo._enabled = True
            servo._apply_angle(ang)
            print(f"    _apply_angle({ang:.0f})  →  {label}")
            wait(1.5)

        servo.disable()
        print("  B 段结束")
        return True
    except Exception:  # noqa: BLE001
        print("  ❌ B 段异常：")
        traceback.print_exc()
        return False


def seg_c() -> bool:
    """C 段：走正常的 set_angle + update 流程。"""
    banner("C 段：正常流程 set_angle() + update()")
    try:
        from grape_gripper.actuator import create_servo
        from grape_gripper.config import Config, default_config_path

        cfg = Config.load(default_config_path())
        servo = create_servo(cfg)
        print(f"  {servo.describe()}")

        for label, ang in (("180°(张开)", 180.0),
                           ("0°(闭合)", 0.0),
                           ("180°(张开)", 180.0)):
            servo.set_angle(ang)
            t0 = time.monotonic()
            n = 0
            while time.monotonic() - t0 < 2.5:
                servo.update()
                n += 1
                time.sleep(0.02)
            print(f"    → {label}  目标={servo.target_angle:.1f} "
                  f"实际={servo.angle:.1f}  update次数={n} "
                  f"at_target={servo.at_target}")
            wait(0.8)

        servo.disable()
        print("  C 段结束")
        return True
    except Exception:  # noqa: BLE001
        print("  ❌ C 段异常：")
        traceback.print_exc()
        return False


def main() -> int:
    print("=" * 66)
    print("舵机最小对照测试")
    print("=" * 66)
    print(f"GPIO{PIN}  脉宽 {PULSE_MIN:.0f}~{PULSE_MAX:.0f}us  周期 {PERIOD:.0f}us")
    print(f"root = {os.geteuid() == 0}")
    if os.geteuid() != 0:
        print("⚠️ 不是 root，lgpio 可能打不开 /dev/gpiochip0，请用 sudo 运行")

    results = {}
    results["A"] = seg_a()
    results["B"] = seg_b()
    results["C"] = seg_c()

    banner("结果汇总")
    for k, v in results.items():
        print(f"  {k} 段： {'✅ 无异常' if v else '❌ 有问题'}")
    print()
    print("对照说明：")
    print("  A 动 / B 不动  → 问题在 create_servo 或 LgpioServo")
    print("  A 动 / B 动 / C 不动 → 问题在速率限幅或 update()")
    print("  A 不动         → 引脚或接线问题（但 mg996r_test.py 能动，矛盾）")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())