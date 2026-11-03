#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
夹爪手势 —— 用机械动作表达状态，代替屏幕和扬声器。

为什么是"手势"而不是声音
------------------------
本项目交付形态是「上电即待命、不接显示器、不接键盘」的独立终端。
但开发时一切状态都靠 SSH 看日志，交付后**没人看得见**。

真实产品通常会加蜂鸣器或小屏，本项目**刻意不加硬件**，理由：

    1. 用户明确不要扬声器
    2. 夹爪本来就是个能动的执行器 —— 让它自己"说话"最省
    3. 视觉上更直观：评委看视频时，夹爪动两下比听一声"叮"更清楚
    4. 顺带证明舵机通路是好的（开机自检的一部分）

手势设计
--------
    就绪 ready     张开 → 半闭 → 张开 → 半闭 → 张开
                   两下"招手"，一眼看出"我准备好了"

    应答 ack       半闭 → 微开
                   很短的抖动，表示"听到了"

    开始 grasp     缓缓闭合到预闭合位
                   表示"要动手了"

    完成 done      张开 → 闭合 → 张开（捏一下空气）
                   表示"做完了"

    失败 fail      快速抖动两下（半闭-开-半闭-开，间隔很短）
                   表示"出问题了"

安全约束
--------
所有手势**都不会低于 pulse_min_us（机械安全下限 750µs）**，
半闭位置默认 950µs（开口约 54mm），远大于任何被抓物体 ——
所以手势过程中不会夹到东西，也不会压到传感器。

用法::

    # 播一个手势（默认走 config 里的舵机参数）
    sudo python3 apps/servo_gesture.py ready
    sudo python3 apps/servo_gesture.py done

    # 列出所有手势
    sudo python3 apps/servo_gesture.py --list

    # 调节速度
    sudo python3 apps/servo_gesture.py ready --speed 1.5
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

#: 手势定义：[(目标脉宽 us, 停留秒数), ...]
#: 脉宽越大开口越大。OPEN 用 config 的 pulse_max_us，
#: HALF 是"半闭"—— 留足开口，保证不会夹到东西。
GESTURES = {
    "ready": [("OPEN", 0.0), ("HALF", 0.0), ("OPEN", 0.0),
              ("HALF", 0.0), ("OPEN", 0.0)],
    "ack": [("HALF", 0.0), ("OPEN", 0.0)],
    "grasp": [("PRE", 0.0)],
    "done": [("OPEN", 0.0), ("HALF", 0.0), ("OPEN", 0.0)],
    "fail": [("HALF", 0.0), ("OPEN", 0.0), ("HALF", 0.0), ("OPEN", 0.0)],
}

#: 每个手势的默认速度倍数（1.0 = 正常；失败手势要急促）
SPEED = {"ready": 1.0, "ack": 1.6, "grasp": 0.8,
         "done": 1.0, "fail": 2.2}

DESC = {
    "ready": "就绪 —— 张开/半闭 两下招手",
    "ack": "应答 —— 短促抖动一次",
    "grasp": "开始 —— 缓缓闭合到预闭合位",
    "done": "完成 —— 捏一下空气",
    "fail": "失败 —— 快速抖动两下",
}


def main() -> int:
    ap = argparse.ArgumentParser(description="夹爪手势")
    ap.add_argument("gesture", nargs="?", default=None,
                    help="手势名（ready / ack / grasp / done / fail）")
    ap.add_argument("--list", action="store_true", help="列出手势")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="速度倍数（越大越快）")
    ap.add_argument("--half-us", type=float, default=950.0,
                    help="「半闭」位置的脉宽（默认 950，开口约 54mm）")
    ap.add_argument("--step", type=float, default=18.0,
                    help="每次脉宽变化的步长（µs），越小越平滑")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    if args.list or not args.gesture:
        print("可用手势：")
        for k in ("ready", "ack", "grasp", "done", "fail"):
            pts = " → ".join(p[0] for p in GESTURES[k])
            print(f"  {k:<8} {DESC[k]:<28} ({pts})")
        print()
        print("用法： sudo python3 apps/servo_gesture.py ready")
        return 0 if args.list else 2

    if args.gesture not in GESTURES:
        print(f"❌ 未知手势：{args.gesture}")
        print("   可用：" + " / ".join(GESTURES))
        return 2

    cfg = Config.load(args.config or default_config_path())
    pin = cfg.get_int("servo.gpio", 49)
    pmin = cfg.get_float("servo.pulse_min_us", 750.0)
    pmax = cfg.get_float("servo.pulse_max_us", 1450.0)
    freq = cfg.get_float("servo.freq_hz", 50.0)

    # 安全：半闭位置绝不能低于机械安全下限
    half = max(pmin, args.half_us)
    # 预闭合位置在 grasp_policy.<物体>.approach_pulse 下（默认取葡萄的）
    pre = cfg.get_float("grasp_policy.grape.approach_pulse", 940.0)
    if pre <= pmin:
        pre = 940.0

    point = {"OPEN": pmax, "HALF": half, "PRE": pre}

    from grape_gripper.actuator import BusyPwmServo
    servo = BusyPwmServo(gpio_pin=pin, period_us=1e6 / freq,
                         pulse_min_us=pmin, pulse_max_us=pmax)
    speed = args.speed * SPEED.get(args.gesture, 1.0) or 1.0
    step = max(2.0, args.step * speed)

    def glide(to_us: float) -> None:
        """按步长平滑移动到目标脉宽（避免舵机猛甩）。"""
        cur = getattr(glide, "cur", pmax)
        to_us = max(pmin, min(pmax, to_us))
        if to_us > cur:
            p = cur
            while p < to_us:
                p = min(to_us, p + step)
                servo.set_pulse_us(p)
                time.sleep(0.012 / speed)
        else:
            p = cur
            while p > to_us:
                p = max(to_us, p - step)
                servo.set_pulse_us(p)
                time.sleep(0.012 / speed)
        glide.cur = to_us

    glide.cur = pmax
    try:
        servo.set_pulse_us(pmax)
        time.sleep(0.25)

        for key, hold in GESTURES[args.gesture]:
            glide(point[key])
            time.sleep((hold + 0.12) / speed)

        print(f"✅ 手势完成：{args.gesture}（{DESC[args.gesture]}）")
        return 0
    except KeyboardInterrupt:
        print("\n中断")
        return 130
    finally:
        # 松手时张开，保证下次能夹
        try:
            servo.set_pulse_us(pmax)
            time.sleep(0.2)
        except Exception:  # noqa: BLE001
            pass
        try:
            servo.close()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    sys.exit(main())