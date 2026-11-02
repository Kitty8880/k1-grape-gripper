#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
逐步下降找"最小安全脉宽"—— 装传感器时用。

为什么需要它
------------
脉宽 → 开口的映射是**两指之间的净间隙**。
每贴一只传感器（约 15mm 厚），净间隙就减少约 15mm。
现在两指都贴了传感器，同样的脉宽会让两指**更靠近**，
用单传感器时代的下限（650us）有压坏传感器的风险。

所以要先抬高下限（config 里的 pulse_min_us），再用本工具
**一步一步往下走**，手工确认什么时候两指（或传感器）开始接触。

安全设计
--------
    ① 起步在全开位置，不会一上来就闭合
    ② 每步只走 --step 微秒（默认 10），走完**停下来等你按回车**
    ③ 任何时候按 q 回车立即退回全开并退出
    ④ 永远不低于 config 里的 pulse_min_us（不会越过你设的底线）
    ⑤ Ctrl+C 立即退回全开
    ⑥ 全程可选读力传感器，接触时会看到读数变化

用法::

    # 用配置里的下限起步，每步 10us，人工确认
    sudo python3 apps/servo_step_test.py

    # 指定起点和步长（想从更开的位置开始）
    sudo python3 apps/servo_step_test.py --start 950 --step 10

    # 不读传感器（只想看机械位置）
    sudo python3 apps/servo_step_test.py --no-sensor

判读
----
    · 两指的**受力面**（贴传感器那一面）刚开始接触 → 这就是接触点
    · 建议把 config 的 pulse_min_us 设为「接触点 + 40~50us」
    · 如果看到传感器读数明显上升，说明已经在压传感器了 → 立即按 q
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


def main() -> int:
    ap = argparse.ArgumentParser(description="逐步下降找最小安全脉宽")
    ap.add_argument("--config", default=None)
    ap.add_argument("--start", type=float, default=None,
                    help="起始脉宽(us)，默认取 pulse_max_us")
    ap.add_argument("--step", type=float, default=10.0,
                    help="每步下降多少微秒")
    ap.add_argument("--floor", type=float, default=None,
                    help="临时下限覆盖（测试用）。默认取 config 的 pulse_min_us。"
                         "⚠️ 设得比 config 低会绕过安全下限，请格外小心")
    ap.add_argument("--settle", type=float, default=0.8,
                    help="每步后的稳定等待（秒）")
    ap.add_argument("--no-sensor", action="store_true",
                    help="不读力传感器")
    ap.add_argument("--repeat", type=int, default=1,
                    help="每个位置重复读几次力（取平均）")
    args = ap.parse_args()

    cfg = Config.load(args.config or default_config_path())
    pmin_cfg = cfg.get_float("servo.pulse_min_us", 800.0)
    pmax = cfg.get_float("servo.pulse_max_us", 1450.0)
    pin = cfg.get_int("servo.gpio_pin", 49)
    # --floor 是"测试时临时往下探"用的；不给就用 config 的安全下限
    pmin = args.floor if args.floor is not None else pmin_cfg
    start = args.start if args.start is not None else pmax

    print("=" * 70)
    print("逐步下降找最小安全脉宽")
    print("=" * 70)
    print(f"  舵机引脚    GPIO_{pin}（物理引脚 22）")
    print(f"  全开脉宽    {pmax:.0f}us")
    if args.floor is not None and args.floor < pmin_cfg:
        print(f"  **本次下限** {pmin:.0f}us   ⚠️ 已低于 config 安全值 "
              f"{pmin_cfg:.0f}us（测试专用，请盯着受力面）")
    else:
        print(f"  **安全下限** {pmin:.0f}us   ← 本工具不会低于这个值")
    print(f"  起始位置    {start:.0f}us")
    print(f"  每步步长    {args.step:.0f}us")
    print()
    print("  ⚠️ 操作提示：")
    print("     · 手指放在电源开关上")
    print("     · 眼睛盯着两指的受力面（贴传感器那一面）什么时候碰上")
    print("     · 按 q + 回车 可随时退回全开")
    print("=" * 70)
    print()

    if start < pmin:
        print(f"❌ 起始脉宽 {start:.0f} 小于安全下限 {pmin:.0f}")
        return 1

    # ---- 舵机 ----
    try:
        from grape_gripper.actuator import BusyPwmServo
        servo = BusyPwmServo(gpio_pin=pin, period_us=20000.0,
                             pulse_min_us=pmin, pulse_max_us=pmax,
                             angle_min=0.0, angle_max=180.0)
    except Exception as exc:  # noqa: BLE001
        print(f"❌ 舵机初始化失败：{exc}")
        return 1

    # ---- 传感器（可选）----
    adc = None
    if not args.no_sensor:
        try:
            from grape_gripper.sensor import create_adc
            adc = create_adc(cfg)
            print(f"✅ 传感器就绪：{adc.describe()}")

            # ⚠️ 零点必须用**中位数**，不能用均值！
            #    PS01 会周期性冒出一个大尖峰（A +11,300 / B +6,800 计数），
            #    均值会被拉高约 1,900 计数 = 1.2N，导致"空载时就显示有受力"。
            #    实测踩过这个坑：所有 Δ 都偏移 −1,900，看起来像一直在受力。
            def robust_zero(ch: int, n: int = 60) -> tuple[float, float]:
                vals = []
                for _ in range(n):
                    vals.append(adc.read_raw(ch))
                    time.sleep(0.03)
                vals.sort()
                med = vals[len(vals) // 2]
                mean = sum(vals) / len(vals)
                return med, mean

            med_a, mean_a = robust_zero(0)
            med_b, mean_b = robust_zero(1)
            base_a, base_b = med_a, med_b
            print(f"   空载零点（中位数）：A {med_a:,.0f}   B {med_b:,.0f}")
            print(f"   同批均值（会被尖峰拉高）：A {mean_a:,.0f}（{mean_a-med_a:+,.0f}）"
                  f"   B {mean_b:,.0f}（{mean_b-med_b:+,.0f}）")
            if abs(mean_a - med_a) > 500 or abs(mean_b - med_b) > 500:
                print("   ⚠️ 尖峰干扰明显，已自动改用中位数作零点")
            print()
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️ 传感器不可用（{exc}）—— 只测机械位置")
            adc = None

    pulse = start
    log = []
    try:
        print(f"→ 先到全开位置 {pmax:.0f}us")
        servo.set_pulse_us(pmax)
        time.sleep(1.2)
        print(f"→ 走到起点 {start:.0f}us")
        servo.set_pulse_us(start)
        time.sleep(args.settle)
        print()
        print(f"{'脉宽(us)':>10} {'A读数':>12} {'B读数':>12} {'ΔA':>10} {'ΔB':>10}")
        print("-" * 60)

        while pulse >= pmin:
            servo.set_pulse_us(pulse)
            time.sleep(args.settle)

            if adc is not None:
                # 同样用中位数，避免单个尖峰伪造出"受力"
                sa, sb = [], []
                for _ in range(max(3, args.repeat)):
                    sa.append(adc.read_raw(0))
                    sb.append(adc.read_raw(1))
                    time.sleep(0.12)
                sa.sort()
                sb.sort()
                ra, rb = sa[len(sa) // 2], sb[len(sb) // 2]
                da, db = ra - base_a, rb - base_b
                flag = ""
                if abs(da) > 800 or abs(db) > 800:
                    flag = "   ← ⚠️ 有受力迹象"
                print(f"{pulse:>10.0f} {ra:>12,.0f} {rb:>12,.0f} "
                      f"{da:>+10,.0f} {db:>+10,.0f}{flag}")
                log.append((pulse, da, db))
            else:
                print(f"{pulse:>10.0f} {'—':>12} {'—':>12}")

            nxt = pulse - args.step
            if nxt < pmin:
                print()
                print(f"已到下限于 {pmin:.0f}us，停止")
                break

            print()
            try:
                ans = input(f"  两指受力面碰上了吗？"
                            f"（回车继续降到 {nxt:.0f}us / 输入 n 结束）... ")
            except EOFError:
                break
            if ans.strip().lower() in ("n", "q", "no", "停"):
                print(f"  → 停在这里：{pulse:.0f}us")
                break
            pulse = nxt

        # ---- 总结 ----
        print()
        print("=" * 70)
        print("结果")
        print("=" * 70)
        if adc is not None and log:
            print(f"{'脉宽(us)':>10} {'ΔA':>12} {'ΔB':>12}")
            print("-" * 40)
            for p, da, db in log:
                print(f"{p:>10.0f} {da:>+12,.0f} {db:>+12,.0f}")
            print()
        print(f"  你停下的位置：{pulse:.0f}us")
        print()
        print("  建议：")
        print(f"    · 若 {pulse:.0f}us 是「刚好接触」，")
        print(f"      把 config.yaml 的 pulse_min_us 设为 "
              f"{pulse + 45:.0f}（留 45us 余量）")
        print(f"    · 然后重新标定 grasp_policy 里的")
        print(f"      approach_pulse / grasp_pulse（单传感器时代的值已失效）")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n中断 —— 退回全开")
    finally:
        try:
            servo.set_pulse_us(pmax)
            time.sleep(1.0)
        except Exception:  # noqa: BLE001
            pass
        for obj in (servo, adc):
            if obj is None:
                continue
            try:
                obj.close()
            except Exception:  # noqa: BLE001
                pass

    return 0


if __name__ == "__main__":
    sys.exit(main())