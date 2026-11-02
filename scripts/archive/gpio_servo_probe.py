#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPIO 输出诊断 —— 排查"舵机为什么不动"。

直接绕过 gpiozero，用 lgpio 操作 GPIO，分步验证：
    第 1 步：能否打开 gpiochip 并申请引脚
    第 2 步：把引脚拉高 5 秒（用万用表量，应该是 3.3V）
    第 3 步：把引脚拉低 5 秒（应该是 0V）
    第 4 步：1 Hz 慢速翻转 10 次（接 LED 能明显看到闪）
    第 5 步：lgpio 的 tx_pwm 输出 50Hz 舵机信号
    第 6 步：慢速扫描脉宽 1000 → 2000 → 1000

没有万用表也能判断：第 2/3 步时用 LED + 电阻接在信号脚和 GND 之间，
应该看到亮/灭。

用法::

    sudo python3 gpio_servo_probe.py --pin 34
    sudo python3 gpio_servo_probe.py --pin 34 --step 2      # 只做第 2 步
"""
from __future__ import annotations

import argparse
import sys
import time

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description="GPIO 输出诊断")
    ap.add_argument("--pin", type=int, default=34, help="GPIO 编号（不是物理引脚号）")
    ap.add_argument("--step", type=int, default=0, help="只跑某一步（1~6）")
    ap.add_argument("--chip", type=int, default=0)
    args = ap.parse_args()

    try:
        import lgpio
    except ImportError:
        print("缺少 lgpio：sudo apt install -y python3-lgpio")
        return 1

    pin = args.pin
    print("=" * 66)
    print(f"GPIO 输出诊断   GPIO{pin}   gpiochip{args.chip}")
    print("=" * 66)

    # ---- 打开 gpiochip ----
    try:
        h = lgpio.gpiochip_open(args.chip)
        print(f"✅ 打开 gpiochip{args.chip} 成功（handle={h}）")
    except Exception as exc:  # noqa: BLE001
        print(f"❌ 打不开 gpiochip{args.chip}：{exc}")
        print("   试试： sudo chmod a+rw /dev/gpiochip0")
        return 1

    # ---- 申请引脚 ----
    try:
        lgpio.gpio_claim_output(h, pin)
        print(f"✅ 申请 GPIO{pin} 为输出成功")
    except Exception as exc:  # noqa: BLE001
        print(f"❌ 申请 GPIO{pin} 失败：{exc}")
        print("   可能是被别的驱动占用，换个引脚试试")
        lgpio.gpiochip_close(h)
        return 1

    def want(step: int) -> bool:
        return args.step == 0 or args.step == step

    try:
        # ---- 第 1 步：拉高 ----
        if want(1):
            print()
            print("【第 1 步】引脚拉高 5 秒")
            print("  → 用万用表量「信号脚 ↔ GND」，应该读到 3.3V")
            print("  → 接 LED（串 1kΩ 电阻）应该亮")
            lgpio.gpio_write(h, pin, 1)
            for i in range(5, 0, -1):
                print(f"    {i}...", end="\r", flush=True)
                time.sleep(1)
            print("    完成                  ")

        # ---- 第 2 步：拉低 ----
        if want(2):
            print()
            print("【第 2 步】引脚拉低 5 秒")
            print("  → 万用表应该读到 0V；LED 应该灭")
            lgpio.gpio_write(h, pin, 0)
            for i in range(5, 0, -1):
                print(f"    {i}...", end="\r", flush=True)
                time.sleep(1)
            print("    完成                  ")

        # ---- 第 3 步：1Hz 翻转 ----
        if want(3):
            print()
            print("【第 3 步】1 Hz 翻转 10 次（LED 应该明显闪烁）")
            for i in range(10):
                lgpio.gpio_write(h, pin, 1)
                time.sleep(0.5)
                lgpio.gpio_write(h, pin, 0)
                time.sleep(0.5)
                print(f"    第 {i+1} 次翻转完成")

        # ---- 第 4 步：tx_pwm 输出舵机信号 ----
        if want(4):
            print()
            print("【第 4 步】lgpio 的 tx_pwm 输出 50Hz / 1500us（舵机中位）")
            print("  → 舵机应该转到中位并保持")
            try:
                lgpio.gpio_free(h, pin)
            except Exception:  # noqa: BLE001
                pass
            try:
                lgpio.tx_pwm(h, pin, 50, 7.5)   # 50Hz, 7.5% = 1500us
                print("    ✅ tx_pwm 已启动，保持 6 秒...")
                time.sleep(6)
                lgpio.tx_pwm(h, pin, 50, 0)
                print("    已停止 PWM")
            except Exception as exc:  # noqa: BLE001
                print(f"    ❌ tx_pwm 失败：{exc}")

        # ---- 第 5 步：慢速扫描脉宽 ----
        if want(5):
            print()
            print("【第 5 步】慢速扫描脉宽（观察夹爪开合）")
            try:
                lgpio.gpio_free(h, pin)
            except Exception:  # noqa: BLE001
                pass
            seq = [1500, 1400, 1300, 1200, 1100, 1000,
                   1000, 1100, 1200, 1300, 1400, 1500]
            for pulse in seq:
                duty = pulse / 20000.0 * 100.0     # 周期 20000us
                try:
                    lgpio.tx_pwm(h, pin, 50, duty)
                    print(f"    脉宽 {pulse}us  (占空比 {duty:.2f}%)")
                except Exception as exc:  # noqa: BLE001
                    print(f"    ❌ {pulse}us 失败：{exc}")
                    break
                time.sleep(1.0)
            try:
                lgpio.tx_pwm(h, pin, 50, 0)
            except Exception:  # noqa: BLE001
                pass

    except KeyboardInterrupt:
        print("\n中断")
    finally:
        try:
            lgpio.gpio_write(h, pin, 0)
        except Exception:  # noqa: BLE001
            pass
        try:
            lgpio.gpio_free(h, pin)
        except Exception:  # noqa: BLE001
            pass
        lgpio.gpiochip_close(h)
        print()
        print("=" * 66)
        print("诊断结束")
        print("=" * 66)

    return 0


if __name__ == "__main__":
    sys.exit(main())