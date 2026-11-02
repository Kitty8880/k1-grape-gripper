#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PADCONF 复用值扫描 —— 用舵机当指示器，找出哪个复用值能让引脚输出 GPIO。

背景
----
K1 的引脚功能由 PADCONF 寄存器控制（pinctrl-single）。
如果某个脚的默认复用值不是 GPIO，那么写 GPIO 寄存器不会影响引脚电平，
舵机就收不到信号。

本脚本自动把复用值从 0 试到 7，每个值都发一段舵机信号，
由人观察舵机是否动作，从而定位正确的复用值。

安全
----
只修改**一个引脚**的 PADCONF（默认是引脚 32 / GPIO34），
这个脚当前只接着舵机信号线，改动风险很低。
退出时会自动恢复原始值。

用法::

    # 确保舵机信号线接在引脚 32，然后：
    sudo python3 padconf_mux_sweep.py

    # 指定别的引脚
    sudo python3 padconf_mux_sweep.py --gpio 35
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass


PADCONF_BASE = 0xD401E000


def devmem_read(addr: int) -> int:
    out = subprocess.run(["sudo", "busybox", "devmem", hex(addr)],
                         capture_output=True, text=True, timeout=10)
    text = (out.stdout or "").strip()
    m = re.search(r"0x([0-9A-Fa-f]+)", text)
    if not m:
        raise RuntimeError(f"读取 {hex(addr)} 失败：{text!r} {out.stderr!r}")
    return int(m.group(1), 16)


def devmem_write(addr: int, value: int) -> None:
    subprocess.run(["sudo", "busybox", "devmem", hex(addr), "32", hex(value)],
                   capture_output=True, text=True, timeout=10)


def main() -> int:
    ap = argparse.ArgumentParser(description="PADCONF 复用值扫描")
    ap.add_argument("--gpio", type=int, default=34, help="GPIO 编号")
    ap.add_argument("--physical", type=int, default=32, help="对应的物理引脚号（仅用于提示）")
    ap.add_argument("--values", default="0,1,2,3,4,5,6,7",
                    help="要尝试的复用值（逗号分隔）")
    args = ap.parse_args()

    try:
        import lgpio
    except ImportError:
        print("缺少 lgpio：sudo apt install -y python3-lgpio")
        return 1

    gpio = args.gpio
    addr = PADCONF_BASE + gpio * 4

    print("=" * 70)
    print(f"PADCONF 复用值扫描   GPIO{gpio}（物理引脚 {args.physical}）")
    print("=" * 70)
    print(f"寄存器地址： {hex(addr)}")
    print()
    print("⚠️ 开始前请确认：")
    print(f"   1. 舵机的橙色信号线接在物理引脚 {args.physical} 上")
    print("   2. 舵机电源已打开")
    print("   3. 共地已接好（舵机棕线 → GND，电源 GND → K1 另一个 GND 脚）")
    print("   4. 手放在电源开关上，异常时能立刻断电")
    print()
    input("   确认后按回车开始...")

    # 保存原值
    original = devmem_read(addr)
    print()
    print(f"原始 PADCONF = 0x{original:08X}（退出时会恢复）")

    # 打开 gpiochip 并申请引脚
    try:
        h = lgpio.gpiochip_open(0)
        try:
            lgpio.gpio_free(h, gpio)
        except Exception:  # noqa: BLE001
            pass
        lgpio.gpio_claim_output(h, gpio)
    except Exception as exc:  # noqa: BLE001
        print(f"❌ 申请 GPIO{gpio} 失败：{exc}")
        return 1

    values = [int(v) for v in args.values.split(",") if v.strip()]
    results: dict[int, str] = {}

    def emit_pwm(pulse_us: int, seconds: float) -> None:
        duty = pulse_us / 20000.0 * 100.0
        try:
            lgpio.tx_pwm(h, gpio, 50, duty)
        except Exception as exc:  # noqa: BLE001
            print(f"      tx_pwm 失败：{exc}")
            return
        time.sleep(seconds)

    try:
        for v in values:
            newval = (original & ~0x7) | (v & 0x7)   # 低 3 位作为复用值
            devmem_write(addr, newval)

            print()
            print("─" * 70)
            print(f"【复用值 {v}】 PADCONF = 0x{newval:08X}")
            print("─" * 70)

            # 发射一段明显的动作：先中位，再小角度，再回中位
            print("    发送 1500us ...")
            emit_pwm(1500, 1.5)
            print("    发送 1100us（应该往一个方向走）...")
            emit_pwm(1100, 2.0)
            print("    发送 1900us（应该往另一个方向走）...")
            emit_pwm(1900, 2.0)
            print("    回到 1500us ...")
            emit_pwm(1500, 1.0)
            lgpio.tx_pwm(h, gpio, 50, 0)
            time.sleep(0.3)

            ans = input("    ❓ 舵机刚才动了吗？ (y=动了 / n=没动 / q=退出) ").strip().lower()
            if ans.startswith("y"):
                results[v] = "✅ 舵机动作"
                print(f"    ★ 复用值 {v} 有效！")
                cont = input("    继续试其他值吗？(y/n，n=结束扫描) ").strip().lower()
                if cont.startswith("n"):
                    break
            elif ans.startswith("q"):
                break
            else:
                results[v] = "❌ 无反应"

    except KeyboardInterrupt:
        print("\n中断")
    finally:
        try:
            lgpio.tx_pwm(h, gpio, 50, 0)
        except Exception:  # noqa: BLE001
            pass
        try:
            lgpio.gpio_free(h, gpio)
        except Exception:  # noqa: BLE001
            pass
        lgpio.gpiochip_close(h)
        devmem_write(addr, original)

    # 汇总
    print()
    print("=" * 70)
    print("扫描结果")
    print("=" * 70)
    for v in values:
        if v in results:
            print(f"  复用值 {v}:  {results[v]}")
    print()
    print(f"PADCONF 已恢复为 0x{original:08X}")

    good = [v for v, r in results.items() if r.startswith("✅")]
    if good:
        print()
        print(f"★ 有效的复用值： {good}")
        print(f"  记住这个值，可以把引脚固定配置成它。")
    else:
        print()
        print("没有找到有效的复用值。可能的原因：")
        print("  1. 舵机的信号线/电源/共地有问题")
        print("  2. 舵机本身有问题（可以用另一个舵机对比）")
        print("  3. 引脚不是这个 GPIO 编号")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())