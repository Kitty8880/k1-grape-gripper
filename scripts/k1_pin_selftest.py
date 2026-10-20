#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K1 MUSE Pi Pro 40Pin 自检脚本

检查项：
  1. 电源状态（3.3V / 5V regulator）
  2. GPIO 控制器
  3. 串口设备
  4. I2C 总线
  5. 关键引脚复用（PADCONF 寄存器）

用法（在 K1 上）：
    sudo python3 k1_pin_selftest.py
"""
import glob
import os
import subprocess
import sys


def run(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return (r.stdout or "") + (r.stderr or "")
    except Exception as exc:  # noqa: BLE001
        return f"执行失败: {exc}"


def section(title):
    print()
    print("=" * 66)
    print(title)
    print("=" * 66)


def check_regulators():
    section("1. 电源状态")
    found_3v3 = found_5v = False
    for path in sorted(glob.glob("/sys/class/regulator/regulator.*")):
        try:
            name = open(os.path.join(path, "name")).read().strip()
        except OSError:
            continue
        if not name or name == "regulator-dummy":
            continue
        try:
            uvolts = int(open(os.path.join(path, "microvolts")).read().strip())
        except (OSError, ValueError):
            uvolts = 0
        try:
            state = open(os.path.join(path, "state")).read().strip()
        except OSError:
            state = ""
        mark = ""
        if 3000000 <= uvolts <= 3600000 and state == "enabled":
            mark = "  ← 3.3V 正常"
            found_3v3 = True
        elif 4700000 <= uvolts <= 5300000 and state == "enabled":
            mark = "  ← 5V 正常"
            found_5v = True
        print(f"  {name:<20} {uvolts/1e6:5.2f}V  {state or '-':<10}{mark}")

    print()
    print(f"  3.3V 输出: {'✅ 正常' if found_3v3 else '❌ 未找到'}")
    print(f"  5V  输出: {'✅ 正常' if found_5v else '❌ 未找到'}")
    print()
    print("  提示：3.3V/5V 都正常说明板子供电没问题；")
    print("        如果外设不工作，问题在接线或外设本身。")


def check_gpio():
    section("2. GPIO 控制器")
    chips = []
    try:
        for line in open("/sys/class/gpio/gpiochip0/ngpio"):
            pass
    except OSError:
        pass
    out = run(["gpiodetect"])
    print(out.rstrip() or "  gpiodetect 未安装（sudo apt install -y gpiod）")

    # 统计被占用的 GPIO
    info = run(["gpioinfo", "gpiochip0"])
    used = [ln for ln in info.splitlines()
            if "used" in ln and "unused" not in ln]
    print(f"\n  gpiochip0 上被占用的引脚数: {len(used)}")
    for ln in used[:15]:
        print(f"    {ln.strip()}")


def check_serial():
    section("3. 串口设备")
    for dev in sorted(glob.glob("/dev/ttyS*")):
        try:
            target = os.path.realpath(f"/sys/class/tty/{os.path.basename(dev)}/device")
            name = os.path.basename(target)
        except OSError:
            name = "?"
        try:
            perm = oct(os.stat(dev).st_mode)[-3:]
        except OSError:
            perm = "???"
        print(f"  {dev:<14} -> {name:<28} 权限 {perm}")

    print("\n  UART 对应关系（来自设备树）：")
    print("    ttyS0  = d4017000.serial   (主 uart0，PAD 68/69)")
    print("    ttyS2  = d4017100.uart     (uart2，PAD 21/22)")
    print("    ttyS10 = c0881000.r_uart0  (RCPU uart0，GPIO 48/49)")


def check_padconf():
    section("4. 关键引脚复用（PADCONF 寄存器）")
    base = 0xD401E000
    pins = {
        "GPIO47 (R_UART0_TXD 候选)": 47,
        "GPIO48 (R_UART0_RXD 候选)": 48,
        "GPIO49": 49,
        "GPIO70 (UART5_TXD, 脚7)": 70,
        "GPIO71 (UART5_RXD, 脚11)": 71,
    }
    for label, gpio in pins.items():
        addr = base + gpio * 4
        out = run(["busybox", "devmem", hex(addr)]).strip()
        mux = "?"
        if out.startswith("0x"):
            try:
                mux = int(out, 16) & 0x7
            except ValueError:
                pass
        desc = {0: "GPIO（未复用）", 1: "mux1", 2: "mux2",
                3: "mux3", 4: "mux4"}.get(mux, str(mux))
        print(f"  {label:<30} {out:<14} {desc}")


def check_i2c():
    section("5. I2C 总线")
    buses = sorted(glob.glob("/dev/i2c-*"))
    print(f"  找到 {len(buses)} 条 I2C 总线: "
          + " ".join(os.path.basename(b) for b in buses))
    print("\n  提示：用 i2cdetect -y -r <总线号> 扫描设备")
    print("        OV5647 摄像头地址是 0x36")
    print("        ADS1115 压力传感器默认 0x48")


def main() -> int:
    print("K1 MUSE Pi Pro 40Pin 自检")
    print("=" * 66)
    check_regulators()
    check_gpio()
    check_serial()
    check_padconf()
    check_i2c()
    print()
    print("=" * 66)
    print("自检完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
