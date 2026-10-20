#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USB 方案完整测试：CH340 插在 K1 的 USB 口上，传感器接 CH340。

自动完成：
    1. 找到 /dev/ttyUSB*
    2. 读上电初始化日志（8 秒）
    3. 发送 gnst
    4. 读响应（8 秒）
    5. 若失败，自动换波特率再试（115200 / 118154 / 109714 ...）
    6. 打印完整诊断结论

用法（在 K1 上）::

    sudo python3 usb_ps01_test.py
    sudo python3 usb_ps01_test.py --seconds 6
"""
import argparse
import glob
import os
import sys
import time

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    print("缺少 pyserial: sudo apt install -y python3-serial")
    sys.exit(1)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass


def find_usb_serial() -> list[str]:
    devs = sorted(glob.glob("/dev/ttyUSB*")) + sorted(glob.glob("/dev/ttyACM*"))
    return devs


def drain(ser, seconds: float, label: str, max_lines: int = 80) -> tuple[int, list[str]]:
    total = 0
    lines: list[str] = []
    buf = ""
    t0 = time.time()
    while time.time() - t0 < seconds:
        d = ser.read(4096)
        if not d:
            continue
        total += len(d)
        buf += d.decode("utf-8", "ignore")
        while "\n" in buf:
            ln, buf = buf.split("\n", 1)
            ln = ln.strip()
            if ln:
                lines.append(ln)
                if len(lines) <= max_lines:
                    print(f"    {ln}")
    if buf.strip():
        lines.append(buf.strip())
    print(f"    [{label}] {total} 字节 / {len(lines)} 行")
    return total, lines


def try_baud(port: str, baud: int, init_s: float, after_s: float,
             dtr: bool, rts: bool) -> tuple[int, int, list[str]]:
    """返回 (总字节, 是否出现 RAW/started, 行列表)。"""
    try:
        ser = serial.Serial(
            port, baud, timeout=0.2,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            xonxoff=False, rtscts=False, dsrdtr=False,
        )
    except serial.SerialException as exc:
        print(f"  ❌ 打不开 {port}: {exc}")
        return 0, 0, []

    try:
        # 显式设置 DTR/RTS
        try:
            ser.dtr = dtr
            ser.rts = rts
        except (OSError, serial.SerialException):
            pass
        time.sleep(0.2)
        ser.reset_input_buffer()

        print(f"  读初始化（{init_s}s）...")
        n1, lines1 = drain(ser, init_s, "初始化")

        print(f"  发送 gnst\\r\\n ...")
        ser.write(b"gnst\r\n")
        ser.flush()
        time.sleep(0.3)

        print(f"  读响应（{after_s}s）...")
        n2, lines2 = drain(ser, after_s, "响应")

        all_lines = lines1 + lines2
        joined = "\n".join(all_lines)
        hits = 0
        if "RAW[" in joined:
            hits += 1
        if "measurement started" in joined:
            hits += 1
        if '{"status"' in joined:
            hits += 1
        return n1 + n2, hits, all_lines
    finally:
        ser.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="USB 方案 PS01 测试")
    ap.add_argument("--port", default=None, help="默认自动找 /dev/ttyUSB*")
    ap.add_argument("--init", type=float, default=8.0, help="读初始化秒数")
    ap.add_argument("--after", type=float, default=8.0, help="发命令后读秒数")
    args = ap.parse_args()

    print("=" * 70)
    print("PS01 USB 方案测试（CH340 插 K1 的 USB 口）")
    print("=" * 70)
    print()

    # 1. 找设备
    print("【1】查找串口设备")
    devs = find_usb_serial()
    if not devs:
        print("  ❌ 没有 /dev/ttyUSB* 或 /dev/ttyACM*")
        print("  请检查：")
        print("    - CH340 是否插在 K1 的 USB 口上")
        print("    - lsusb | grep -i ch34")
        print("    - dmesg | tail -5（看有没有插入日志）")
        return 2
    for d in devs:
        try:
            print(f"  {d}  权限 {oct(os.stat(d).st_mode)[-3:]}")
        except OSError:
            print(f"  {d}  (stat 失败)")
    port = args.port or devs[0]
    print(f"  使用: {port}")
    print()

    # 2. 先试标准 115200
    print("【2】标准 115200 波特率测试")
    n, hits, lines = try_baud(port, 115200, args.init, args.after, True, True)
    print()

    if n > 0 and hits:
        print("=" * 70)
        print(f"✅ 成功！收到 {n} 字节，且有 RAW/started 标志")
        print(f"   下一步: sudo python3 ~/ps01_driver.py --port {port} --seconds 10")
        print("=" * 70)
        return 0

    if n > 0:
        print("⚠️ 有数据但没有 RAW/started 标志，内容：")
        for ln in lines[:15]:
            print(f"    {ln}")
        print()

    # 3. 换波特率
    print("【3】尝试其他波特率")
    candidates = [118154, 109714, 115158, 113359, 57600, 9600]
    best = (115200, n)
    for baud in candidates:
        print(f"  --- {baud} ---")
        nb, hb, lb = try_baud(port, baud, min(args.init, 3.0),
                              min(args.after, 4.0), True, True)
        if nb > best[1]:
            best = (baud, nb)
        if nb > 0 and hb:
            print()
            print("=" * 70)
            print(f"✅ 在 {baud} bps 下成功！{nb} 字节")
            print(f"   下一步: sudo python3 ~/ps01_driver.py --port {port} "
                  f"--baud {baud} --seconds 10")
            print("=" * 70)
            return 0
        print()

    # 4. 试 DTR/RTS 组合
    print("【4】尝试 DTR/RTS 组合（115200）")
    for dtr in (True, False):
        for rts in (True, False):
            print(f"  --- DTR={dtr} RTS={rts} ---")
            nd, hd, ld = try_baud(port, 115200, 2.0, 3.0, dtr, rts)
            if nd > 100 and hd:
                print()
                print("=" * 70)
                print(f"✅ 成功！DTR={dtr} RTS={rts}，收到 {nd} 字节")
                print(f"   下一步: sudo python3 ~/ps01_driver.py --port {port} "
                      f"--dtr {int(dtr)} --rts {int(rts)} --seconds 10")
                print("=" * 70)
                return 0
            print()

    # 5. 结论
    print("=" * 70)
    print("结论：全部尝试失败")
    print("=" * 70)
    print(f"  最佳结果: {best[0]} bps -> {best[1]} 字节")
    print()
    print("  可能原因：")
    print("    1. K1 的 USB 口供电不足（CH340 的 3V3 带不动传感器）")
    print("    2. 传感器需要 PS01Tool 在连接时做的某个初始化动作")
    print("    3. CH340 的 TXD 到传感器 RXD 这段线有问题")
    print()
    print("  建议下一步：")
    print("    - 用万用表量 CH340 的 3V3 和 GND 之间电压（应 >3.2V）")
    print("    - 把 CH340 插回电脑，确认 PS01Tool 仍能工作")
    print("    - 考虑给传感器独立供电（AMS1117-3.3 模块 + 5V）")
    return 1


if __name__ == "__main__":
    sys.exit(main())
