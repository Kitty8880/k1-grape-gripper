#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PS01 原始数据抓取 —— 直接把传感器吐出来的每一行原样打印，用于诊断。

和 ps01_driver.py 的区别：本脚本不做任何解析和过滤，
连"不认识的行走"也打印出来，所以能看清传感器到底在发什么。

用法::

    sudo python3 scripts/ps01_rawdump.py
    sudo python3 scripts/ps01_rawdump.py --seconds 5 --show 100
    sudo python3 scripts/ps01_rawdump.py --no-start      # 不发启动命令，只看有没有自发数据
    sudo python3 scripts/ps01_rawdump.py --dtr 0 --rts 0 # 改变 DTR/RTS 组合
"""
from __future__ import annotations

import argparse
import re
import sys
import time

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

try:
    import serial
except ImportError:
    print("缺少 pyserial：sudo apt install -y python3-serial")
    sys.exit(1)

START_CMD = b"\xd4gnst\r\n"      # 含 0xD4 唤醒字节
STOP_CMD = b"\xd4gned\r\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="PS01 原始数据抓取")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--show", type=int, default=60, help="最多打印多少行")
    ap.add_argument("--no-start", action="store_true", help="不发启动命令")
    ap.add_argument("--data-only", action="store_true",
                    help="只看 RAW 数据行，并把索引/通道/数值解析出来")
    ap.add_argument("--async-start", action="store_true",
                    help="不等初始化，打开就发命令（复现旧行为，用于对比）")
    ap.add_argument("--skip", type=int, default=0,
                    help="跳过前 N 行（用来越过初始化日志）")
    ap.add_argument("--dtr", type=int, default=None, choices=[0, 1])
    ap.add_argument("--rts", type=int, default=None, choices=[0, 1])
    args = ap.parse_args()

    print("=" * 70)
    print("PS01 原始数据抓取")
    print("=" * 70)
    print(f"  端口    {args.port} @ {args.baud}")
    print(f"  时长    {args.seconds}s")
    print(f"  启动命令 {'不发送' if args.no_start else ' '.join(f'{b:02X}' for b in START_CMD)}")
    if args.dtr is not None or args.rts is not None:
        print(f"  DTR={args.dtr}  RTS={args.rts}")

    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.1,
                            bytesize=serial.EIGHTBITS,
                            parity=serial.PARITY_NONE,
                            stopbits=serial.STOPBITS_ONE,
                            rtscts=False, dsrdtr=False)
    except Exception as exc:  # noqa: BLE001
        print(f"❌ 打不开串口：{exc}")
        return 1

    print(f"✅ 串口已打开")

    try:
        if args.dtr is not None:
            ser.dtr = bool(args.dtr)
        if args.rts is not None:
            ser.rts = bool(args.rts)
        print(f"  实际 DTR={ser.dtr} RTS={ser.rts}")
    except Exception as exc:  # noqa: BLE001
        print(f"  设置 DTR/RTS 失败：{exc}")

    ser.reset_input_buffer()

    # ---------------------------------------------------------------
    # ★ 第一步：等初始化日志打完
    #   打开串口会让 CH340 的 DTR/RTS 跳变 → 传感器复位 → 自动打印约 50 行
    #   初始化日志，最后一行是 "[INFO] Data callback registered"。
    #   在这之前发的命令会被丢弃！
    # ---------------------------------------------------------------
    if not args.no_start and not args.async_start:
        print("  → 等待初始化日志（寻找 'Data callback registered'）...")
        init_buf = b""
        init_lines = 0
        init_ok = False
        t_wait = time.time()
        while time.time() - t_wait < 6.0:
            chunk = ser.read(512)
            if not chunk:
                continue
            init_buf += chunk
            while b"\n" in init_buf:
                raw_line, init_buf = init_buf.split(b"\n", 1)
                init_lines += 1
                text = raw_line.decode("utf-8", "replace")
                if "Data callback registered" in text:
                    init_ok = True
                    break
            if init_ok:
                break
        print(f"    读到 {init_lines} 行；"
              f"{'✅ 初始化完成' if init_ok else '⚠️ 未看到完成标志（超时）'}")

    # ---------------------------------------------------------------
    # ★ 第二步：现在才发启动命令
    # ---------------------------------------------------------------
    if not args.no_start:
        ser.write(START_CMD)
        ser.flush()
        print(f"  → 已发送启动命令（{len(START_CMD)} 字节）")
        time.sleep(0.2)

    print()
    print("-" * 70)
    print("原始输出（非打印字符用 \\xNN 显示）")
    print("-" * 70)

    buf = b""
    total_bytes = 0
    total_lines = 0
    data_lines = 0
    shown = 0
    t0 = time.time()

    try:
        while time.time() - t0 < args.seconds:
            chunk = ser.read(512)
            if not chunk:
                continue
            total_bytes += len(chunk)
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                total_lines += 1
                text = line.decode("utf-8", "replace").rstrip("\r")

                if total_lines <= args.skip:
                    continue

                # ---- 只看数据行模式 ----
                if args.data_only:
                    m = re.search(
                        r"RAW\[\s*(\d+)\s*\]\s*:\s*0x([0-9A-Fa-f]+)"
                        r"\s*\(\s*status\s*=\s*0x([0-9A-Fa-f]+)\s*\)", text)
                    if not m:
                        continue
                    idx = int(m.group(1))
                    val = int(m.group(2), 16)
                    st = int(m.group(3), 16)
                    ch = {0: 3, 1: 2, 2: 1}.get(idx % 3, -1)
                    data_lines += 1
                    if shown < args.show:
                        print(f"{data_lines:5d} | idx={idx:<6d} idx%3={idx%3}  "
                              f"→PD{ch}  raw=0x{m.group(2):>8} = {val:>9,d}  "
                              f"st=0x{st:02X}")
                        shown += 1
                    elif shown == args.show:
                        print("      | ...（后续行不再显示）")
                        shown += 1
                    continue

                # ---- 原始模式 ----
                if shown < args.show:
                    printable = "".join(
                        c if 32 <= ord(c) < 127 else f"\\x{ord(c):02X}"
                        for c in text
                    )
                    print(f"{total_lines:5d} | {printable[:120]}")
                    shown += 1
                elif shown == args.show:
                    print(f"      | ...（后续行不再显示）")
                    shown += 1
    except KeyboardInterrupt:
        print("\n中断")
    finally:
        elapsed = time.time() - t0
        print()
        print("-" * 70)
        print(f"统计（{elapsed:.1f}s）：")
        print(f"  收到字节数    {total_bytes}")
        print(f"  收到行数      {total_lines}")
        if args.data_only:
            print(f"  其中数据行    {data_lines}")
        if elapsed > 0:
            print(f"  平均速率      {total_bytes/elapsed:.0f} 字节/秒，"
                  f"{total_lines/elapsed:.0f} 行/秒")
        if args.show and total_lines > args.show:
            print(f"  （只显示了前 {args.show} 行）")

        try:
            if not args.no_start:
                ser.write(STOP_CMD)
                ser.flush()
        except Exception:  # noqa: BLE001
            pass
        ser.close()

    print()
    print("判读：")
    if total_bytes == 0:
        print("  ❌ 一个字节都没收到 → 接线/供电/波特率问题")
    elif total_lines < 5:
        print("  ⚠️ 只有零星几行 → 可能是初始化没走完，或波特率不对")
    else:
        print(f"  ✅ 有数据流（{total_lines} 行）→ 通信正常")
        print("     若 python 库里读不到，比对上面行的格式和解析正则")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())