#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PS01 命令探测器 —— 自动尝试多种命令格式，找出能触发数据的那一条。

为什么不能用 printf：
    printf 'ON' > /dev/ttyUSB0  每次都会"打开设备→写→关闭"。
    打开串口会改变 DTR/RTS 电平，有些固件因此复位或拒绝响应。
    本脚本保持串口常开，逐个尝试命令，观察响应。

用法（在 K1 上）::

    sudo apt install -y python3-serial
    python3 ps01_probe.py

    # 只试某个命令
    python3 ps01_probe.py --only "ON\\n"

输出会告诉你：哪条命令让传感器开始吐数据。
"""

from __future__ import annotations

import argparse
import sys
import time

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

try:
    import serial
except ImportError:
    print("缺少 pyserial。先执行： sudo apt install -y python3-serial")
    sys.exit(1)

PORT = "/dev/ttyUSB0"
BAUD = 115200

# 候选命令：按可能性排序
# 依据：从 PS01Tool.exe 里逆向出的字符串包含 'ON' / 'OFF'，
#       且中文提示为"发送开始命令"、"采集已开启"
VARIANTS = [
    ("ON + LF",          b"ON\n"),
    ("ON + CRLF",        b"ON\r\n"),
    ("ON + CR",          b"ON\r"),
    ("ON 无结尾",         b"ON"),
    ("on + LF",          b"on\n"),
    ("ON + NULL",        b"ON\x00"),
    ("JSON cmd=ON",      b'{"cmd":"ON"}\n'),
    ("JSON cmd=on",      b'{"cmd":"on"}\n'),
    ("JSON cmd=start",   b'{"cmd":"start"}\n'),
    ("JSON cmd=acq",     b'{"cmd":"acq"}\n'),
    ("JSON start=1",     b'{"start":1}\n'),
    ("START + LF",       b"START\n"),
    ("start + LF",       b"start\n"),
    ("ACQ + LF",         b"ACQ\n"),
    ("数字 1 + LF",       b"1\n"),
    ("HEX 0x01",         b"\x01"),
]


def hexdump(data: bytes, limit: int = 120) -> str:
    """把字节转成可读形式：hex + ASCII。"""
    d = data[:limit]
    hx = " ".join(f"{b:02X}" for b in d)
    asc = "".join(chr(b) if 32 <= b < 127 else "." for b in d)
    tail = " …" if len(data) > limit else ""
    return f"{hx}{tail}\n         |{asc}|"


def drain(ser, seconds: float, label: str = "") -> bytes:
    """在 seconds 秒内持续读取，返回全部数据。"""
    buf = bytearray()
    t0 = time.time()
    while time.time() - t0 < seconds:
        try:
            n = ser.in_waiting
        except OSError:
            n = 0
        if n:
            chunk = ser.read(n)
            buf.extend(chunk)
            if label:
                print(f"    [{label}] +{len(chunk)} 字节: {chunk[:60]!r}")
        else:
            time.sleep(0.02)
    return bytes(buf)


def looks_like_data(payload: bytes) -> bool:
    """判断响应里是否包含测量数据（RAW[...] 或 JSON 且不是 status 消息）。"""
    text = payload.decode("utf-8", "replace")
    if "RAW[" in text:
        return True
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{") and ('"status"' not in line or 'value' in line):
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="PS01 命令探测")
    ap.add_argument("--port", default=PORT)
    ap.add_argument("--baud", type=int, default=BAUD)
    ap.add_argument("--only", default=None, help="只测一条命令（支持 \\n 转义）")
    ap.add_argument("--listen", type=float, default=1.2,
                    help="每条命令后监听多少秒")
    args = ap.parse_args()

    print("=" * 74)
    print("PS01 命令探测器")
    print("=" * 74)
    print(f"串口 {args.port} @ {args.baud} 8N1\n")

    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.1,
                            bytesize=serial.EIGHTBITS,
                            parity=serial.PARITY_NONE,
                            stopbits=serial.STOPBITS_ONE,
                            rtscts=False, dsrdtr=False)
    except serial.SerialException as exc:
        print(f"❌ 打不开 {args.port}: {exc}")
        print("   先执行： sudo chmod a+rw /dev/ttyUSB0")
        return 2

    try:
        ser.reset_input_buffer()
    except OSError:
        pass

    # 先看看不打扰它的时候有没有数据
    print("[0] 静默监听 2 秒（不发命令）…")
    idle = drain(ser, 2.0)
    if idle:
        print(f"    收到 {len(idle)} 字节：")
        print("    " + hexdump(idle))
        if looks_like_data(idle):
            print("\n✅ 不用发命令就在输出数据！")
            return 0
    else:
        print("    无数据（符合预期：需要命令触发）")

    variants = VARIANTS
    if args.only:
        raw = args.only.encode().decode("unicode_escape").encode("latin-1")
        variants = [(repr(raw), raw)]

    print(f"\n[1] 依次尝试 {len(variants)} 种命令格式，每条监听 "
          f"{args.listen} 秒\n")

    winner = None
    for i, (label, payload) in enumerate(variants, 1):
        print(f"  ({i}/{len(variants)}) {label:<18} 发送 {payload!r}")
        try:
            ser.reset_input_buffer()
        except OSError:
            pass
        try:
            ser.write(payload)
            ser.flush()
        except serial.SerialException as exc:
            print(f"      写入失败: {exc}")
            continue

        resp = drain(ser, args.listen)
        if not resp:
            print("      → 无响应")
            continue

        print(f"      → 收到 {len(resp)} 字节")
        print("      " + hexdump(resp).replace("\n", "\n      "))
        if looks_like_data(resp):
            winner = (label, payload)
            print("      ★★★ 这条产生了测量数据！ ★★★")
            break

    print()
    print("=" * 74)
    if winner:
        label, payload = winner
        print(f"✅ 找到命令：{label}  →  {payload!r}")
        print(f"\n在 K1 上的用法：")
        print(f"    printf '{payload.decode('latin-1')}' > /dev/ttyUSB0")
        print(f"然后用 Python 持续读取解析。")
    else:
        print("没找到能触发的命令。可能原因：")
        print("  1. 传感器需要重新上电（拔插 CH340）后再试")
        print("  2. 命令格式不在候选列表里 —— 需要用串口嗅探工具抓 PS01Tool 的实际发送")
        print("  3. 波特率不是 115200（试试 9600 / 57600 / 921600）")
        print("  4. 命令需要带校验和或帧头（如 0xAA 0x55 ...）")
        print("\n建议：把 CH340 插回 Windows，用串口监控软件抓 PS01Tool 的 TX 数据。")
    print("=" * 74)
    ser.close()
    return 0 if winner else 1


if __name__ == "__main__":
    sys.exit(main())
