# -*- coding: utf-8 -*-
"""
PS01 串口协议嗅探脚本（在你的 Windows 电脑上跑）

目的：搞清楚 PS01 压力传感器在串口上到底输出什么格式的数据，
      以便在 K1 上写解析代码。

前置：
    1. 关掉 PS01Tool.exe（串口独占，不能两个程序同时开）
    2. CH340 插在电脑上，传感器接好并供电（3.3V！）
    3. pip install pyserial

用法：
    python ps01_sniff.py                 # 自动找串口并监听
    python ps01_sniff.py --port COM18    # 指定串口
    python ps01_sniff.py --list          # 只列出可用串口
    python ps01_sniff.py --send "01 03 00 00 00 01 84 0A"   # 顺便发一条命令

输出：把收到的原始字节用十六进制 + ASCII 双栏打印，同时存一份到 ps01_raw.bin
"""

from __future__ import annotations

import argparse
import sys
import time
import os

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    print("缺少 pyserial。先执行：  pip install pyserial")
    sys.exit(1)

BAUD = 115200


def list_ports_cmd() -> int:
    print("可用串口：")
    ports = list(list_ports.comports())
    if not ports:
        print("  （一个都没有）")
        return 1
    for p in ports:
        print(f"  {p.device:<10} {p.description}")
        print(f"      HWID: {p.hwid}")
    return 0


def auto_pick_port() -> str | None:
    """优先挑 CH340。"""
    ports = list(list_ports.comports())
    for p in ports:
        text = f"{p.device} {p.description} {p.hwid}".lower()
        if "ch340" in text or "1a86" in text:
            return p.device
    # 退而求其次：任何 USB 串口
    for p in ports:
        if "usb" in (p.hwid or "").lower():
            return p.device
    return ports[0].device if ports else None


def hexdump(data: bytes, base_offset: int = 0) -> str:
    """标准 hexdump：偏移 + 16 字节十六进制 + ASCII。"""
    lines = []
    for i in range(0, len(data), 16):
        chunk = data[i:i + 16]
        hexpart = " ".join(f"{b:02X}" for b in chunk)
        asciipart = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        lines.append(f"  {base_offset + i:08X}  {hexpart:<48}  |{asciipart}|")
    return "\n".join(lines)


def parse_hex(text: str) -> bytes:
    cleaned = text.replace(",", " ").replace("0x", " ").replace("0X", " ")
    return bytes(int(x, 16) for x in cleaned.split() if x)


def main() -> int:
    ap = argparse.ArgumentParser(description="PS01 串口协议嗅探")
    ap.add_argument("--port", default=None, help="串口名，如 COM18")
    ap.add_argument("--baud", type=int, default=BAUD, help=f"波特率，默认 {BAUD}")
    ap.add_argument("--list", action="store_true", help="只列出可用串口")
    ap.add_argument("--seconds", type=float, default=20.0, help="监听时长（秒）")
    ap.add_argument("--send", default=None,
                    help="连接后先发送这条十六进制命令，如 \"01 03 00 00 00 01 84 0A\"")
    ap.add_argument("--send-text", default=None,
                    help="连接后先发送这段文本（会加换行）")
    ap.add_argument("--file", default="ps01_raw.bin", help="原始数据保存路径")
    args = ap.parse_args()

    if args.list:
        return list_ports_cmd()

    port = args.port or auto_pick_port()
    if not port:
        print("找不到可用串口。先插好 CH340，再跑 --list 看看。")
        return 1

    print("=" * 74)
    print("PS01 串口协议嗅探")
    print("=" * 74)
    print(f"串口   : {port}")
    print(f"波特率 : {args.baud}  8N1")
    print(f"时长   : {args.seconds} 秒")
    print()

    try:
        ser = serial.Serial(port, args.baud, timeout=1.0,
                            bytesize=serial.EIGHTBITS,
                            parity=serial.PARITY_NONE,
                            stopbits=serial.STOPBITS_ONE)
    except serial.SerialException as exc:
        print(f"❌ 打不开 {port}：{exc}")
        print("   可能原因：PS01Tool.exe 还开着（串口被独占）。先关掉它。")
        return 2

    print("✅ 已打开串口\n")

    # 可选：主动发命令
    if args.send:
        payload = parse_hex(args.send)
        print(f"→ 发送 {len(payload)} 字节：{payload.hex(' ').upper()}")
        ser.write(payload)
        ser.flush()
        time.sleep(0.3)
    if args.send_text:
        payload = (args.send_text + "\n").encode()
        print(f"→ 发送文本：{args.send_text!r}")
        ser.write(payload)
        ser.flush()
        time.sleep(0.3)

    print("开始监听。请用手按压传感器，方便对照数据变化。")
    print("按 Ctrl+C 提前结束。\n")
    print("-" * 74)

    buf = bytearray()
    total = 0
    t0 = time.time()
    last_report = t0

    try:
        while time.time() - t0 < args.seconds:
            chunk = ser.read(4096)
            if chunk:
                buf.extend(chunk)
                total += len(chunk)
                # 实时打印前若干字节，便于现场观察
                print(f"[{time.time() - t0:6.2f}s] 收到 {len(chunk):4d} 字节: "
                      f"{chunk[:24].hex(' ').upper()}"
                      f"{' …' if len(chunk) > 24 else ''}")
            now = time.time()
            if now - last_report >= 5.0:
                last_report = now
                rate = total / max(0.001, now - t0)
                print(f"  --- 累计 {total} 字节，平均 {rate:.0f} B/s ---")
    except KeyboardInterrupt:
        print("\n（用户中断）")

    ser.close()
    print("-" * 74)

    if not buf:
        print("\n⚠️ 一个字节都没收到。排查：")
        print("  1. 传感器是否供电？（3V3 引脚，3.3V）")
        print("  2. TXD/RXD 是否接反？（传感器的 TXD 要接转接板的 RXD）")
        print("  3. 波特率是否 115200？")
        print("  4. 是否需要在软件里点『开始采集』才会输出？（关键！）")
        return 3

    print(f"\n共收到 {len(buf)} 字节。十六进制转储（前 512 字节）：")
    print(hexdump(bytes(buf[:512])))

    with open(args.file, "wb") as fh:
        fh.write(bytes(buf))
    print(f"\n完整原始数据已保存：{os.path.abspath(args.file)}  ({len(buf)} 字节)")

    # 猜一下常见的包结构
    print("\n" + "=" * 74)
    print("初步分析")
    print("=" * 74)
    data = bytes(buf)
    for size in (2, 4, 6, 8, 10, 12, 16, 20):
        if len(data) >= size * 2 and len(data) % size == 0:
            print(f"  数据长度 {len(data)} 能被 {size} 整除 → 包长可能是 {size} 字节")
    # 找重复的同步头
    if len(data) > 8:
        first = data[:4]
        idx = data.find(first, 4)
        if idx > 0:
            print(f"  开头 4 字节 {first.hex(' ').upper()} 在第 {idx} 字节处重复 "
                  f"→ 包长可能是 {idx} 字节")
    print("\n把以上输出（以及 ps01_raw.bin）发给我，我来写解析代码。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
