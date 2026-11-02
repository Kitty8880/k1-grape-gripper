# -*- coding: utf-8 -*-
"""
串口接收测试 —— 在 Windows 上接收 K1 从 r_uart0 发出的数据。

用途：验证 K1 的 GPIO47（脚 8）是否真的复用成了 UART TX。

用法::

    python win_serial_rx.py --port COM11 --seconds 20
"""

from __future__ import annotations

import argparse
import sys
import time

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    print("缺少 pyserial。pip install pyserial")
    sys.exit(1)


def main() -> int:
    ap = argparse.ArgumentParser(description="串口接收测试")
    ap.add_argument("--port", default=None)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--seconds", type=float, default=20.0)
    args = ap.parse_args()

    port = args.port
    if not port:
        for p in list_ports.comports():
            t = f"{p.device} {p.description} {p.hwid}".lower()
            if "ch340" in t or "1a86" in t or "cp210" in t or "ftdi" in t:
                port = p.device
                break
        if not port:
            ports = list(list_ports.comports())
            if ports:
                print("可用串口：")
                for p in ports:
                    print(f"  {p.device}  {p.description}")
            print("用 --port COMxx 指定")
            return 1

    print("=" * 60)
    print(f"接收测试：{port} @ {args.baud}，持续 {args.seconds} 秒")
    print("=" * 60)
    print("现在去 K1 上执行：")
    print("  printf 'HELLO_FROM_K1\\n' > /dev/ttyS10")
    print()

    try:
        ser = serial.Serial(port, args.baud, timeout=0.3)
    except serial.SerialException as exc:
        print(f"❌ 打不开 {port}: {exc}")
        return 2

    total = 0
    t0 = time.time()
    try:
        while time.time() - t0 < args.seconds:
            data = ser.read(256)
            if data:
                total += len(data)
                print(f"[{time.time()-t0:5.1f}s] 收到 {len(data)} 字节:")
                print(f"          hex : {data.hex(' ')}")
                print(f"          text: {data.decode('utf-8', 'replace')!r}")
    except KeyboardInterrupt:
        print("\n中断")
    finally:
        ser.close()

    print()
    print("=" * 60)
    if total:
        print(f"✅ 共收到 {total} 字节 —— TX 引脚工作正常！")
    else:
        print("❌ 一个字节都没收到")
        print("   可能：pinmux 没生效 / 接线不对 / 脚位错")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
