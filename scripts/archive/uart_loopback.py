#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UART 回环测试：需要先把指定两个引脚短接。

用法（在 K1 上）::

    # 默认测 ttyS10，发 3 次
    sudo python3 uart_loopback.py

    # 指定端口
    sudo python3 uart_loopback.py --port /dev/ttyS0
"""
import argparse
import sys
import time

try:
    import serial
except ImportError:
    print("缺少 pyserial: sudo apt install -y python3-serial")
    sys.exit(1)


def main() -> int:
    ap = argparse.ArgumentParser(description="UART 回环测试")
    ap.add_argument("--port", default="/dev/ttyS10")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()

    print("=" * 60)
    print("UART 回环测试")
    print(f"  端口: {args.port} @ {args.baud}")
    print("  前提: 对应的 TX 和 RX 引脚已经短接")
    print("=" * 60)

    try:
        s = serial.Serial(args.port, args.baud, timeout=0.5,
                          bytesize=serial.EIGHTBITS,
                          parity=serial.PARITY_NONE,
                          stopbits=serial.STOPBITS_ONE)
    except serial.SerialException as exc:
        print(f"打不开 {args.port}: {exc}")
        return 2

    ok = 0
    try:
        for i in range(args.rounds):
            payload = f"LOOPBACK_{i}_{time.time():.3f}\r\n".encode()
            s.reset_input_buffer()
            s.write(payload)
            s.flush()
            time.sleep(0.25)

            data = b""
            t0 = time.time()
            while time.time() - t0 < 0.6:
                chunk = s.read(256)
                if chunk:
                    data += chunk
                if len(data) >= len(payload):
                    break

            if payload.strip() in data:
                ok += 1
                print(f"  [{i+1}] OK   发出 {payload.strip()!r} -> 收回 {data.strip()!r}")
            else:
                print(f"  [{i+1}] FAIL 发出 {payload.strip()!r} -> 收回 {data!r}")
    finally:
        s.close()

    print()
    print("=" * 60)
    if ok == args.rounds:
        print(f"全部成功 ({ok}/{args.rounds})")
        print("UART 收发完全正常！问题在传感器或接线那一段。")
    elif ok:
        print(f"部分成功 ({ok}/{args.rounds})，UART 基本可用但可能有时序/接触问题")
    else:
        print(f"全部失败 (0/{args.rounds})")
        print("UART 收发不通。可能原因：")
        print("  1. 短接的不是 TX/RX 这一对引脚")
        print("  2. UART 控制器被占用或未初始化")
        print("  3. 该端口不是 UART 功能（换 ttyS0 试）")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
