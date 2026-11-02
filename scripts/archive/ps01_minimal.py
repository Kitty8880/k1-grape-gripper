#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最小复现脚本：完全模仿 PS01Tool 的串口行为。

差异点（与 ps01_driver.py 相比）：
    1. 只传 port / baudrate / timeout，不传 bytesize/parity/... （pyserial 默认就是 8N1）
    2. 不设置 DTR / RTS（不产生电平跳变）
    3. 不调用 reset_input_buffer
    4. 打开后持续读，读到 'Data callback registered' 就发 gnst
    5. 发送后持续读 10 秒

用法（在 K1 上）::

    sudo python3 ps01_minimal.py
    sudo python3 ps01_minimal.py --port /dev/ttyUSB0 --gap 0
"""
import argparse
import sys
import time

try:
    import serial
except ImportError:
    print("缺少 pyserial: sudo apt install -y python3-serial")
    sys.exit(1)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--gap", type=float, default=0.0,
                    help="看到初始化完成后等多久再发（默认 0，立刻发）")
    ap.add_argument("--after", type=float, default=10.0)
    args = ap.parse_args()

    print("=" * 64)
    print("最小复现（模仿 PS01Tool）")
    print(f"  端口: {args.port} @ {args.baud}")
    print("=" * 64)

    # ★ 关键：只传三个参数，和 PS01Tool 完全一致
    ser = serial.Serial(args.port, args.baud, timeout=0.05)
    print(f"已打开: {ser.name}")
    print(f"  settings: {ser.get_settings()}")
    print()

    buf = ""
    ready = False
    rx_bytes = 0
    t0 = time.time()

    print("--- 阶段 1：读初始化（最多 8 秒）---")
    while time.time() - t0 < 8.0:
        d = ser.read(ser.in_waiting or 1)
        if not d:
            continue
        rx_bytes += len(d)
        text = d.decode("utf-8", "ignore")
        buf += text
        # 逐行打印
        while "\n" in buf:
            ln, buf = buf.split("\n", 1)
            ln = ln.strip()
            if ln:
                print(f"  {ln}")
                if "Data callback registered" in ln:
                    ready = True
        if ready:
            break

    print()
    print(f"初始化: {rx_bytes} 字节, ready={ready}")

    if args.gap:
        print(f"等 {args.gap}s ...")
        time.sleep(args.gap)

    print()
    print("--- 发送 gnst\\r\\n ---")
    n = ser.write(b"gnst\r\n")
    ser.flush()
    print(f"  write() 返回 {n}")

    print()
    print(f"--- 阶段 2：读响应（{args.after}s）---")
    rx2 = 0
    t1 = time.time()
    buf = ""
    while time.time() - t1 < args.after:
        d = ser.read(ser.in_waiting or 1)
        if not d:
            continue
        rx2 += len(d)
        text = d.decode("utf-8", "ignore")
        buf += text
        while "\n" in buf:
            ln, buf = buf.split("\n", 1)
            ln = ln.strip()
            if ln:
                print(f"  {ln}")

    ser.close()
    print()
    print("=" * 64)
    print(f"初始化 {rx_bytes} 字节 / 响应 {rx2} 字节")
    if rx2 > 0:
        print("✅ 有响应！对比 ps01_driver.py 的差异就是原因")
    else:
        print("❌ 仍然 0 字节")
        print("   说明不是 DTR/RTS 或参数设置的问题")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
