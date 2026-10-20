# -*- coding: utf-8 -*-
"""
PS01 手动触发测试 —— 完全复现 PS01Tool 的操作节奏。

思路：
    自动化脚本和 PS01Tool 的唯一差别是"发送时刻由人决定"。
    本脚本打开串口后暂停，让你在任意时刻按 Enter 发命令，
    期间你甚至可以像平时一样移动鼠标、切换窗口。

用法::

    python win_ps01_manual.py --port COM11

流程：
    1. 脚本打开串口，开始读初始化日志
    2. 看到 "Data callback registered" 后提示你
    3. 你按 Enter（想等多久就等多久）
    4. 脚本发送 gnst，然后持续打印收到的所有内容
"""

from __future__ import annotations

import argparse
import sys
import time

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    print("缺少 pyserial。pip install pyserial -i https://pypi.tuna.tsinghua.edu.cn/simple")
    sys.exit(1)


def main() -> int:
    ap = argparse.ArgumentParser(description="PS01 手动触发测试")
    ap.add_argument("--port", default=None)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--cmd", default="gnst")
    ap.add_argument("--no-newline", action="store_true")
    ap.add_argument("--listen", type=float, default=15.0)
    args = ap.parse_args()

    port = args.port
    if not port:
        for p in list_ports.comports():
            if "ch340" in f"{p.device}{p.description}".lower() or "1a86" in (p.hwid or ""):
                port = p.device
                break
    if not port:
        print("找不到 CH340。先 --list 查看。")
        for p in list_ports.comports():
            print(f"  {p.device}  {p.description}")
        return 1

    payload = args.cmd.encode() + (b"" if args.no_newline else b"\r\n")

    print("=" * 70)
    print(f"串口 {port} @ {args.baud}   命令 {payload!r}")
    print("=" * 70)

    ser = serial.Serial(port, args.baud, timeout=0.1,
                        bytesize=serial.EIGHTBITS,
                        parity=serial.PARITY_NONE,
                        stopbits=serial.STOPBITS_ONE)
    try:
        print("\n[1] 正在读初始化日志…（看到 Data callback registered 后暂停）\n")
        t0 = time.time()
        ready = False
        tail = ""
        while time.time() - t0 < 10.0:
            n = ser.in_waiting
            if n:
                text = ser.read(n).decode("utf-8", "ignore")
                buf = tail + text
                for line in buf.splitlines():
                    s = line.strip()
                    if s:
                        print(f"    {s}")
                        if "Data callback registered" in s or "Init success" in s:
                            ready = True
                tail = buf.splitlines()[-1] if buf.splitlines() else ""
                if ready:
                    break
            time.sleep(0.02)

        if ready:
            print("\n    ✅ 初始化完成信号已收到")
        else:
            print("\n    ⚠️ 10 秒内未见完成信号，仍然继续")

        # ---- 关键：让用户决定发送时刻 ----
        print("\n" + "=" * 70)
        print("现在请你像平时操作 PS01Tool 那样：")
        print("  - 想等多久就等多久（甚至可以去点一下 PS01Tool 的界面）")
        print("  - 准备好了就按 Enter，脚本立刻发送 " + repr(payload))
        print("=" * 70)
        try:
            input("\n按 Enter 发送命令… ")
        except (EOFError, KeyboardInterrupt):
            print("\n已取消")
            return 1

        print(f"\n[2] 发送 {payload.hex(' ').upper()}")
        ser.reset_input_buffer()
        ser.write(payload)
        ser.flush()

        print(f"\n[3] 监听 {args.listen} 秒，实时打印收到的内容：\n")
        t1 = time.time()
        total = 0
        buf = ""
        started = False
        while time.time() - t1 < args.listen:
            n = ser.in_waiting
            if n:
                chunk = ser.read(n)
                total += len(chunk)
                buf += chunk.decode("utf-8", "ignore")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    s = line.strip()
                    if s:
                        mark = ""
                        if "measurement started" in s:
                            mark = "  ★★★"
                            started = True
                        print(f"    {s}{mark}")
            time.sleep(0.01)

        print("\n" + "=" * 70)
        print(f"共收到 {total} 字节")
        if started:
            print("✅ 成功！传感器开始测量了。")
        elif total == 0:
            print("❌ 完全没有响应")
        else:
            print("⚠️ 有数据但不是预期的启动响应")
        print("=" * 70)
    finally:
        ser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
