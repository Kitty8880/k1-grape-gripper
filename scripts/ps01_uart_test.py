#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PS01 在 K1 r_uart0 (/dev/ttyS10) 上的完整链路测试。

自动完成：
    1. 停 getty、开权限、设 115200 8N1
    2. 监听 3 秒（看传感器的初始化日志）
    3. 发送启动命令 gnst
    4. 监听 6 秒（看 RAW 数据）
    5. 汇总判断

用法（在 K1 上）::

    sudo python3 ps01_uart_test.py
    sudo python3 ps01_uart_test.py --port /dev/ttyS10
    sudo python3 ps01_uart_test.py --cmd-raw      # gnst 不带换行
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
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
    print("缺少 pyserial： sudo apt install -y python3-serial")
    sys.exit(1)


def setup_port(dev: str) -> None:
    """停 getty、开权限、用 stty 设置参数。"""
    name = os.path.basename(dev)
    print(f"[准备] 停止 {name} 上的 getty（如果有）")
    subprocess.run(["systemctl", "stop", f"serial-getty@{name}"],
                   capture_output=True, text=True)

    print(f"[准备] 放开 {dev} 权限")
    subprocess.run(["chmod", "a+rw", dev], capture_output=True, text=True)

    print(f"[准备] stty 设置 115200 raw -crtscts")
    r = subprocess.run(["stty", "-F", dev, "115200", "raw", "-echo", "-crtscts"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"        stty 失败: {r.stderr.strip()}")

    # 打印实际参数
    r = subprocess.run(["stty", "-F", dev, "-a"], capture_output=True, text=True)
    if r.returncode == 0:
        line = r.stdout.strip().splitlines()[0] if r.stdout.strip() else ""
        print(f"        当前参数: {line[:100]}")


def listen(ser, seconds: float, label: str,
           show_lines: int = 25) -> tuple[int, list]:
    """监听并返回 (字节数, 解析出的行列表)。"""
    buf = bytearray()
    t0 = time.time()
    shown = 0
    lines = []
    while time.time() - t0 < seconds:
        try:
            n = ser.in_waiting
        except OSError:
            n = 0
        if n:
            chunk = ser.read(n)
            buf.extend(chunk)
            text = chunk.decode("utf-8", "ignore")
            for line in text.splitlines():
                s = line.strip()
                if s:
                    lines.append(s)
                    if shown < show_lines:
                        shown += 1
                        print(f"    {s}")
        else:
            time.sleep(0.01)
    print(f"    [{label}] 共 {len(buf)} 字节 / {len(lines)} 行")
    return len(buf), lines


def main() -> int:
    ap = argparse.ArgumentParser(description="PS01 UART 链路测试")
    ap.add_argument("--port", default="/dev/ttyS10")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--cmd-raw", action="store_true",
                    help="启动命令用 gnst（不带换行）")
    ap.add_argument("--listen-before", type=float, default=3.0)
    ap.add_argument("--listen-after", type=float, default=6.0)
    args = ap.parse_args()

    dev = args.port
    payload = b"gnst" if args.cmd_raw else b"gnst\r\n"

    print("=" * 70)
    print(f"PS01 UART 链路测试   {dev} @ {args.baud} 8N1")
    print("=" * 70)
    print()

    if not os.path.exists(dev):
        print(f"❌ {dev} 不存在")
        return 2

    setup_port(dev)
    print()

    try:
        ser = serial.Serial(dev, args.baud, timeout=0.1,
                            bytesize=serial.EIGHTBITS,
                            parity=serial.PARITY_NONE,
                            stopbits=serial.STOPBITS_ONE,
                            rtscts=False, dsrdtr=False)
    except serial.SerialException as exc:
        print(f"❌ 打不开 {dev}: {exc}")
        return 2

    try:
        ser.reset_input_buffer()
        print(f"[1] 监听 {args.listen_before}s（传感器上电后应输出初始化日志）")
        n1, lines1 = listen(ser, args.listen_before, "阶段1")

        if n1 == 0:
            print("    ⚠️ 没有任何数据")
            print("       可能：传感器没供电 / 接线错 / 引脚位置错 / 初始化日志已错过")
        print()

        print(f"[2] 发送启动命令 {payload!r}")
        ser.write(payload)
        ser.flush()
        print(f"    已写入 {len(payload)} 字节")
        print()

        print(f"[3] 监听 {args.listen_after}s（应看到 measurement started 和 RAW 数据）")
        n2, lines2 = listen(ser, args.listen_after, "阶段2")
        print()

        # ---- 汇总 ----
        all_lines = lines1 + lines2
        joined = "\n".join(all_lines)
        has_raw = bool(re.search(r"RAW\[\d+\]", joined))
        has_started = "measurement started" in joined
        has_init = "JSON module initialized" in joined or "AS7150" in joined
        has_json = '{"status"' in joined

        print("=" * 70)
        print("结果汇总")
        print("=" * 70)
        print(f"  初始化日志:      {'✅' if has_init else '❌'}")
        print(f"  JSON 状态消息:   {'✅' if has_json else '❌'}")
        print(f"  measurement started: {'✅' if has_started else '❌'}")
        print(f"  RAW 测量数据:    {'✅' if has_raw else '❌'}")
        print(f"  总字节数:        {n1 + n2}")
        print()

        if has_raw:
            print("🎉 完全成功！传感器数据已到达 K1。")
            print(f"   下一步：python3 ps01_driver.py --port {dev} --seconds 10")
        elif has_init:
            print("✅ 接收方向（传感器→K1）通了！但没触发测量。")
            print("   说明 TX 方向（K1→传感器）还没通，或命令格式不对。")
            print("   试试： sudo python3 ps01_uart_test.py --cmd-raw")
        elif n1 + n2 == 0:
            print("❌ 完全没有数据。排查顺序：")
            print("   1. 传感器 3V3(脚1) / GND(脚6) 有没有接对、有没有 3.3V")
            print("   2. 传感器 TXD 是否接到 K1 脚 10（不是脚 8！）")
            print("   3. 引脚位置是否数错（用万用表确认脚 1 和脚 6 之间 3.3V）")
            print("   4. GPIO47/48 的 pinmux 未生效 → 需要改设备树")
        else:
            print("⚠️ 收到了一些数据但不是预期格式：")
            for ln in all_lines[:10]:
                print(f"     {ln}")
        print("=" * 70)
    finally:
        ser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
