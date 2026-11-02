#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PS01 命令重发测试：验证是否需要连发两次 gnst。

依据：DMS 抓包显示 PS01Tool 在一次"开始采集"中发送过 2 次 gnst，
      间隔约 0.2 秒（中间夹约 2.4KB 的 RX 数据）。

用法（在 K1 上）::

    sudo python3 ps01_repeat.py
    sudo python3 ps01_repeat.py --port /dev/ttyUSB0 --gap 0.2
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


def drain(ser, seconds: float, label: str, max_lines: int = 40) -> tuple[int, list[str]]:
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
                    print(f"      {ln}")
    if buf.strip():
        lines.append(buf.strip())
    print(f"      [{label}] {total} 字节 / {len(lines)} 行")
    return total, lines


def main() -> int:
    ap = argparse.ArgumentParser(description="PS01 命令重发测试")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--gap", type=float, default=0.2, help="两次 gnst 间隔")
    ap.add_argument("--init", type=float, default=6.0, help="等初始化秒数")
    ap.add_argument("--after", type=float, default=8.0, help="发完后读秒数")
    args = ap.parse_args()

    print("=" * 70)
    print(f"PS01 命令重发测试  {args.port} @ {args.baud}")
    print("=" * 70)

    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.2,
                            bytesize=serial.EIGHTBITS,
                            parity=serial.PARITY_NONE,
                            stopbits=serial.STOPBITS_ONE)
    except serial.SerialException as exc:
        print(f"打不开: {exc}")
        return 2

    try:
        # ---------- 阶段 1：等初始化 ----------
        print(f"\n[1] 读初始化（最多 {args.init}s）")
        t0 = time.time()
        buf = ""
        ready = False
        init_bytes = 0
        while time.time() - t0 < args.init:
            d = ser.read(4096)
            if d:
                init_bytes += len(d)
                buf += d.decode("utf-8", "ignore")
                if "Data callback registered" in buf:
                    ready = True
                    print("      看到 'Data callback registered'，初始化完成")
                    break
        print(f"      初始化共 {init_bytes} 字节，ready={ready}")

        # ---------- 阶段 2：连发两次 gnst ----------
        print(f"\n[2] 连发两次 gnst\\r\\n（间隔 {args.gap}s）")
        ser.write(b"gnst\r\n")
        ser.flush()
        print("      第 1 次已发送")
        time.sleep(args.gap)
        ser.write(b"gnst\r\n")
        ser.flush()
        print("      第 2 次已发送")

        # ---------- 阶段 3：读响应 ----------
        print(f"\n[3] 读响应（{args.after}s）")
        n, lines = drain(ser, args.after, "响应")

        joined = "\n".join(lines)
        has_started = "measurement started" in joined
        has_raw = "RAW[" in joined

        print()
        print("=" * 70)
        if has_raw:
            print(f"🎉 成功！收到 RAW 数据（{n} 字节）")
            print("   结论：需要连发两次 gnst")
            print(f"   下一步: sudo python3 ~/ps01_driver.py --port {args.port} --seconds 10")
        elif has_started:
            print("✅ 收到 'measurement started'，但没看到 RAW")
            print("   传感器已进入测量模式，可能数据在下一轮输出")
        elif n > 0:
            print(f"⚠️ 收到 {n} 字节但不是预期格式：")
            for ln in lines[:10]:
                print(f"     {ln}")
        else:
            print("❌ 连发两次也没有响应")
            print("   说明问题不在重发次数，需要继续排查")
        print("=" * 70)
    finally:
        ser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
