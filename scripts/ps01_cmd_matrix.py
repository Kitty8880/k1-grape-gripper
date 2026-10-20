#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
命令变体全扫描：对 /dev/ttyUSB0 依次尝试各种"发送方式"，看哪种能触发传感器。

依据：
    DMS 抓包的长度字段是 4，但字节是 6（gnst\r\n）——
    所以不确定 PS01Tool 到底发的是 4 字节还是 6 字节。
    本脚本把常见可能性都试一遍。

用法（在 K1 上）::

    sudo python3 ps01_cmd_matrix.py
    sudo python3 ps01_cmd_matrix.py --port /dev/ttyUSB0
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


# (名称, 字节序列, 是否连发两次)
CASES = [
    ("gnst\\r\\n",              b"gnst\r\n",  False),
    ("gnst",                    b"gnst",      False),
    ("gnst\\r",                 b"gnst\r",    False),
    ("gnst\\n",                 b"gnst\n",    False),
    ("GNST\\r\\n",              b"GNST\r\n",  False),
    ("gnst\\r\\n x2",           b"gnst\r\n",  True),
    ("gned\\r\\n 然后 gnst\\r\\n", None,        False),
]


def read_for(ser, seconds: float) -> tuple[int, str]:
    total = 0
    buf = ""
    t0 = time.time()
    while time.time() - t0 < seconds:
        d = ser.read(4096)
        if d:
            total += len(d)
            buf += d.decode("utf-8", "ignore")
    return total, buf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--after", type=float, default=4.0,
                    help="每种命令发完后读多久")
    args = ap.parse_args()

    print("=" * 70)
    print(f"命令变体扫描  {args.port} @ {args.baud}")
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
        # 每轮开始前，确保传感器处于已知状态
        print("\n准备：发 gned\\r\\n 停止采集，然后清缓冲")
        ser.write(b"gned\r\n")
        ser.flush()
        time.sleep(1.0)
        ser.reset_input_buffer()

        print("\n开始逐项测试：\n")
        for name, payload, twice in CASES:
            print(f"  [{name}]", end="", flush=True)

            # 对特殊项：先 gned 再 gnst
            if payload is None:
                ser.write(b"gned\r\n")
                ser.flush()
                time.sleep(0.5)
                ser.reset_input_buffer()
                ser.write(b"gnst\r\n")
                ser.flush()
            else:
                ser.write(payload)
                ser.flush()
                if twice:
                    time.sleep(0.2)
                    ser.write(payload)
                    ser.flush()

            n, text = read_for(ser, args.after)
            has_raw = "RAW[" in text
            has_started = "measurement started" in text
            mark = ""
            if has_raw:
                mark = "  ✅✅ 收到 RAW！"
            elif has_started:
                mark = "  ✅ 收到 measurement started"
            elif n > 0:
                mark = f"  ({n} 字节但不是预期格式)"
            print(f"\r  [{name:<22}] {n:>6} 字节{mark}")

            if has_raw:
                print()
                print("=" * 70)
                print(f"🎉 找到可用命令：{name}")
                print(f"   发送字节: {payload!r}" if payload else
                      "   序列: gned\\r\\n 然后 gnst\\r\\n")
                print("=" * 70)
                # 打印前几行 RAW
                for ln in text.splitlines():
                    if "RAW[" in ln:
                        print(f"   {ln}")
                        break
                return 0

            # 每项之间清理
            ser.write(b"gned\r\n")
            ser.flush()
            time.sleep(0.5)
            ser.reset_input_buffer()
            time.sleep(0.3)
    finally:
        ser.close()

    print()
    print("=" * 70)
    print("所有命令变体都没有触发传感器")
    print("=" * 70)
    print("说明：问题不在命令字节，而在更底层（驱动/硬件）。")
    print("建议：换 CP2102 或 FT232 模块。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
