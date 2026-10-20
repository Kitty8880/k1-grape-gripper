#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
波特率扫描：对同一个 UART 依次尝试多个波特率，看哪个能触发传感器响应。

背景：
    ttyS10 (r_uart0) 的 base_baud = 1536000
    1536000 / 115200 = 13.33  ← 非整数，硬件只能取 13，实际输出 118154
    所以标准 115200 在这路 UART 上可能产生 2.5% 误差，传感器认不出。

用法（在 K1 上）::

    sudo python3 ps01_baud_scan.py
    sudo python3 ps01_baud_scan.py --port /dev/ttyS0
"""
import argparse
import sys
import time

try:
    import serial
except ImportError:
    print("缺少 pyserial: sudo apt install -y python3-serial")
    sys.exit(1)


def probe(port: str, baud: int, settle: float) -> tuple[int, int]:
    """用指定波特率打开、发 gnst、读响应。返回 (字节数, 行数)。"""
    try:
        s = serial.Serial(port, baud, timeout=0.2,
                          bytesize=serial.EIGHTBITS,
                          parity=serial.PARITY_NONE,
                          stopbits=serial.STOPBITS_ONE)
    except serial.SerialException as exc:
        print(f"    打不开: {exc}")
        return 0, 0

    total = 0
    lines = 0
    try:
        s.reset_input_buffer()
        time.sleep(0.3)
        s.write(b"gnst\r\n")
        s.flush()
        end = time.time() + settle
        buf = ""
        while time.time() < end:
            d = s.read(4096)
            if not d:
                continue
            total += len(d)
            buf += d.decode("utf-8", "ignore")
            while "\n" in buf:
                ln, buf = buf.split("\n", 1)
                if ln.strip():
                    lines += 1
        # 收尾：把残余当一行
        if buf.strip():
            lines += 1
    finally:
        s.close()
    return total, lines


def main() -> int:
    ap = argparse.ArgumentParser(description="PS01 波特率扫描")
    ap.add_argument("--port", default="/dev/ttyS10")
    ap.add_argument("--settle", type=float, default=2.0,
                    help="每个波特率读多久（秒），默认 2.0")
    args = ap.parse_args()

    # 候选波特率
    candidates = [
        115200,     # 标准值
        118154,     # 1536000 / 13   （ttyS10 的实际输出）
        109714,     # 1536000 / 14
        117647,     # 2000000 / 17
        115158,     # 906875 / 7.875（ttyS0 小数分频）
        113359,     # 906875 / 8
        57600,      # 降速兜底
        9600,
    ]

    print("=" * 66)
    print(f"波特率扫描: {args.port}，每个测 {args.settle}s")
    print("=" * 66)
    print()
    print("提示：传感器每收到一次 gnst 都会开始输出 RAW，")
    print("      如果某个波特率正确，应该能看到明显的数据量。")
    print()

    results = []
    for baud in candidates:
        print(f"  [{baud:>6}] 测试中 ...", end="", flush=True)
        n, ln = probe(args.port, baud, args.settle)
        results.append((baud, n, ln))
        flag = ""
        if n > 100:
            flag = "  <<< 有数据！"
        elif n > 0:
            flag = "  << 少量"
        print(f"\r  [{baud:>6}] {n:>6} 字节 / {ln:>4} 行{flag}")

    print()
    print("=" * 66)
    results.sort(key=lambda x: -x[1])
    print("排序（按收到字节数）:")
    for baud, n, ln in results:
        print(f"  {baud:>7} bps  ->  {n:>6} 字节 / {ln:>4} 行")
    print("=" * 66)

    best = results[0]
    if best[1] > 100:
        print(f"\n最佳: {best[0]} bps（{best[1]} 字节）")
        print(f"下一步: sudo python3 ~/ps01_driver.py --port {args.port} "
              f"--baud {best[0]} --seconds 10")
    else:
        print("\n所有波特率都没有明显数据。")
        print("说明问题不在波特率，而在接线、供电或传感器状态。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
