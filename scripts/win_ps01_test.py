# -*- coding: utf-8 -*-
"""
Windows 侧 PS01 发送参数扫描 —— 找出让 gnst 生效的确切条件。

背景：
    PS01Tool 在 Windows 上点"开始采集"能触发传感器（已验证），
    但用 pyserial 直接发同样的 "gnst" 却没反应。
    唯一可疑差异是"发送时机"和"重试策略"。

本脚本会横扫这些变量：
    - 初始化完成后延迟 pre-delay 秒再发
    - 发不成功就重试 N 次
    - DTR/RTS 电平组合

用法::

    python win_ps01_test.py --port COM11 --scan
    python win_ps01_test.py --port COM11 --pre-delay 5
    python win_ps01_test.py --port COM11 --pre-delay 3 --retries 3
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


def pick_ch340() -> str | None:
    for p in list_ports.comports():
        t = f"{p.device} {p.description} {p.hwid}".lower()
        if "ch340" in t or "1a86" in t:
            return p.device
    return None


def open_port(port: str, baud: int, dtr: bool, rts: bool):
    ser = serial.Serial(port, baud, timeout=0.1,
                        bytesize=serial.EIGHTBITS,
                        parity=serial.PARITY_NONE,
                        stopbits=serial.STOPBITS_ONE,
                        rtscts=False, dsrdtr=False)
    ser.dtr = dtr
    ser.rts = rts
    return ser


def wait_init(ser, timeout: float) -> tuple[bool, int]:
    """读到 'Data callback registered' 就返回。返回 (是否成功, 读到的字节数)。"""
    t0 = time.time()
    total = 0
    seen = False
    tail = ""
    while time.time() - t0 < timeout:
        try:
            n = ser.in_waiting
        except OSError:
            n = 0
        if n:
            chunk = ser.read(n)
            total += len(chunk)
            text = tail + chunk.decode("utf-8", "ignore")
            for line in text.splitlines():
                if ("Data callback registered" in line
                        or "Init success" in line):
                    seen = True
            tail = text.splitlines()[-1] if text.splitlines() else ""
            if seen:
                break
        time.sleep(0.01)
    return seen, total


def try_trigger(ser, payload: bytes, listen: float,
                retries: int, retry_gap: float) -> tuple[bool, bytes]:
    """发送命令并监听。返回 (是否触发, 收到的一切数据)。"""
    buf = bytearray()
    for attempt in range(1, retries + 1):
        # 清掉缓冲区残留，避免误判
        try:
            ser.reset_input_buffer()
        except OSError:
            pass
        ser.write(payload)
        ser.flush()

        t0 = time.time()
        got = bytearray()
        while time.time() - t0 < listen:
            try:
                n = ser.in_waiting
            except OSError:
                n = 0
            if n:
                got.extend(ser.read(n))
            time.sleep(0.01)
        buf.extend(got)

        text = bytes(got).decode("utf-8", "ignore")
        if "measurement started" in text or "RAW[" in text:
            return True, bytes(buf)
        if attempt < retries:
            time.sleep(retry_gap)
    return False, bytes(buf)


def one_shot(port: str, baud: int, payload: bytes, pre_delay: float,
             listen: float, retries: int, retry_gap: float,
             dtr: bool, rts: bool) -> tuple[bool, str]:
    """完整跑一次：开串口 → 等初始化 → 延迟 → 发命令 → 判断。"""
    try:
        ser = open_port(port, baud, dtr, rts)
    except serial.SerialException as exc:
        return False, f"打开失败: {exc}"
    try:
        seen, total = wait_init(ser, 6.0)
        if not seen:
            return False, f"初始化未完成（读到 {total} 字节）"
        if pre_delay > 0:
            time.sleep(pre_delay)
        ok, data = try_trigger(ser, payload, listen, retries, retry_gap)
        text = data.decode("utf-8", "ignore")
        if ok:
            return True, f"触发成功（{len(data)} 字节）"
        return False, f"无响应（{len(data)} 字节: {text[:80]!r}）"
    finally:
        ser.close()


def scan(port: str, baud: int, payload: bytes) -> int:
    """横扫 pre-delay 和 DTR/RTS 组合。"""
    print("=" * 76)
    print("PS01 发送参数扫描")
    print("=" * 76)
    print(f"串口 {port} @ {baud}   命令 {payload!r}\n")

    delays = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
    dtr_rts = [(True, True), (False, False), (True, False), (False, True)]

    winner = None
    total = len(delays) * len(dtr_rts)
    n = 0
    for dtr, rts in dtr_rts:
        print(f"\n--- DTR={int(dtr)} RTS={int(rts)} ---")
        for d in delays:
            n += 1
            print(f"  [{n}/{total}] pre-delay={d:>4.1f}s … ", end="", flush=True)
            ok, msg = one_shot(port, baud, payload, d, 3.0, 2, 1.0, dtr, rts)
            print(("✅ " + msg) if ok else ("❌ " + msg))
            if ok and winner is None:
                winner = (dtr, rts, d)
                print(f"\n  ★★★ 找到可用组合：DTR={int(dtr)} RTS={int(rts)} "
                      f"pre-delay={d}s ★★★\n")
                return 0
            time.sleep(0.5)

    print("\n" + "=" * 76)
    if winner is None:
        print("所有组合都没触发。说明不是时序或 DTR/RTS 的问题。")
        print("下一步：必须抓到 PS01Tool 真正的 TX 字节（区分方向）。")
    print("=" * 76)
    return 0 if winner else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Windows 侧 PS01 发送参数测试")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--port", default=None)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--cmd", default="gnst", help="命令文本，默认 gnst")
    ap.add_argument("--no-newline", action="store_true", help="不加换行")
    ap.add_argument("--pre-delay", type=float, default=3.0,
                    help="初始化完成后等多久再发命令，默认 3s")
    ap.add_argument("--listen", type=float, default=4.0)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--retry-gap", type=float, default=1.0)
    ap.add_argument("--dtr", type=int, choices=[0, 1], default=1)
    ap.add_argument("--rts", type=int, choices=[0, 1], default=1)
    ap.add_argument("--scan", action="store_true",
                    help="横扫 pre-delay 和 DTR/RTS 组合（推荐先跑这个）")
    args = ap.parse_args()

    if args.list:
        for p in list_ports.comports():
            print(f"  {p.device:<8} {p.description}")
        return 0

    port = args.port or pick_ch340()
    if not port:
        print("没找到 CH340，用 --list 查看后 --port COMxx")
        return 1

    payload = args.cmd.encode()
    if not args.no_newline:
        payload += b"\r\n"

    if args.scan:
        return scan(port, args.baud, payload)

    print("=" * 76)
    print(f"串口 {port} @ {args.baud}  命令 {payload!r}")
    print(f"pre-delay={args.pre_delay}s  重试={args.retries}  "
          f"DTR={args.dtr} RTS={args.rts}")
    print("=" * 76)
    ok, msg = one_shot(port, args.baud, payload, args.pre_delay, args.listen,
                       args.retries, args.retry_gap,
                       bool(args.dtr), bool(args.rts))
    print(("✅ " + msg) if ok else ("❌ " + msg))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
