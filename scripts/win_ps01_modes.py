# -*- coding: utf-8 -*-
"""
PS01 发送模式横扫 —— 试各种"发几次、间隔多少"的组合。

依据：
    Device Monitoring Studio 日志里 "gnst" 出现了两次，中间夹着约 2.4KB 的 RX 数据
    （按 115200 算约 0.2 秒）。说明 PS01Tool 可能在极短间隔内连发了两次。

模式候选：
    1. 单次
    2. 两次，间隔 0.1s
    3. 两次，间隔 0.2s
    4. 三次，间隔 0.1s
    5. 先 gned 再 gnst
    6. 五次，间隔 0.5s
    7. 单次，但在发送前清空 RX 缓冲
    8. 单次，发完立刻读（不 sleep）

用法::

    python win_ps01_modes.py --port COM11
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


def wait_ready(ser, timeout: float = 8.0) -> bool:
    """读到 Data callback registered 就返回 True。"""
    t0 = time.time()
    tail = ""
    while time.time() - t0 < timeout:
        try:
            n = ser.in_waiting
        except OSError:
            n = 0
        if n:
            text = tail + ser.read(n).decode("utf-8", "ignore")
            for line in text.splitlines():
                if ("Data callback registered" in line
                        or "Init success" in line):
                    return True
            tail = text.splitlines()[-1] if text.splitlines() else ""
        time.sleep(0.01)
    return False


def listen(ser, seconds: float) -> bytes:
    buf = bytearray()
    t0 = time.time()
    while time.time() - t0 < seconds:
        try:
            n = ser.in_waiting
        except OSError:
            n = 0
        if n:
            buf.extend(ser.read(n))
        time.sleep(0.01)
    return bytes(buf)


def run_mode(port: str, baud: int, name: str, sender) -> tuple[bool, str]:
    """跑一种发送模式。sender(ser) 是一个函数，负责实际发送。"""
    try:
        ser = serial.Serial(port, baud, timeout=0.1,
                            bytesize=serial.EIGHTBITS,
                            parity=serial.PARITY_NONE,
                            stopbits=serial.STOPBITS_ONE)
    except serial.SerialException as exc:
        return False, f"打开失败: {exc}"

    try:
        if not wait_ready(ser):
            return False, "初始化未完成"
        time.sleep(0.3)
        sender(ser)
        data = listen(ser, 6.0)
        text = data.decode("utf-8", "ignore")
        if "measurement started" in text or "RAW[" in text:
            return True, f"✅ 触发（{len(data)} 字节）"
        return False, f"无响应（{len(data)} 字节）"
    finally:
        ser.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="PS01 发送模式横扫")
    ap.add_argument("--port", default=None)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--cmd", default="gnst")
    args = ap.parse_args()

    port = args.port
    if not port:
        for p in list_ports.comports():
            if "ch340" in f"{p.device}{p.description}".lower() or "1a86" in (p.hwid or ""):
                port = p.device
                break
    if not port:
        print("找不到 CH340")
        return 1

    CMD = args.cmd.encode() + b"\r\n"
    GNED = b"gned\r\n"

    # 每种模式：(名字, 发送函数)
    modes = [
        ("单次", lambda s: s.write(CMD)),
        ("两次/0.1s", lambda s: (s.write(CMD), time.sleep(0.1), s.write(CMD))),
        ("两次/0.2s", lambda s: (s.write(CMD), time.sleep(0.2), s.write(CMD))),
        ("三次/0.1s", lambda s: [s.write(CMD) or time.sleep(0.1) for _ in range(3)]),
        ("先 gned 再 gnst", lambda s: (s.write(GNED), time.sleep(0.3), s.write(CMD))),
        ("五次/0.5s", lambda s: [s.write(CMD) or time.sleep(0.5) for _ in range(5)]),
        ("清缓冲后单次", lambda s: (s.reset_input_buffer(), s.write(CMD))),
        ("两次/1.0s", lambda s: (s.write(CMD), time.sleep(1.0), s.write(CMD))),
    ]

    print("=" * 72)
    print(f"PS01 发送模式横扫  串口 {port} @ {args.baud}")
    print("=" * 72)

    winner = None
    for i, (name, sender) in enumerate(modes, 1):
        print(f"\n[{i}/{len(modes)}] 模式：{name}")
        # sender 里的 lambda 若返回 list 会有副作用问题，这里包一层
        def wrapped(ser, _sender=sender):
            _sender(ser)
            try:
                ser.flush()
            except Exception:  # noqa: BLE001
                pass
        ok, msg = run_mode(port, args.baud, name, wrapped)
        print(f"    {msg}")
        if ok:
            winner = name
            print(f"\n★★★ 找到可行模式：{name} ★★★")
            break
        time.sleep(0.5)

    print("\n" + "=" * 72)
    if winner:
        print(f"✅ 有效模式：{winner}")
    else:
        print("所有发送模式都失败。")
        print("说明 PS01Tool 与 pyserial 之间还有别的差异（控制线/驱动层）。")
        print("务实建议：改走 K1 的 40Pin 硬件 UART，绕开 CH340。")
    print("=" * 72)
    return 0 if winner else 1


if __name__ == "__main__":
    sys.exit(main())
