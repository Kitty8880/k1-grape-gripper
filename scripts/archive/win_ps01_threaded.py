# -*- coding: utf-8 -*-
"""
PS01 多线程复现测试 —— 完全模仿 PS01Tool 的架构。

关键差异假设：
    PS01Tool 是 GUI 程序，有一个后台线程（SerialWorker.run）**从不间断地读取**串口：
        while running:
            if serial.in_waiting:
                data = serial.read(...)
    而我们的脚本是单线程：读一会儿 → 停下来发命令 → 再读。

    如果 CH340 的 RX 缓冲积压会影响 TX，或者传感器要求"被持续读取"
    才接受命令，那这就是根因。

本脚本用双线程复现：
    线程 A：持续读取，把数据丢进队列（永不停）
    主线程：等初始化 → 发命令 → 监控响应

用法::

    python win_ps01_threaded.py --port COM11
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    print("缺少 pyserial。pip install pyserial")
    sys.exit(1)


class Reader(threading.Thread):
    """模仿 PS01Tool 的 SerialWorker：持续读，绝不阻塞主线程。"""

    def __init__(self, ser, out_queue: queue.Queue):
        super().__init__(daemon=True)
        self.ser = ser
        self.q = out_queue
        self.stop_flag = threading.Event()
        self.total = 0
        self.reads = 0

    def run(self) -> None:
        while not self.stop_flag.is_set():
            try:
                n = self.ser.in_waiting
            except OSError:
                n = 0
            if n:
                try:
                    chunk = self.ser.read(n)
                except serial.SerialException as exc:
                    self.q.put(("error", str(exc)))
                    return
                self.total += len(chunk)
                self.reads += 1
                self.q.put(("data", chunk))
            else:
                # 极短 sleep，保持轮询密度接近 GUI 程序
                time.sleep(0.002)

    def stop(self) -> None:
        self.stop_flag.set()


def pick_port() -> str | None:
    for p in list_ports.comports():
        t = f"{p.device} {p.description} {p.hwid}".lower()
        if "ch340" in t or "1a86" in t:
            return p.device
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="PS01 多线程复现（模拟 PS01Tool）")
    ap.add_argument("--port", default=None)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--cmd", default="gnst")
    ap.add_argument("--no-newline", action="store_true")
    ap.add_argument("--wait-ready", type=float, default=8.0)
    ap.add_argument("--listen", type=float, default=10.0)
    args = ap.parse_args()

    port = args.port or pick_port()
    if not port:
        print("找不到 CH340。")
        for p in list_ports.comports():
            print(f"  {p.device}  {p.description}")
        return 1

    payload = args.cmd.encode() + (b"" if args.no_newline else b"\r\n")

    print("=" * 74)
    print("PS01 多线程复现测试（模仿 PS01Tool 的读取线程）")
    print("=" * 74)
    print(f"串口 {port} @ {args.baud}   命令 {payload!r}\n")

    ser = serial.Serial(port, args.baud, timeout=0.05,
                        bytesize=serial.EIGHTBITS,
                        parity=serial.PARITY_NONE,
                        stopbits=serial.STOPBITS_ONE,
                        rtscts=False, dsrdtr=False)
    # 关键：先用 pyserial 默认的 DTR/RTS（高），和 PS01Tool 一致
    try:
        ser.dtr = True
        ser.rts = True
    except (OSError, AttributeError):
        pass

    q: queue.Queue = queue.Queue()
    reader = Reader(ser, q)
    reader.start()
    print("[1] 读取线程已启动（持续轮询，永不停止）")

    try:
        # ---- 等初始化完成 ----
        t0 = time.time()
        ready = False
        line_buf = ""
        lines_shown = 0
        while time.time() - t0 < args.wait_ready:
            try:
                kind, payload_item = q.get(timeout=0.1)
            except queue.Empty:
                continue
            if kind == "error":
                print(f"    读线程错误: {payload_item}")
                break
            text = payload_item.decode("utf-8", "ignore")
            line_buf += text
            while "\n" in line_buf:
                line, line_buf = line_buf.split("\n", 1)
                s = line.strip()
                if s:
                    lines_shown += 1
                    if lines_shown <= 5 or "callback" in s.lower() or "Init success" in s:
                        print(f"    {s}")
                    if ("Data callback registered" in s
                            or "Init success" in s):
                        ready = True
            if ready:
                break

        print(f"\n    累计读 {reader.total} 字节；"
              + ("✅ 初始化完成信号已收到" if ready else "⚠️ 超时"))

        # ---- 发送命令（读取线程仍在运行！）----
        print(f"\n[2] 发送命令 {payload!r}（此时读线程仍在持续读）")
        ser.reset_input_buffer()
        written = ser.write(payload)
        ser.flush()
        print(f"    写入 {written} 字节")

        # ---- 监控响应 ----
        print(f"\n[3] 监听 {args.listen} 秒 …\n")
        before = reader.total
        started = False
        raw_count = 0
        t1 = time.time()
        while time.time() - t1 < args.listen:
            try:
                kind, item = q.get(timeout=0.1)
            except queue.Empty:
                continue
            if kind != "data":
                continue
            text = item.decode("utf-8", "ignore")
            line_buf += text
            while "\n" in line_buf:
                line, line_buf = line_buf.split("\n", 1)
                s = line.strip()
                if not s:
                    continue
                if "measurement started" in s:
                    started = True
                    print(f"    ★★★ {s}")
                elif "RAW[" in s:
                    raw_count += 1
                    if raw_count <= 12:
                        print(f"    {s}")
                elif "[ERROR]" in s:
                    print(f"    {s}")

        got = reader.total - before
        print("\n" + "=" * 74)
        print(f"发送后共收到 {got} 字节，RAW 行 {raw_count} 条")
        if started or raw_count > 0:
            print("✅ 成功！多线程读取是关键 —— PS01Tool 的架构差异就是根因。")
            print("   → 把 ps01_driver.py 改成双线程即可，不用换硬件。")
        else:
            print("❌ 仍然失败。")
            print("   → 多线程不是根因，问题在 CH340 驱动/硬件层。")
        print("=" * 74)
    finally:
        reader.stop()
        reader.join(timeout=1.0)
        ser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
