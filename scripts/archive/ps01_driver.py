#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PS01 光学力传感器驱动（K1 MusePi Pro）

硬件背景
--------
PS01 是 ams OSRAM **AS7150 光学力传感 AFE** 为核心的压力/触觉传感器：
    LED 发光 → 弹性体/光路 → 光电二极管(PD) → AS7150 处理 → 串口 JSON/RAW 输出

接口：CH340 (USB-TTL) → /dev/ttyUSB0，115200 8N1

数据协议（从 PS01Tool.exe 逆向 + 实测抓包确认）
--------------------------------------------
传感器输出「行」文本，每行是下面两种之一：

    RAW[3]: 0xAACE48 (status=0x0A)          ← 测量数据
    {"status": "sensor_state", "value": "NORMAL"}   ← 状态消息

RAW 行的 index 是**循环计数**，三个通道轮流出：

    RAW[0] → PD3 (sub=1)
    RAW[1] → PD2 (sub=2)
    RAW[2] → PD1 (sub=3)
    RAW[3] → PD3
    ...

对应关系来自初始化日志：
    [INFO] AS7150 PD assigned: SUB1(PD3->MOD1), SUB2(PD2->MOD1), SUB3(PD1->MOD2)

启动采集（已逆向确认）
------------------------
传感器上电后只输出初始化日志，必须发送启动命令才开始持续上报数据。

    启动命令： gnst    （ASCII 四字节；固件回显 "~measurement started~"）
    停止命令： gned    （固件回显 "~measurement ended~"）

这两个是厂商自定义命令，不是标准 AT 指令，所以试 ON / START / JSON 都没用。
证据来源：
  1. 对 PS01Tool.exe 的逆向 —— 界面调试标记 "采集已开启 (gnst)" / "采集已暂停 (gned)"
  2. Device Monitoring Studio 抓到的实际 TX 字节："gnst"

用法::

    from ps01_driver import Ps01Sensor

    sensor = Ps01Sensor("/dev/ttyUSB0")
    sensor.open()
    sensor.start()                      # 发送 gnst
    while True:
        reading = sensor.read()         # 返回 Ps01Reading 或 None
        if reading:
            print(reading)
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional, List, Callable

try:
    import serial
except ImportError:  # pragma: no cover
    serial = None  # type: ignore


#: 启动采样命令。固件回显 "~measurement started~"。
#:
#: 关键：前面的 0xD4（212）是"唤醒字节"，必须保留！
#: 这个字节来自 PS01Tool.py 的反汇编（send_text_command）：
#:
#:     self.serial.write(bytes([212]))
#:     self.serial.write((cmd + "\r\n").encode())
#:
#: 少了 0xD4，传感器会把整个包当作无效命令丢弃 —— 这就是之前
#: 一直收不到数据的原因。DMS 抓包没记录这个字节（被当作控制字符过滤），
#: 导致长期误判。
DEFAULT_START_COMMAND = b"\xd4gnst\r\n"

#: 停止采样命令。固件回显 "~measurement ended~"。
DEFAULT_STOP_COMMAND = b"\xd4gned\r\n"


# RAW 数据行正则（与 PS01Tool 程序里使用的完全相同）
RE_RAW = re.compile(
    r"RAW\[(\d+)\]:\s*0x([0-9A-Fa-f]+)\s*\(status=0x([0-9A-Fa-f]+)\)"
)
RE_JSON = re.compile(r"^\s*\{")


@dataclass
class Ps01Reading:
    """一次完整的 3 通道采样。"""
    pd1: int = 0
    pd2: int = 0
    pd3: int = 0
    status1: int = 0
    status2: int = 0
    status3: int = 0
    timestamp: float = 0.0

    #: 三个通道是否都收到过（第一轮之前可能不全）
    complete: bool = False

    def to_dict(self) -> dict:
        return {
            "pd1": self.pd1, "pd2": self.pd2, "pd3": self.pd3,
            "status1": self.status1, "status2": self.status2,
            "status3": self.status3,
            "complete": self.complete,
            "timestamp": self.timestamp,
        }

    def __str__(self) -> str:
        return (f"PD1={self.pd1:7d} PD2={self.pd2:7d} PD3={self.pd3:7d} "
                f"(st: 0x{self.status1:02X} 0x{self.status2:02X} 0x{self.status3:02X})"
                + ("" if self.complete else " [不完整]"))


class Ps01Error(Exception):
    """PS01 驱动错误。"""


class Ps01Sensor:
    """
    PS01 传感器串口驱动。

    设计要点
    --------
    1. **非阻塞**：``read()`` 每次最多读一小批，返回最新的完整采样或 None。
       适合放进 100Hz 控制循环里。
    2. **按 index % 3 分通道**：不依赖 RAW 编号的连续性（固件每轮重置编号）。
    3. **容错**：串口断开/乱码时抛 Ps01Error 或返回 None，不会把脏数据
       塞进力闭环。
    """

    #: RAW 索引 → 物理通道名。0/1/2 循环。
    CHANNEL_ORDER = ("PD3", "PD2", "PD1")

    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 115200,
                 timeout: float = 0.05,
                 start_command: Optional[bytes] = None,
                 stop_command: Optional[bytes] = None,
                 dtr: bool = True, rts: bool = True,
                 log: Optional[Callable[[str], None]] = None):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        # 默认使用已确认的 gnst / gned；传 None 则用默认值
        self.start_command = (DEFAULT_START_COMMAND if start_command is None
                              else start_command)
        self.stop_command = (DEFAULT_STOP_COMMAND if stop_command is None
                             else stop_command)
        # DTR/RTS：pyserial 默认都拉高，但 CH340 的这两根常接目标板
        # 的 EN/BOOT 脚，拉高会让传感器停在复位或待机状态。
        # 默认设成低电平（更接近"空闲"），可用 --dtr/--rts 覆盖。
        self.dtr = dtr
        self.rts = rts
        self._log = log or (lambda _m: None)

        self._ser = None
        self._buf = ""
        self._current = Ps01Reading()
        self._seen = set()          # 本轮已收到的通道
        self._last_complete: Optional[Ps01Reading] = None
        self._sample_count = 0
        self._parse_errors = 0
        self._raw_lines = 0
        self._status_messages: List[dict] = []

    # ------------------------------------------------------------------
    # 连接
    # ------------------------------------------------------------------

    def open(self) -> "Ps01Sensor":
        if serial is None:
            raise Ps01Error(
                "缺少 pyserial。安装：\n"
                "  sudo apt install -y python3-serial"
            )
        try:
            self._ser = serial.Serial(
                self.port, self.baudrate, timeout=self.timeout,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                rtscts=False,
                dsrdtr=False,
            )
        except serial.SerialException as exc:
            raise Ps01Error(
                f"打不开 {self.port}: {exc}\n"
                "  1. 串口权限： sudo chmod a+rw /dev/ttyUSB0\n"
                "  2. 设备存在： ls -l /dev/ttyUSB*\n"
                "  3. 被占用：先关掉其他读串口的程序"
            ) from exc

        # 关键：CH340 模块的 DTR/RTS 常接到目标板的 EN/BOOT，
        # 打开串口时 pyserial 默认把两者都拉高，可能让传感器处于
        # 复位或等待状态。这里按配置显式设置。
        try:
            self._ser.dtr = self.dtr
            self._ser.rts = self.rts
        except (OSError, AttributeError) as exc:
            self._log(f"设置 DTR/RTS 失败（可忽略）：{exc}")

        self._log(f"已打开 {self.port} @ {self.baudrate} "
                  f"(DTR={self.dtr} RTS={self.rts})")
        try:
            self._ser.reset_input_buffer()
        except OSError:
            pass
        return self

    def close(self) -> None:
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:  # noqa: BLE001
                pass
            self._ser = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    @property
    def is_open(self) -> bool:
        return self._ser is not None and self._ser.is_open

    # ------------------------------------------------------------------
    # 控制
    # ------------------------------------------------------------------

    def send(self, payload: bytes) -> None:
        """发送原始字节。"""
        if not self.is_open:
            raise Ps01Error("串口未打开")
        try:
            self._ser.write(payload)
            self._ser.flush()
        except serial.SerialException as exc:
            raise Ps01Error(f"写入失败：{exc}") from exc

    def start(self) -> None:
        """发送启动采集命令 gnst。"""
        if not self.start_command:
            raise Ps01Error("启动命令为空")
        self.send(self.start_command)
        self._log(f"已发送启动命令 {self.start_command!r}")

    def stop(self) -> None:
        """发送停止采集命令 gned。"""
        if not self.stop_command:
            return
        self.send(self.stop_command)
        self._log(f"已发送停止命令 {self.stop_command!r}")

    # ------------------------------------------------------------------
    # 读数据
    # ------------------------------------------------------------------

    def read(self) -> Optional[Ps01Reading]:
        """
        读一批数据，返回**最新的完整采样**，或 None。

        每次调用把串口缓冲区里所有可读字节吃掉，逐行解析；
        如果这一批里凑齐了 3 个通道，就返回一个完整的 Ps01Reading。
        """
        if not self.is_open:
            raise Ps01Error("串口未打开")

        try:
            waiting = self._ser.in_waiting
        except OSError:
            return None
        if not waiting:
            return None

        try:
            chunk = self._ser.read(waiting)
        except serial.SerialException as exc:
            raise Ps01Error(f"读取失败：{exc}") from exc

        text = chunk.decode("utf-8", errors="ignore")
        self._buf += text

        # 按行处理（最后不完整的一行留在缓冲区）
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._handle_line(line.strip())

        return self._last_complete

    def _handle_line(self, line: str) -> None:
        if not line:
            return

        m = RE_RAW.match(line)
        if m:
            self._raw_lines += 1
            try:
                idx = int(m.group(1))
                value = int(m.group(2), 16)
                status = int(m.group(3), 16)
            except ValueError:
                self._parse_errors += 1
                return
            self._apply_sample(idx, value, status)
            return

        if RE_JSON.match(line):
            # 状态消息，记录备查
            try:
                import json
                payload = json.loads(line)
            except (ValueError, TypeError):
                return
            self._status_messages.append(payload)
            if len(self._status_messages) > 50:
                self._status_messages.pop(0)
            if isinstance(payload, dict) and payload.get("status") == "sensor_state":
                self._log(f"传感器状态: {payload.get('value')}")
            return

        # 其他都是 [INFO]/[DEBUG] 日志，忽略

    def _apply_sample(self, idx: int, value: int, status: int) -> None:
        slot = idx % 3
        reading = self._current

        if slot == 0:
            reading.pd3, reading.status3 = value, status
            self._seen.add("PD3")
        elif slot == 1:
            reading.pd2, reading.status2 = value, status
            self._seen.add("PD2")
        else:
            reading.pd1, reading.status1 = value, status
            self._seen.add("PD1")

        # 三个通道都到齐 → 这是一次完整采样
        if len(self._seen) == 3:
            reading.complete = True
            reading.timestamp = time.monotonic()
            self._sample_count += 1
            self._last_complete = reading
            # 开启新一轮
            self._current = Ps01Reading()
            self._seen = set()

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "samples": self._sample_count,
            "raw_lines": self._raw_lines,
            "parse_errors": self._parse_errors,
            "status_messages": len(self._status_messages),
        }

    def describe(self) -> str:
        return (f"PS01({self.port} @ {self.baudrate} 8N1, "
                f"采样 {self._sample_count}, RAW行 {self._raw_lines})")


# ---------------------------------------------------------------------------
# 自测：直接打印原始解析结果
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse
    import sys

    ap = argparse.ArgumentParser(
        description="PS01 传感器读取测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
默认行为：
  1. 打开串口
  2. 读掉初始化日志（约 3 秒），直到传感器安静下来
  3. 发送启动命令 gnst
  4. 等待 ~measurement started~ 回显
  5. 开始统计并打印三通道读数

常用：
  python3 ps01_driver.py                      # 默认 gnst\\r\\n
  python3 ps01_driver.py --cmd-lf             # 改用 gnst\\n
  python3 ps01_driver.py --cmd-raw            # 改用 gnst（无结尾）
  python3 ps01_driver.py --start-cmd "67 6E 73 74 0D 0A"
""")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200,
                    help="波特率，默认 115200（可用来补偿时钟误差）")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--start-cmd", default=None,
                    help="自定义启动命令（十六进制字节），如 '67 6E 73 74 0D 0A'")
    ap.add_argument("--cmd-lf", action="store_true",
                    help="启动命令用 gnst\\n（仅 LF）")
    ap.add_argument("--cmd-raw", action="store_true",
                    help="启动命令用 gnst（无行结尾）")
    ap.add_argument("--cmd-cr", action="store_true",
                    help="启动命令用 gnst\\r（仅 CR）")
    ap.add_argument("--init-wait", type=float, default=3.5,
                    help="打开串口后等初始化日志输出完的秒数，默认 3.5")
    ap.add_argument("--grace", type=float, default=1.0,
                    help="发命令后等待多少秒再开始统计")
    ap.add_argument("--no-start", action="store_true",
                    help="不发启动命令，只被动监听（调试用）")
    ap.add_argument("--dtr", type=int, choices=[0, 1], default=1,
                    help="DTR 电平，默认 1（高，匹配 PS01Tool/pyserial 默认）")
    ap.add_argument("--rts", type=int, choices=[0, 1], default=1,
                    help="RTS 电平，默认 1（高，匹配 PS01Tool/pyserial 默认）")
    args = ap.parse_args()

    # 决定启动命令
    if args.start_cmd:
        cmd = bytes(int(x, 16) for x in
                    args.start_cmd.replace(",", " ").split())
    elif args.cmd_lf:
        cmd = b"gnst\n"
    elif args.cmd_raw:
        cmd = b"gnst"
    elif args.cmd_cr:
        cmd = b"gnst\r"
    else:
        cmd = DEFAULT_START_COMMAND     # b"gnst\r\n"

    sensor = Ps01Sensor(args.port, baudrate=args.baud, start_command=cmd,
                        dtr=bool(args.dtr), rts=bool(args.rts), log=print)
    try:
        sensor.open()
    except Ps01Error as exc:
        print(f"❌ {exc}")
        return 2

    try:
        # ---- 1. 等待初始化完成 ----
        # 关键：PS01Tool 的实际做法是"看到 Data callback registered 就立刻发 gnst"。
        # 这行日志是固件初始化完成的信号，比固定等待可靠得多。
        print(f"\n[1] 等待初始化（最多 {args.init_wait}s，"
              f"看到 'Data callback registered' 立即继续）…")
        t_init = time.monotonic()
        init_lines = 0
        ready = False
        tail = ""          # 跨行的残片，防止信号被行拆分截断
        while time.monotonic() - t_init < args.init_wait:
            try:
                waiting = sensor._ser.in_waiting
            except OSError:
                waiting = 0
            if waiting:
                chunk = sensor._ser.read(waiting).decode("utf-8", "ignore")
                buf = tail + chunk
                for line in buf.splitlines():
                    s = line.strip()
                    if not s:
                        continue
                    init_lines += 1
                    if "RAW[" in s:
                        sensor._handle_line(s)
                    if "Data callback registered" in s or "Init success" in s:
                        ready = True
                tail = buf.splitlines()[-1] if buf.splitlines() else ""
                if ready:
                    break
            time.sleep(0.02)
        print(f"    读到 {init_lines} 行"
              + ("；✅ 收到初始化完成信号" if ready else "；⚠️ 未见完成信号（超时）"))

        # ---- 2. 立即发送启动命令 ----
        if args.no_start:
            print("\n[2] --no-start：跳过发送命令，直接监听")
        else:
            print(f"\n[2] 立即发送启动命令 {cmd!r}")
            sensor.send(cmd)
            time.sleep(args.grace)

            # ---- 3. 检查是否触发 ----
            triggered = False
            try:
                waiting = sensor._ser.in_waiting
            except OSError:
                waiting = 0
            if waiting:
                echo = sensor._ser.read(waiting).decode("utf-8", "ignore")
                # 回显里可能已经带着 RAW 数据，一并交给解析器
                for line in echo.splitlines():
                    if "RAW[" in line:
                        sensor._handle_line(line.strip())
                if "measurement started" in echo:
                    triggered = True
                    print("    ✅ 收到回显：~measurement started~")
                else:
                    print(f"    回显片段: {echo[:300]!r}")
            if not triggered:
                print("    ⚠️ 没看到 'measurement started' 回显")
                print("       依次试： --dtr 1 --rts 1 / --cmd-raw / --cmd-lf")
                print("       或者： --no-start 看它自己会不会输出")

        # ---- 4. 开始统计 ----
        print(f"\n[3] 开始读取 {args.seconds}s（按压传感器）\n")
        t0 = time.monotonic()
        last_print = 0.0
        n = 0
        while time.monotonic() - t0 < args.seconds:
            reading = sensor.read()
            if reading is not None:
                n += 1
                now = time.monotonic()
                if now - last_print > 0.2:
                    last_print = now
                    rate = n / max(0.001, now - t0)
                    print(f"\r{reading}   [{rate:5.1f} Hz]   ",
                          end="", flush=True)
            else:
                time.sleep(0.005)
    except KeyboardInterrupt:
        print("\n中断")
    finally:
        print()
        print(f"\n统计: {sensor.stats()}")
        sensor.close()
    return 0 if n > 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
