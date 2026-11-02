#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PS01 光学力传感器驱动库。

传感器简介
----------
PS01 = ams OSRAM AS7150 光学力传感 AFE + 红外 LED + 光电二极管（PD）+ 厂商 MCU。

    · 接口：UART 115200 8N1，3.3V TTL，4 pin（3V3 / TXD / RXD / GND）
    · 输出：三路 PD 读数（PD1 / PD2 / PD3），约 135 Hz
    · 接线：经 CH340 USB-TTL 接到板子的 USB 口 → /dev/ttyUSB0

★ 核心发现：隐藏的 0xD4 唤醒字节
--------------------------------
厂商上位机 PS01Tool 每次发命令都是 **`0xD4` + cmd + `\\r\\n`**（共 7 字节）。
`0xD4` 是传感器固件要求的唤醒/同步字节，**缺失时整包被丢弃**，
传感器不会有任何响应。

这个字节是通过反编译厂商上位机（PyInstaller 打包的 Python 3.9 程序，
用 pyinstxtractor-ng 解包 + xdis 反汇编）发现的 —— 见
``docs/技术文档.md`` §4。

数据格式
--------
初始化阶段输出约 49~51 行日志，以 ``[INFO] Data callback registered`` 结束。

数据行::

    RAW[<索引>]: 0x<AD值 hex> (status=0x<状态 hex>)

通道分配（``索引 % 3``）::

    0 → PD3
    1 → PD2
    2 → PD1

状态行（JSON）::

    {"status":"sensor_init","result":1}
    {"status":"sensor_state","value":"NORMAL"}

用法
----
::

    from grape_gripper.ps01 import Ps01Reader

    with Ps01Reader("/dev/ttyUSB0") as r:
        r.start()
        while True:
            print(r.pd1, r.pd2, r.pd3)
"""
from __future__ import annotations

import re
import threading
import time
from typing import Optional

try:
    import serial  # pyserial
except ImportError:  # pragma: no cover
    serial = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# 通信常量
# ---------------------------------------------------------------------------
# ⚠️ 0xD4 不能省！详见模块文档和 docs/技术文档.md §4.5
DEFAULT_START_COMMAND = b"\xd4gnst\r\n"     # D4 67 6E 73 74 0D 0A
DEFAULT_STOP_COMMAND = b"\xd4gned\r\n"      # D4 67 6E 65 64 0D 0A

DEFAULT_BAUDRATE = 115200
DEFAULT_PORT = "/dev/ttyUSB0"

# 数据行正则
_RAW_RE = re.compile(
    r"RAW\[\s*(\d+)\s*\]\s*:\s*0x([0-9A-Fa-f]+)\s*\(\s*status\s*=\s*0x([0-9A-Fa-f]+)\s*\)"
)

# 索引 → PD 通道（0-based: 0=PD3, 1=PD2, 2=PD1）
_INDEX_TO_CHANNEL = {0: 3, 1: 2, 2: 1}


class Ps01Error(Exception):
    """PS01 通信或解析错误。"""


class Ps01Reader:
    """
    PS01 力传感器读取器（后台线程持续采集，对外暴露最新值）。

    设计说明：传感器是**持续推流**的（约 135 Hz），所以这里用后台线程
    不断读串口并更新缓存，调用方随时取最新值即可，不需要阻塞等待。
    """

    def __init__(self,
                 port: str = DEFAULT_PORT,
                 baudrate: int = DEFAULT_BAUDRATE,
                 start_command: bytes = DEFAULT_START_COMMAND,
                 stop_command: bytes = DEFAULT_STOP_COMMAND,
                 read_timeout: float = 0.05,
                 auto_start: bool = True,
                 wait_init: bool = True,
                 init_timeout: float = 5.0,
                 dtr: bool = True,
                 rts: bool = True):
        if serial is None:
            raise Ps01Error("缺少 pyserial： sudo apt install -y python3-serial")

        self.port = port
        self.baudrate = baudrate
        self.start_command = start_command
        self.stop_command = stop_command
        self.read_timeout = read_timeout
        self.auto_start = auto_start
        # ★ 必须先等初始化日志打完，再发启动命令 —— 见 open() 的说明
        self.wait_init = wait_init
        self.init_timeout = init_timeout
        self.dtr = dtr
        self.rts = rts

        self._ser = None
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()

        # 最新读数
        self._pd = {1: None, 2: None, 3: None}   # type: dict[int, Optional[int]]
        self._last_update = 0.0

        # 统计
        self.sample_count = 0
        self.raw_lines = 0
        self.total_lines = 0        # 收到的所有行（含未识别的，便于诊断）
        self.parse_errors = 0
        self.status_messages = 0
        self.init_done = False
        self.init_seen = False     # 是否真的看到过初始化日志
        self.last_status = ""

    # ---------------- 生命周期 ----------------

    def open(self) -> "Ps01Reader":
        """
        打开串口并启动采集。

        ★ 启动时序（关键，实测得出）
        --------------------------------
        打开串口这个动作会让 CH340 的 DTR/RTS 跳变，传感器因此复位并
        自动打印约 50 行初始化日志，最后一行是 ``[INFO] Data callback registered``。
        **在这之前发任何命令都会被丢掉。**

        正确时序::

            ① 打开串口
            ② 等初始化日志打完（看到 "Data callback registered"）
            ③ 才发送 \\xd4gnst\\r\\n
            ④ 传感器回显 ~measurement started~，随后以约 135Hz 推流

        因此本方法分两步，而不是打开后立刻发命令。
        """
        try:
            self._ser = serial.Serial(
                self.port, self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=self.read_timeout,
                rtscts=False,
                dsrdtr=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise Ps01Error(
                f"打开 {self.port} 失败：{exc}\n"
                "  检查：设备节点是否存在（ls /dev/ttyUSB*）、"
                "是否被别的程序占用、权限是否足够（用 sudo）"
            ) from exc

        # CH340 的 DTR/RTS 常接到目标板的 EN/BOOT 脚，显式设成和
        # 已验证脚本（scripts/ps01_driver.py）一致的电平
        try:
            self._ser.dtr = self.dtr
            self._ser.rts = self.rts
        except (OSError, AttributeError):
            pass

        # 丢弃打开串口时产生的残留字节
        try:
            self._ser.reset_input_buffer()
        except OSError:
            pass

        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._reader_loop,
                                        name="ps01-reader", daemon=True)
        self._thread.start()

        # ★ 第一步：等初始化完成（但不是必须的）
        #
        # 为什么"没看到初始化日志"不该直接报错：
        #   传感器 A 用 CH340，打开时必然打印约 50 行初始化日志，
        #   最后一行是 "[INFO] Data callback registered"。
        #   传感器 B 用 PL2303，实测打开后**一行都没有** ——
        #   因为它的 MCU 状态还停在"已初始化"（之前在电脑上测过），
        #   发启动命令后直接就开始推流。
        #   所以这里改成：超时只警告，继续发启动命令；
        #   由后面的 wait_first_sample() 用"有没有数据"来做最终判断。
        if self.wait_init:
            if self.wait_init_done(self.init_timeout):
                self.init_seen = True
            else:
                print(f"⚠️ {self.port} 未出现初始化日志"
                      f"（{self.init_timeout:.1f}s）—— 传感器可能已处于就绪态，"
                      f"继续尝试启动")

        # ★ 第二步：现在才发启动命令
        if self.auto_start:
            self.start()

        return self

    def wait_init_done(self, timeout: float = 5.0) -> bool:
        """等传感器打印完初始化日志（出现 'Data callback registered'）。"""
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.init_done:
                return True
            time.sleep(0.01)
        return False

    def start(self) -> None:
        """发送启动命令（含 0xD4 唤醒字节）。"""
        self.send(self.start_command)

    def stop(self) -> None:
        """发送停止命令。"""
        self.send(self.stop_command)

    def send(self, payload: bytes) -> None:
        if self._ser is None:
            raise Ps01Error("串口未打开")
        try:
            self._ser.write(payload)
            self._ser.flush()
        except Exception as exc:  # noqa: BLE001
            raise Ps01Error(f"发送失败：{exc}") from exc

    def close(self) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
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
        return self._ser is not None and bool(getattr(self._ser, "is_open", False))

    # ---------------- 采集线程 ----------------

    def _reader_loop(self) -> None:
        buf = b""
        while not self._stop_evt.is_set():
            if self._ser is None:
                break
            try:
                chunk = self._ser.read(256)
            except Exception:  # noqa: BLE001
                break
            if not chunk:
                continue
            buf += chunk
            # 按行切分
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                text = line.decode("utf-8", "replace").strip()
                if text:
                    self._handle_line(text)
            # 防止缓冲无限增长
            if len(buf) > 8192:
                buf = buf[-1024:]

    def _handle_line(self, line: str) -> None:
        self.total_lines += 1

        m = _RAW_RE.search(line)
        if m:
            self.raw_lines += 1
            try:
                idx = int(m.group(1))
                value = int(m.group(2), 16)
            except ValueError:
                self.parse_errors += 1
                return

            channel = _INDEX_TO_CHANNEL.get(idx % 3)
            if channel is None:
                self.parse_errors += 1
                return

            with self._lock:
                self._pd[channel] = value
                self._last_update = time.time()
                self.sample_count += 1
            return

        if "Data callback registered" in line:
            self.init_done = True
            return

        if "status" in line.lower():
            self.status_messages += 1
            self.last_status = line

    # ---------------- 读数 ----------------

    @property
    def pd1(self) -> Optional[int]:
        with self._lock:
            return self._pd[1]

    @property
    def pd2(self) -> Optional[int]:
        with self._lock:
            return self._pd[2]

    @property
    def pd3(self) -> Optional[int]:
        with self._lock:
            return self._pd[3]

    def channel(self, ch: int) -> Optional[int]:
        """按通道号取读数（1 / 2 / 3）。"""
        with self._lock:
            return self._pd.get(ch)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "pd1": self._pd[1], "pd2": self._pd[2], "pd3": self._pd[3],
                "samples": self.sample_count,
                "age_s": (time.time() - self._last_update) if self._last_update else None,
            }

    @property
    def data_age_s(self) -> Optional[float]:
        """距离最后一次更新的时间（秒）。用于判断数据是否新鲜。"""
        with self._lock:
            if not self._last_update:
                return None
            return time.time() - self._last_update

    def wait_first_sample(self, timeout: float = 3.0) -> bool:
        """等第一帧数据到达。"""
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.sample_count > 0:
                return True
            time.sleep(0.01)
        return False

    def stats(self) -> dict:
        return {
            "port": self.port,
            "baudrate": self.baudrate,
            "total_lines": self.total_lines,
            "samples": self.sample_count,
            "raw_lines": self.raw_lines,
            "parse_errors": self.parse_errors,
            "status_messages": self.status_messages,
            "init_done": self.init_done,
            "pd1": self.pd1, "pd2": self.pd2, "pd3": self.pd3,
        }

    def describe(self) -> str:
        age = self.data_age_s
        fresh = "无数据" if age is None else f"{age*1000:.0f}ms前"
        return (f"Ps01Reader({self.port}@{self.baudrate} "
                f"samples={self.sample_count} {fresh})")