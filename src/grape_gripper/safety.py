# -*- coding: utf-8 -*-
"""
独立安全看门狗。

为什么要在控制逻辑之外再写一层？
    控制逻辑越复杂，越容易因为一个参数写错、一个异常没捕获而"卡住"。
    卡住的时候舵机会保持最后的脉宽输出 → 一直夹着葡萄 → 压破。

    安全层的工作只有一个：**发现不对劲，就卸力。**

三层保护
--------
1. ``HeartbeatWatchdog``  — 主循环心跳超时（主程序卡死/崩溃）
2. ``ForceWatchdog``      — 峰值力超限，独立于控制器的判断
3. ``EmergencyStop``      — 外部信号（键盘 / 按钮 / 上位机指令）
"""

from __future__ import annotations

import signal
import threading
import time
from typing import Callable, Optional


class HeartbeatWatchdog:
    """
    心跳看门狗。

    主循环每周期调用 ``beat()``。如果在后台线程里发现超过
    ``timeout_s`` 没有心跳，就执行 ``on_timeout``（通常是舵机卸力）。

    作用场景：主程序抛了未捕获异常、陷入死循环、或者被 SIGSTOP。

    注意：看门狗线程只做"卸力"这一件简单的事，不要在里面做复杂逻辑，
    否则它自己也可能卡住。
    """

    def __init__(self, timeout_s: float = 0.5,
                 on_timeout: Optional[Callable[[], None]] = None,
                 log: Optional[Callable[[str], None]] = None):
        self.timeout_s = max(0.05, timeout_s)
        self.on_timeout = on_timeout
        self._log = log or (lambda _m: None)
        self._last_beat = time.monotonic()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._tripped = False

    def beat(self) -> None:
        """主循环每周期调用一次。"""
        self._last_beat = time.monotonic()
        self._tripped = False

    @property
    def tripped(self) -> bool:
        return self._tripped

    def start(self) -> "HeartbeatWatchdog":
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="watchdog", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        # 用较短的轮询间隔，保证及时性
        interval = min(0.05, self.timeout_s / 4.0)
        while not self._stop.wait(interval):
            idle = time.monotonic() - self._last_beat
            if idle > self.timeout_s:
                if not self._tripped:
                    self._tripped = True
                    self._log(f"[看门狗] 心跳超时 {idle:.2f}s > {self.timeout_s}s，执行安全动作")
                    if self.on_timeout is not None:
                        try:
                            self.on_timeout()
                        except Exception as exc:   # noqa: BLE001 - 安全层必须吞异常
                            self._log(f"[看门狗] 安全动作本身失败：{exc}")

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._thread = None


class ForceWatchdog:
    """
    独立的力峰值监控。

    与控制器的 ``_check_safety`` 是**冗余**关系 —— 故意重复。
    因为控制器里可能有 bug，而这里只做一次最笨的比较。

    用法：每个周期 ``feed(sensors.max_force_n)``。
    """

    def __init__(self, limit_n: float = 1.5,
                 on_violation: Optional[Callable[[float], None]] = None,
                 log: Optional[Callable[[str], None]] = None):
        self.limit_n = limit_n
        self.on_violation = on_violation
        self._log = log or (lambda _m: None)
        self.violated = False
        self.peak_n = 0.0

    def feed(self, force_n: float) -> bool:
        self.peak_n = max(self.peak_n, force_n)
        if force_n > self.limit_n and not self.violated:
            self.violated = True
            self._log(f"[力监控] {force_n:.3f}N 超过硬限 {self.limit_n}N")
            if self.on_violation is not None:
                try:
                    self.on_violation(force_n)
                except Exception as exc:   # noqa: BLE001
                    self._log(f"[力监控] 触发动作失败：{exc}")
                return True
        return False

    def reset(self) -> None:
        self.violated = False
        self.peak_n = 0.0


class EmergencyStop:
    """
    急停：捕获 Ctrl+C（SIGINT）和 SIGTERM，转成一个可查询的标志位。

    这样主循环可以优雅收尾 —— 而不是被 KeyboardInterrupt 打断在
    某个中间状态、留下一个还夹着葡萄的舵机。
    """

    def __init__(self, log: Optional[Callable[[str], None]] = None):
        self._log = log or (lambda _m: None)
        self._requested = False
        self._installed = False
        self._prev_handlers: dict = {}

    @property
    def requested(self) -> bool:
        return self._requested

    def request(self, reason: str = "外部请求") -> None:
        if not self._requested:
            self._log(f"[急停] {reason}")
        self._requested = True

    def install(self) -> "EmergencyStop":
        if self._installed:
            return self
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._prev_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, self._handler)
            except (ValueError, OSError):
                # 非主线程里装不了信号处理器，忽略
                pass
        self._installed = True
        return self

    def _handler(self, signum, _frame) -> None:
        name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        self.request(f"收到信号 {name}")

    def uninstall(self) -> None:
        if not self._installed:
            return
        for sig, prev in self._prev_handlers.items():
            try:
                signal.signal(sig, prev)
            except (ValueError, OSError, TypeError):
                pass
        self._installed = False

    def clear(self) -> None:
        self._requested = False
