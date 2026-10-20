# -*- coding: utf-8 -*-
"""
夹爪力闭环控制器 ★ 本项目核心

控制思想
--------
葡萄能承受的力约 1~3N，而"夹住不掉"只需要 0.8N 左右（靠硅胶垫的摩擦）。
所以这个任务不是"用力夹住"，而是"轻轻碰住"。

难点在于：舵机是**位置器件**，你只能告诉它转到多少度，
它不知道现在用了多大力。所以必须：

    角度 ↑  ──▶  夹指逼近葡萄  ──▶  接触  ──▶  力 ↑
                                                  │
    角度增量 ←──  PI 控制器  ←──  力误差 (F* - F) ─┘

即把压力传感器的读数作为反馈，把舵机的**位置增量**当作"力的执行器"。
这就是本模块做的事。

状态机
------
    INIT ──▶ OPEN ──▶ APPROACH ──▶ CONTACT ──▶ RAMP ──▶ HOLD ──▶ RELEASE ──▶ DONE
               ▲                                  │          │
               └────────────── ABORT ◀────────────┴──────────┘
                     （超力 / 失稳 / 超时 / 外部急停）

安全设计（每一层都是独立的，任一层触发就停）
------------------------
 1. 硬限：单侧力 > force_max_n          → 立即反向松开
 2. 速率限幅：力的上升速率受 force_ramp_n_per_s 限制 → 防过冲
 3. 失稳检测：力上不去但舵机一直在推进     → 判定滑移，退开重试
 4. 超时：任何阶段超时                    → 卸力（disable，不是保持夹紧）
 5. 心跳：主循环卡住                      → 外部看门狗卸力
"""

from __future__ import annotations

import csv
import enum
import os
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .sensor import ForceSensorPair


# ---------------------------------------------------------------------------
# 状态定义
# ---------------------------------------------------------------------------

class GripState(enum.Enum):
    INIT = "init"
    OPEN = "open"              # 张开到位，等待开始
    APPROACH = "approach"      # 闭合接近，尚未接触
    CONTACT = "contact"        # 刚检测到接触，切换到力控
    RAMP = "ramp"              # 力上升阶段（受速率限制）
    HOLD = "hold"              # 到达目标力，保持
    RELEASE = "release"        # 松开
    DONE = "done"              # 成功完成
    ABORT = "abort"            # 安全中止


class AbortReason(enum.Enum):
    NONE = "none"
    OVER_FORCE = "over_force"          # 超力
    SLIP = "slip"                      # 滑移/失稳
    TIMEOUT = "timeout"                # 超时
    ASYMMETRY = "asymmetry"            # 严重偏心
    EXTERNAL = "external"              # 外部请求中止
    HARDWARE = "hardware"              # 硬件异常
    SENSOR_DEAD = "sensor_dead"        # 传感器读数异常（一直为 0 或饱和）


# ---------------------------------------------------------------------------
# 数据记录
# ---------------------------------------------------------------------------

@dataclass
class ControlSample:
    """控制循环的一个采样点，用于事后分析力曲线。"""
    t: float
    state: str
    angle_deg: float
    force_n: float
    force_left_n: float
    force_right_n: float
    target_force_n: float
    asymmetry: float


@dataclass
class GripResult:
    """一次夹取的结果。"""
    success: bool
    state: GripState
    reason: AbortReason = AbortReason.NONE
    message: str = ""
    duration_s: float = 0.0
    peak_force_n: float = 0.0
    final_angle_deg: float = 0.0
    samples: List[ControlSample] = field(default_factory=list)

    def summary(self) -> str:
        status = "成功" if self.success else "失败"
        return (f"[{status}] 状态={self.state.value} "
                f"原因={self.reason.value} 耗时={self.duration_s:.2f}s "
                f"峰值力={self.peak_force_n:.2f}N 最终角度={self.final_angle_deg:.1f}° "
                f"{self.message}")


# ---------------------------------------------------------------------------
# 控制器
# ---------------------------------------------------------------------------

class GripperController:
    """
    夹爪力闭环控制器。

    由外部主循环驱动：每周期调用 ``step()``，然后读取 ``state``。

    典型用法::

        ctrl = GripperController(cfg, sensors, servo)
        ctrl.start_grip()                 # 请求开始一次夹取
        while not ctrl.finished:
            ctrl.step()
            time.sleep(0.01)              # 100Hz
        print(ctrl.result.summary())

    参数全部来自 config.yaml，本类不做任何硬编码的力学假设。
    """

    def __init__(self, cfg, sensors: ForceSensorPair, servo,
                 log: Optional[Callable[[str], None]] = None):
        self.cfg = cfg
        self.sensors = sensors
        self.servo = servo
        self._log_fn = log

        # ---- 力设定值 ----
        self.target_force_n = cfg.get_float("grip.target_force_n", 0.8)
        self.force_approach_n = cfg.get_float("grip.force_approach_n", 0.25)
        self.force_max_n = cfg.get_float("grip.force_max_n", 1.5)
        self.force_release_n = cfg.get_float("grip.force_release_n", 0.05)

        # ---- 速率 ----
        self.approach_rate = cfg.get_float("grip.approach_rate_deg_per_s", 25.0)
        self.force_ramp_rate = cfg.get_float("grip.force_ramp_n_per_s", 1.5)
        self.release_rate = cfg.get_float("grip.release_rate_deg_per_s", 120.0)

        # ---- PI ----
        self.kp = cfg.get_float("grip.kp_deg_per_s_per_n", 60.0)
        self.ki = cfg.get_float("grip.ki_deg_per_s_per_n_s", 40.0)
        self.integral_limit = cfg.get_float("grip.integral_limit", 30.0)

        # ---- 阈值 ----
        self.contact_confirm = cfg.get_int("grip.contact_confirm_samples", 5)
        self.hold_time_s = cfg.get_float("grip.hold_time_s", 2.0)
        self.hold_tolerance_n = cfg.get_float("grip.hold_tolerance_n", 0.15)
        self.timeout_approach_s = cfg.get_float("grip.timeout_approach_s", 8.0)
        self.timeout_hold_s = cfg.get_float("grip.timeout_hold_s", 10.0)

        # ---- 安全 ----
        self.safety_enable = cfg.get_bool("safety.enable", True)
        self.release_backoff_deg = cfg.get_float("safety.release_backoff_deg", 20.0)
        self.slip_ratio = cfg.get_float("safety.slip_ratio_threshold", 0.2)
        self.slip_window_s = cfg.get_float("safety.slip_detect_window_s", 1.5)

        # ---- 机械极限 ----
        self.angle_open = cfg.get_float("servo.angle_open_deg", 10.0)
        self.angle_close = cfg.get_float("servo.angle_close_deg", 95.0)
        # 闭合方向 = 角度增大（正装）；如果结构反过来，把两个角度对调即可
        self.close_sign = 1.0 if self.angle_close >= self.angle_open else -1.0

        # ---- 运行时状态 ----
        self.state = GripState.INIT
        self.result = GripResult(success=False, state=GripState.INIT)
        self._state_enter_time = time.monotonic()
        self._t0 = self._state_enter_time
        self._integral = 0.0
        self._contact_count = 0
        self._contact_force = 0.0
        self._contact_angle = 0.0
        self._hold_enter_time = 0.0
        self._peak_force = 0.0
        self._samples: List[ControlSample] = []
        self._abort_reason = AbortReason.NONE
        self._abort_message = ""
        self._grip_requested = False
        self._ramp_start_force = 0.0
        self._slip_ref_time = 0.0
        self._slip_ref_force = 0.0
        self._max_samples = 20000   # 防止内存无限增长（100Hz 下约 200s）

    # ------------------------------------------------------------------
    # 日志
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self._log_fn is not None:
            self._log_fn(msg)

    # ------------------------------------------------------------------
    # 状态切换
    # ------------------------------------------------------------------

    def _enter(self, state: GripState) -> None:
        if state is self.state:
            return
        self._log(f"[状态] {self.state.value} → {state.value}")
        self.state = state
        self._state_enter_time = time.monotonic()

        if state is GripState.CONTACT:
            # 记录接触时的基准，用于速率限幅和滑移检测
            self._contact_force = self.sensors.force_n
            self._contact_angle = self.servo.angle
            self._integral = 0.0
            self._slip_ref_time = time.monotonic()
            self._slip_ref_force = self.sensors.force_n
        elif state is GripState.RAMP:
            # 力速率限幅的时间基准，与 _contact_force 一起使用
            self._ramp_start_time = time.monotonic()
            self._ramp_start_force = self.sensors.force_n
        elif state is GripState.HOLD:
            self._hold_enter_time = time.monotonic()

    def _elapsed_in_state(self) -> float:
        return time.monotonic() - self._state_enter_time

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------

    def start_grip(self) -> None:
        """请求开始一次夹取（下一个 step 生效）。"""
        if self.state in (GripState.ABORT,):
            self._log("处于 ABORT 状态，先调用 reset()")
            return
        self._grip_requested = True
        self._log("收到夹取请求")

    def reset(self) -> None:
        """复位到 INIT，准备下一次夹取。"""
        self.state = GripState.INIT
        self.result = GripResult(success=False, state=GripState.INIT)
        self._integral = 0.0
        self._contact_count = 0
        self._peak_force = 0.0
        self._samples = []
        self._abort_reason = AbortReason.NONE
        self._abort_message = ""
        self._grip_requested = False
        self._t0 = time.monotonic()
        self.sensors.reset_filter()

    def abort(self, reason: AbortReason = AbortReason.EXTERNAL,
              message: str = "") -> None:
        """外部请求中止（如急停按钮、视觉失败）。"""
        if self.state in (GripState.DONE, GripState.ABORT):
            return
        self._abort_reason = reason
        self._abort_message = message or reason.value
        self._enter(GripState.ABORT)

    @property
    def finished(self) -> bool:
        return self.state in (GripState.DONE, GripState.ABORT)

    @property
    def is_gripping(self) -> bool:
        return self.state in (GripState.RAMP, GripState.HOLD)

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    def step(self, dt: Optional[float] = None) -> GripState:
        """
        推进一个控制周期。返回当前状态。

        ``dt`` 若为 None 则用实际经过时间。控制周期建议 50~200Hz。
        """
        now = time.monotonic()
        if dt is None:
            dt = max(1e-4, now - getattr(self, "_last_step_time", now))
        self._last_step_time = now

        # 采样（每个周期都要采，滤波需要连续数据）
        force = self.sensors.update()
        self._peak_force = max(self._peak_force, self.sensors.max_force_n)

        self._check_safety()
        if self.state is GripState.ABORT:
            self._record(now)
            self._finish(False)
            return self.state

        handler = {
            GripState.INIT: self._step_init,
            GripState.OPEN: self._step_open,
            GripState.APPROACH: self._step_approach,
            GripState.CONTACT: self._step_contact,
            GripState.RAMP: self._step_ramp,
            GripState.HOLD: self._step_hold,
            GripState.RELEASE: self._step_release,
            GripState.DONE: lambda d: None,
            GripState.ABORT: lambda d: None,
        }[self.state]

        handler(dt)
        self.servo.update(now)
        self._record(now)

        if self.finished and self.result.duration_s == 0.0:
            self._finish(self.state is GripState.DONE)
        return self.state

    # ------------------------------------------------------------------
    # 各状态实现
    # ------------------------------------------------------------------

    def _step_init(self, dt: float) -> None:
        """初始化：确认传感器活着，然后张开夹爪。"""
        self.servo.set_angle(self.angle_open)
        # 传感器自检：读数应该远低于接触阈值（否则说明有预载或接线错）
        if self.sensors.force_n > self.force_approach_n:
            self._log(
                f"警告：初始力 {self.sensors.force_n:.2f}N 已超过接触阈值 "
                f"{self.force_approach_n}N。可能夹爪未全开、或传感器零点漂移，"
                "建议先执行 tare（去皮）。"
            )
        self._enter(GripState.OPEN)

    def _step_open(self, dt: float) -> None:
        """等夹爪张开到位，然后等夹取请求。"""
        if not self.servo.at_target:
            return
        if self._grip_requested:
            self._grip_requested = False
            # 开始前清一次滤波器，避免残留旧数据
            self.sensors.reset_filter()
            self._enter(GripState.APPROACH)

    def _step_approach(self, dt: float) -> None:
        """
        闭合接近阶段：**速度受控**地向闭合方向推进，主动寻找接触。

        为什么速度要慢？
            接触瞬间会产生冲量。速度越快，"撞"上去的力越大，越容易破皮。
            慢速接近让力有一个平滑的建立过程，PI 环也来得及反应。
            葡萄（直径 22mm）在典型夹爪上对应约 20~30° 的舵机行程，
            60°/s 大约 0.4s 完成逼近，兼顾速度和温和性。
        """
        if self._elapsed_in_state() > self.timeout_approach_s:
            self._abort_reason = AbortReason.TIMEOUT
            self._abort_message = (
                f"接近阶段超时 {self.timeout_approach_s}s 未检测到接触。"
                "检查：葡萄是否在夹爪行程内？夹爪是否真的在闭合？"
            )
            self._enter(GripState.ABORT)
            return

        # 受控闭合。
        # 目标角度必须同时受夹爪**机械行程**约束：
        #   angle_open ~ angle_close 才是夹爪真正能走的范围，
        #   0~180° 只是舵机的电气行程。不夹这一下就会把夹爪顶死、憋舵机。
        lo, hi = sorted((self.angle_open, self.angle_close))
        step = self.close_sign * self.approach_rate * dt
        self.servo.set_angle(max(lo, min(hi, self.servo.angle + step)))

        # 到位但没接触 → 说明夹空了
        angle_span = max(1e-6, abs(self.angle_close - self.angle_open))
        reached_limit = abs(self.servo.angle - self.angle_close) < max(0.5, angle_span * 0.01)
        if reached_limit and self.servo.at_target:
            if self._elapsed_in_state() > 0.5:
                self._abort_reason = AbortReason.TIMEOUT
                self._abort_message = (
                    f"夹爪已闭合到机械极限 {self.angle_close:.1f}° 仍未接触物体（夹空）。"
                    "检查葡萄是否在行程内。"
                )
                self._enter(GripState.ABORT)
            return

        # 接触判定：力超过阈值，连续 N 个周期都满足（去抖）
        if self.sensors.force_n >= self.force_approach_n:
            self._contact_count += 1
            if self._contact_count >= self.contact_confirm:
                self._log(
                    f"检测到接触：力={self.sensors.force_n:.3f}N "
                    f"角度={self.servo.angle:.1f}°"
                )
                # 接触瞬间记录当前角度作为起始，并把舵机目标锁定在当前位置，
                # 之后由 PI 环一点一点推进（不再有速度惯性）
                self.servo.set_angle(self.servo.angle)
                self._enter(GripState.CONTACT)
        else:
            self._contact_count = 0

    def _step_contact(self, dt: float) -> None:
        """接触确认后短暂停顿，让读数稳定，然后进入力上升阶段。"""
        if self._elapsed_in_state() < 0.15:
            return
        self._enter(GripState.RAMP)

    def _step_ramp(self, dt: float) -> None:
        """
        力上升阶段：**力闭环 PI 控制**。

        输出 = 期望的闭合速度（deg/s），受三重限制：
          1. PI 输出本身
          2. force_ramp_n_per_s 换算出的力上升速率上限  ★防过冲的关键
          3. servo.max_rate_deg_per_s 机械速率上限（在舵机驱动里）

        第 2 条怎么工作：
            当前力 F，接触时基准力 F0，已过时间 T，
            则允许的最大力 = F0 + ramp_rate * T。
            如果 PI 想要的推进会让力超过这个线，就把推进量削掉。
            这样无论 PI 参数多激进，力都不可能"跳"上去。
        """
        F = self.sensors.force_n

        # 达到目标力 → 进入保持
        if F >= self.target_force_n:
            self._log(f"到达目标力 {F:.3f}N（目标 {self.target_force_n}N）")
            self._enter(GripState.HOLD)
            return

        # --- PI 控制 ---
        error = self.target_force_n - F
        self._integral += error * dt
        self._integral = max(-self.integral_limit,
                             min(self.integral_limit, self._integral))
        desired_rate = self.kp * error + self.ki * self._integral   # deg/s

        # 只允许向闭合方向推进（不允许 PI 反向张开，那会突然失力）
        desired_rate = max(0.0, desired_rate)

        # --- 力上升速率限幅 ★ 防过冲的关键 ---
        # 允许的力上限随时间是条斜线，从接触时的力开始，斜率 = force_ramp_n_per_s。
        # 加上一点前瞻量，否则控制滞后（推进角度后力要过几十毫秒才涨）
        # 会让 PI 环被限幅卡死，永远到不了目标。
        ramp_elapsed = time.monotonic() - self._ramp_start_time
        allowed_force = self._contact_force + self.force_ramp_rate * (
            ramp_elapsed + self._ramp_lead_s()
        )
        if F >= allowed_force:
            desired_rate = 0.0

        delta = self.close_sign * desired_rate * dt
        lo, hi = sorted((self.angle_open, self.angle_close))
        self.servo.set_angle(max(lo, min(hi, self.servo.angle + delta)))

        # 滑移检测在 _check_safety 里做
        # 安全兜底：如果推了很久力还是上不去，超时
        if self._elapsed_in_state() > self.timeout_approach_s * 2:
            self._abort_reason = AbortReason.TIMEOUT
            self._abort_message = (
                f"力上升阶段超时：{self._elapsed_in_state():.1f}s 内只到 {F:.3f}N"
            )
            self._enter(GripState.ABORT)

    def _ramp_lead_s(self) -> float:
        """
        速率限幅的"前瞻"量。

        允许力比理想斜坡线略微超前一点，否则 PI 环会被限幅卡死
        （因为控制是有滞后的：推进角度后力要过几十毫秒才涨）。
        0.25s 是经验值，相当于允许一个控制周期的相位超前。
        """
        return 0.25

    def _step_hold(self, dt: float) -> None:
        """
        保持阶段：把力维持在目标附近。

        FSR 会随温度和时间缓慢漂移，葡萄也会轻微蠕变，所以保持不是
        "什么都不做"，而是小幅度微调。
        """
        elapsed = time.monotonic() - self._hold_enter_time
        if elapsed > min(self.hold_time_s, self.timeout_hold_s):
            self._log(f"保持 {elapsed:.1f}s 完成，准备释放")
            self._enter(GripState.RELEASE)
            return

        F = self.sensors.force_n
        error = self.target_force_n - F

        if abs(error) <= self.hold_tolerance_n:
            # 在容差内，不动（避免无意义的抖动）
            self._integral *= 0.9
            return

        # 小增益修正
        correction_rate = max(-20.0, min(20.0, self.kp * 0.35 * error))
        self.servo.set_angle(
            self.servo.angle + self.close_sign * correction_rate * dt
        )
        # 积分项在保持阶段不再累积，防止缓慢饱和
        self._integral *= 0.95

    def _step_release(self, dt: float) -> None:
        """释放：快速张开到全开角度。"""
        self.servo.set_angle(self.angle_open)
        if self.servo.at_target and self.sensors.force_n <= self.force_release_n:
            self._enter(GripState.DONE)

    # ------------------------------------------------------------------
    # 安全监控
    # ------------------------------------------------------------------

    def _check_safety(self) -> None:
        if not self.safety_enable:
            return
        if self.state in (GripState.INIT, GripState.ABORT, GripState.DONE):
            return

        F = self.sensors.force_n
        Fmax = self.sensors.max_force_n

        # ---- 1. 超力：最高优先级 ----
        if Fmax > self.force_max_n:
            self._abort_reason = AbortReason.OVER_FORCE
            self._abort_message = (
                f"超力！单侧峰值 {Fmax:.3f}N > 上限 {self.force_max_n}N。"
                "立即反向松开。"
            )
            # 先反向退开一点再进 ABORT，避免葡萄一直被压着
            self.servo.set_angle(
                self.servo.angle - self.close_sign * self.release_backoff_deg
            )
            self._enter(GripState.ABORT)
            return

        # ---- 2. 严重偏心 ----
        # 只在力已经明显建立后判断，否则噪声会让 asymmetry 虚高
        if F > self.force_approach_n * 1.5 and self.sensors.asymmetry > 0.6:
            self._abort_reason = AbortReason.ASYMMETRY
            self._abort_message = (
                f"夹持严重偏心（asymmetry={self.sensors.asymmetry:.2f}，"
                f"左={self.sensors.force_left_n:.2f}N 右={self.sensors.force_right_n:.2f}N）。"
                "葡萄可能被推向一侧，继续加力会滑脱或压破。"
            )
            self._enter(GripState.ABORT)
            return

        # ---- 3. 滑移 / 失稳检测 ----
        # 原理：舵机一直在往闭合方向推，但力几乎不涨 → 葡萄在动（滑移），
        # 或者夹爪在打滑。这时继续推只会突然"啪"一下把力加上去。
        if self.state is GripState.RAMP:
            now = time.monotonic()
            window = now - self._slip_ref_time
            if window >= self.slip_window_s:
                dF = F - self._slip_ref_force
                expected = self.force_ramp_rate * window
                if expected > 1e-6 and dF < expected * self.slip_ratio:
                    self._abort_reason = AbortReason.SLIP
                    self._abort_message = (
                        f"检测到滑移/失稳：{window:.1f}s 内只涨了 {dF:.3f}N，"
                        f"期望至少 {expected * self.slip_ratio:.3f}N。"
                        "葡萄可能在滚动或夹指打滑。"
                    )
                    self._enter(GripState.ABORT)
                    return
                self._slip_ref_time = now
                self._slip_ref_force = F

        # ---- 4. 传感器异常 ----
        # 夹爪已经在闭合位置附近却一直读 0 → 大概率接线掉了
        if (self.state in (GripState.RAMP, GripState.HOLD)
                and Fmax <= 1e-6
                and abs(self.servo.angle - self.angle_close) < 30.0):
            self._abort_reason = AbortReason.SENSOR_DEAD
            self._abort_message = "力读数恒为 0 且夹爪已接近闭合，疑似传感器断线"
            self._enter(GripState.ABORT)

    # ------------------------------------------------------------------
    # 记录与收尾
    # ------------------------------------------------------------------

    def _record(self, now: float) -> None:
        if len(self._samples) >= self._max_samples:
            return
        self._samples.append(ControlSample(
            t=now - self._t0,
            state=self.state.value,
            angle_deg=self.servo.angle,
            force_n=self.sensors.force_n,
            force_left_n=self.sensors.force_left_n,
            force_right_n=self.sensors.force_right_n,
            target_force_n=self.target_force_n,
            asymmetry=self.sensors.asymmetry,
        ))

    def _finish(self, success: bool) -> None:
        if self.result.duration_s > 0.0:
            return   # 已经收过尾了
        self.result = GripResult(
            success=success,
            state=self.state,
            reason=self._abort_reason,
            message=self._abort_message,
            duration_s=time.monotonic() - self._t0,
            peak_force_n=self._peak_force,
            final_angle_deg=self.servo.angle,
            samples=self._samples,
        )
        if not success:
            # 失败时卸力 —— 比保持夹紧要安全
            self.servo.disable()
            self._log(f"中止：{self._abort_message}")
        self._log(self.result.summary())

    # ------------------------------------------------------------------
    # 数据导出
    # ------------------------------------------------------------------

    def dump_csv(self, path: str) -> str:
        """把本次力曲线写 CSV，方便用 Excel / matplotlib 分析。"""
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["t_s", "state", "angle_deg", "force_n",
                             "force_left_n", "force_right_n",
                             "target_force_n", "asymmetry"])
            for s in self._samples:
                writer.writerow([
                    f"{s.t:.4f}", s.state, f"{s.angle_deg:.2f}",
                    f"{s.force_n:.4f}", f"{s.force_left_n:.4f}",
                    f"{s.force_right_n:.4f}", f"{s.target_force_n:.4f}",
                    f"{s.asymmetry:.4f}",
                ])
        return path
