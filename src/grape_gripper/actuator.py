# -*- coding: utf-8 -*-
"""
夹爪舵机驱动。

三种驱动：
  - ``SysfsPwmServo``  : 原生 PWM（/sys/class/pwm），最稳，推荐
  - ``SoftPwmServo``   : GPIO 软件 PWM，仅应急/调试，会抖动
  - ``SimulatedServo`` : 模拟，可在无硬件环境跑通全部逻辑

统一接口：``set_angle(deg)`` / ``set_normalized(0~1)`` /
``update(now)``（按速率限幅平滑推进）/ ``disable()``（卸力）。
"""

from __future__ import annotations

import abc
import math
import os
import time
from typing import Optional


class ActuatorError(Exception):
    """舵机驱动出错。"""


def _pulse_from_angle(angle_deg: float, pulse_min: float, pulse_max: float,
                      angle_min: float = 0.0, angle_max: float = 180.0) -> float:
    """角度 → 脉宽（微秒），线性映射并做钳位。"""
    span = max(1e-6, angle_max - angle_min)
    ratio = (angle_deg - angle_min) / span
    ratio = max(0.0, min(1.0, ratio))
    return pulse_min + ratio * (pulse_max - pulse_min)


class ServoBase(abc.ABC):
    """舵机抽象基类。"""

    def __init__(self, angle_min: float = 0.0, angle_max: float = 180.0,
                 max_rate_deg_per_s: float = 200.0):
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.max_rate_deg_per_s = max(1e-3, max_rate_deg_per_s)

        self._angle = angle_min
        self._target = angle_min
        self._last_update = time.monotonic()
        self._enabled = False

    # ---- 子类必须实现 ----

    @abc.abstractmethod
    def _apply_angle(self, angle_deg: float) -> None:
        """把角度真正写到硬件上。"""

    # ---- 通用接口 ----

    @property
    def angle(self) -> float:
        return self._angle

    @property
    def target_angle(self) -> float:
        return self._target

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def at_target(self) -> bool:
        return abs(self._angle - self._target) < 0.05

    def set_angle(self, angle_deg: float) -> None:
        """设定目标角度（不会立即跳到，由 update() 按速率限幅推进）。"""
        self._target = max(self.angle_min, min(self.angle_max, float(angle_deg)))

    def set_normalized(self, value: float) -> None:
        """0 = angle_min（张开），1 = angle_max（闭合方向）。"""
        value = max(0.0, min(1.0, value))
        self.set_angle(self.angle_min + value * (self.angle_max - self.angle_min))

    @property
    def normalized(self) -> float:
        span = max(1e-6, self.angle_max - self.angle_min)
        return (self._angle - self.angle_min) / span

    def update(self, now: Optional[float] = None) -> float:
        """
        按 ``max_rate_deg_per_s`` 向目标推进一小步。

        返回当前角度。控制循环每个周期都应调用它。
        """
        now = time.monotonic() if now is None else now
        dt = now - self._last_update
        self._last_update = now
        if dt <= 0:
            return self._angle

        delta = self._target - self._angle
        if abs(delta) < 1e-6:
            self._angle = self._target
            if not self._enabled:
                # 首次调用时即使不需要移动，也要把值写到硬件上，
                # 否则舵机永远处于"未启动"状态（PWM 不会输出）。
                self._enabled = True
                self._apply_angle(self._angle)
            return self._angle

        max_step = self.max_rate_deg_per_s * dt
        step = max(-max_step, min(max_step, delta))
        self._angle += step
        self._enabled = True
        self._apply_angle(self._angle)
        return self._angle

    def move_to(self, angle_deg: float, timeout_s: float = 2.0,
                poll_hz: float = 100.0, on_tick=None) -> bool:
        """
        阻塞式移动到目标角度。返回是否在超时前到达。

        ``on_tick`` 是每个周期的回调，用来喂安全看门狗 —— 阻塞动作同样会让
        主循环"停止心跳"，不喂的话看门狗会误判成程序卡死并卸力。
        """
        self.set_angle(angle_deg)
        deadline = time.monotonic() + timeout_s
        period = 1.0 / max(1.0, poll_hz)
        while time.monotonic() < deadline:
            self.update()
            if on_tick is not None:
                on_tick()
            if self.at_target:
                return True
            time.sleep(period)
        return self.at_target

    def disable(self) -> None:
        """
        停止输出脉宽 → 舵机卸力（可以手动扳动）。

        安全用途：夹取异常时卸力，比"保持夹紧"更安全 —— 葡萄不会一直被压着。
        """
        self._enabled = False
        self._disable_hardware()

    def _disable_hardware(self) -> None:
        pass

    def describe(self) -> str:
        return (f"{type(self).__name__}(angle={self._angle:.1f}, "
                f"target={self._target:.1f}, rate={self.max_rate_deg_per_s}deg/s)")


# ---------------------------------------------------------------------------
# 模拟
# ---------------------------------------------------------------------------

class SimulatedServo(ServoBase):
    """
    模拟舵机。与 SimulatedAdc 联动：把当前闭合程度告知 ADC，
    这样在电脑上跑就能看到真实的"接触→力上升→刹车"曲线。
    """

    def __init__(self, adc=None, **kwargs):
        super().__init__(**kwargs)
        self.adc = adc
        self.pulse_us: float = 0.0
        self._pulse_min = 500.0
        self._pulse_max = 2500.0
        self._push_closure_to_adc()

    def set_pulse_range(self, pulse_min: float, pulse_max: float) -> None:
        self._pulse_min, self._pulse_max = pulse_min, pulse_max

    def _apply_angle(self, angle_deg: float) -> None:
        self.pulse_us = _pulse_from_angle(
            angle_deg, self._pulse_min, self._pulse_max, self.angle_min, self.angle_max
        )
        self._push_closure_to_adc()

    def _push_closure_to_adc(self) -> None:
        if self.adc is not None and hasattr(self.adc, "set_closure"):
            self.adc.set_closure(self.normalized)

    def _disable_hardware(self) -> None:
        self.pulse_us = 0.0

    def update(self, now: Optional[float] = None) -> float:
        angle = super().update(now)
        self._push_closure_to_adc()
        return angle


# ---------------------------------------------------------------------------
# sysfs PWM
# ---------------------------------------------------------------------------

class SysfsPwmServo(ServoBase):
    """
    Linux sysfs PWM 舵机驱动。

    sysfs 布局（每个 pwmchip 是一个目录）：
        /sys/class/pwm/pwmchipN/
            export              # echo 通道号 > export 导出
            unexport
            npwm
            pwmX/
                period          # 周期，纳秒。50Hz → 20000000
                duty_cycle      # 高电平时间，纳秒。脉宽(us)*1000
                enable          # 1 使能
                polarity

    常见坑：
      1. **必须先 pinmux**：K1 上某物理引脚要复用成 PWM 功能，pwmchip 才会出现。
         具体是哪个引脚、用什么工具配置，见 docs/pinout.md。
      2. **period 必须先于 duty_cycle 设置**，且 duty_cycle 不能大于 period。
         顺序错了会报 EINVAL。
      3. **权限**：/sys/class/pwm/*/ 默认 root only。两个办法：
         - 用 root 跑（简单，比赛现场够用）
         - 加 udev 规则给 pwm 组写权限（推荐，但要多一步配置）
    """

    def __init__(self, chip: int = 0, channel: int = 0,
                 period_us: float = 20000.0,
                 pulse_min_us: float = 500.0, pulse_max_us: float = 2500.0,
                 sysfs_root: str = "/sys/class/pwm",
                 angle_min: float = 0.0, angle_max: float = 180.0,
                 max_rate_deg_per_s: float = 200.0,
                 auto_export: bool = True):
        super().__init__(angle_min=angle_min, angle_max=angle_max,
                         max_rate_deg_per_s=max_rate_deg_per_s)
        if pulse_max_us >= period_us:
            raise ActuatorError(
                f"pulse_max_us({pulse_max_us}) 必须小于 period_us({period_us})"
            )

        self.chip = chip
        self.channel = channel
        self.period_us = period_us
        self.pulse_min_us = pulse_min_us
        self.pulse_max_us = pulse_max_us

        self._chip_dir = os.path.join(sysfs_root, f"pwmchip{chip}")
        self._pwm_dir = os.path.join(self._chip_dir, f"pwm{channel}")

        if not os.path.isdir(self._chip_dir):
            available = []
            if os.path.isdir(sysfs_root):
                available = [d for d in os.listdir(sysfs_root) if d.startswith("pwmchip")]
            raise ActuatorError(
                f"找不到 {self._chip_dir}。当前可用：{available or '（无）'}。\n"
                "K1 上通常需要先做 pinmux 把引脚复用为 PWM，pwmchip 才会出现；"
                "见 docs/pinout.md。"
            )

        if not os.path.isdir(self._pwm_dir):
            if not auto_export:
                raise ActuatorError(f"{self._pwm_dir} 不存在且未开启 auto_export")
            self._export()

        self._write("period", int(period_us * 1000))
        self._write("duty_cycle", int(self._pulse_us_for(self.angle_min) * 1000))
        self._write("enable", 1)
        self._enabled = True

    # ---- 内部 ----

    def _pulse_us_for(self, angle_deg: float) -> float:
        return _pulse_from_angle(angle_deg, self.pulse_min_us, self.pulse_max_us,
                                 self.angle_min, self.angle_max)

    def _export(self) -> None:
        export_path = os.path.join(self._chip_dir, "export")
        try:
            with open(export_path, "w", encoding="ascii") as fh:
                fh.write(str(self.channel))
        except PermissionError as exc:
            raise ActuatorError(
                f"写 {export_path} 权限不足。用 sudo 运行，或配置 udev 规则"
            ) from exc
        except OSError as exc:
            # EBUSY / EINVAL 通常表示"已经导出过了"，忽略
            if exc.errno not in (16, 22):
                raise ActuatorError(f"导出 PWM 通道失败：{exc}") from exc

        # 等 sysfs 目录出现（内核异步创建）
        for _ in range(100):
            if os.path.isdir(self._pwm_dir):
                return
            time.sleep(0.01)
        raise ActuatorError(f"导出后仍未出现 {self._pwm_dir}")

    def _write(self, name: str, value: int) -> None:
        path = os.path.join(self._pwm_dir, name)
        try:
            with open(path, "w", encoding="ascii") as fh:
                fh.write(str(int(value)))
        except PermissionError as exc:
            raise ActuatorError(
                f"写 {path} 权限不足。用 sudo 运行，或配置 udev 规则：\n"
                f"  SUBSYSTEM==\"pwm\", ACTION==\"add\", RUN+=\"/bin/chmod -R a+rw /sys/class/pwm\""
            ) from exc
        except OSError as exc:
            raise ActuatorError(f"写 {path} = {value} 失败：{exc}") from exc

    def _apply_angle(self, angle_deg: float) -> None:
        pulse_us = self._pulse_us_for(angle_deg)
        # 只写 duty_cycle，不动 period/enable —— 这是最快也最安全的更新方式
        self._write("duty_cycle", int(pulse_us * 1000))

    def _disable_hardware(self) -> None:
        try:
            self._write("enable", 0)
        except ActuatorError:
            pass

    def describe(self) -> str:
        return (f"SysfsPwmServo(chip={self.chip} ch={self.channel} "
                f"period={self.period_us}us pulse={self.pulse_min_us}-{self.pulse_max_us}us)")


# ---------------------------------------------------------------------------
# 软件 PWM（应急方案）
# ---------------------------------------------------------------------------

class SoftPwmServo(ServoBase):
    """
    用 GPIO 位翻转产生 PWM。**不推荐**，仅在没有可用 PWM 通道时应急。

    缺点很明确：
      - Python 里做位翻转，抖动大（jitter 可达数百微秒）→ 舵机嗡嗡响、发热
      - 会占满一个 CPU 核
      - 多路舵机时互相干扰

    正确做法是等 K1 的硬件 PWM 打通，或者用一个便宜的 MCU（如 ESP32）
    专门负责舵机 + 高频力采样，通过串口和 K1 通信。
    """

    def __init__(self, gpio_pin: int = 0, period_us: float = 20000.0,
                 pulse_min_us: float = 500.0, pulse_max_us: float = 2500.0,
                 gpio_root: str = "/sys/class/gpio",
                 **kwargs):
        super().__init__(**kwargs)
        if os.environ.get("GRAPE_ALLOW_SOFT_PWM") != "1":
            raise ActuatorError(
                "SoftPwmServo 需要显式授权：设置环境变量 GRAPE_ALLOW_SOFT_PWM=1。\n"
                "它会占用 CPU 并产生抖动，优先打通硬件 PWM（见 docs/pinout.md）。"
            )
        self.gpio_pin = gpio_pin
        self.period_us = period_us
        self.pulse_min_us = pulse_min_us
        self.pulse_max_us = pulse_max_us
        self._gpio_root = gpio_root
        self._gpio_dir = os.path.join(gpio_root, f"gpio{gpio_pin}")
        self._setup_gpio()

    def _setup_gpio(self) -> None:
        export = os.path.join(self._gpio_root, "export")
        if not os.path.isdir(self._gpio_dir):
            try:
                with open(export, "w", encoding="ascii") as fh:
                    fh.write(str(self.gpio_pin))
            except (OSError, PermissionError) as exc:
                raise ActuatorError(
                    f"导出 GPIO{self.gpio_pin} 失败：{exc}。"
                    "考虑改用 libgpiod（gpioset）或硬件 PWM。"
                ) from exc
        self._write_gpio("direction", "out")
        self._write_gpio("value", 0)

    def _write_gpio(self, name: str, value) -> None:
        path = os.path.join(self._gpio_dir, name)
        try:
            with open(path, "w", encoding="ascii") as fh:
                fh.write(str(value))
        except (OSError, PermissionError) as exc:
            raise ActuatorError(f"写 {path} 失败：{exc}") from exc

    def _apply_angle(self, angle_deg: float) -> None:
        # 注意：这里只是把目标脉宽记下来，实际波形由 keep_running 循环生成。
        # 在本类的用法里，请把 update() 放在一个高频循环（>=200Hz）里跑。
        pulse_us = _pulse_from_angle(angle_deg, self.pulse_min_us, self.pulse_max_us,
                                     self.angle_min, self.angle_max)
        self._write_gpio("value", 1)
        time.sleep(pulse_us / 1e6)
        self._write_gpio("value", 0)

    def _disable_hardware(self) -> None:
        try:
            self._write_gpio("value", 0)
        except ActuatorError:
            pass

    def describe(self) -> str:
        return f"SoftPwmServo(gpio={self.gpio_pin} period={self.period_us}us)"


class LgpioServo(ServoBase):
    """
    舵机驱动 —— **K1 上实测可用的方案**。

    实现方式：``gpiozero.Servo`` + ``LGPIOFactory`` 引脚工厂。

    为什么是组合而不是单一方案（实测结论）
    --------------------------------------
    K1 上没有可用的硬件 PWM 通道（两个 pwmchip 各只有 1 路，且未引到 40Pin），
    所以只能用软件 PWM。实测对比：

    ==========================  ==========
    方式                        结果
    ==========================  ==========
    ``lgpio.tx_pwm()``           ❌ 舵机无反应
    ``gpiozero.Servo`` + lgpio   ✅ **可用**
    ==========================  ==========

    因此这里沿用 ``apps/mg996r_test.py`` 中已验证的写法。

    ⚠️ **必须使用设备树里默认配置为 GPIO 的引脚。**

    K1 的 40Pin 上，只有 **GPIO_49（物理引脚 22）** 同时满足
    "默认 MUX_MODE0（GPIO 功能）" 和 "3.3V 电平（PAD_3V_DS4）"。
    标着 GPIO 的引脚（如引脚 32 的 GPIO_34）在设备树里并未配置为 GPIO，
    写 GPIO 寄存器不会有任何输出。详见 ``docs/技术文档.md`` §6.4。

    局限性
    ------
    软件 PWM 的定时精度受 Linux 调度影响（抖动可达数十微秒），
    舵机可能有轻微抖动/嗡鸣。若影响力闭环稳定性，可改用 PCA9685 硬件 PWM 模块。
    """

    def __init__(self, gpio_pin: int = 49, period_us: float = 20000.0,
                 pulse_min_us: float = 650.0, pulse_max_us: float = 1450.0,
                 chip: int = 0, **kwargs):
        super().__init__(**kwargs)

        try:
            from gpiozero import Device, Servo
            from gpiozero.pins.lgpio import LGPIOFactory
        except ImportError as exc:
            raise ActuatorError(
                "需要 gpiozero 和 lgpio：\n"
                "  sudo apt install -y python3-gpiozero python3-lgpio"
            ) from exc

        self.gpio_pin = gpio_pin
        self.chip = chip
        self.period_us = period_us
        self.pulse_min_us = pulse_min_us
        self.pulse_max_us = pulse_max_us

        try:
            Device.pin_factory = LGPIOFactory(chip=chip)
        except Exception as exc:  # noqa: BLE001
            raise ActuatorError(
                f"初始化 lgpio 引脚工厂失败：{exc}\n"
                "  试试： sudo chmod a+rw /dev/gpiochip0"
            ) from exc

        try:
            # initial_value=None：上电不动，避免瞬间猛转
            self._servo = Servo(
                gpio_pin,
                min_pulse_width=pulse_min_us / 1e6,
                max_pulse_width=pulse_max_us / 1e6,
                frame_width=period_us / 1e6,
                initial_value=None,
            )
        except Exception as exc:  # noqa: BLE001
            raise ActuatorError(
                f"申请 GPIO{gpio_pin} 失败：{exc}\n"
                "  确认该引脚在设备树里被配置为 GPIO（见类文档）"
            ) from exc

    def _apply_angle(self, angle_deg: float) -> None:
        pulse_us = _pulse_from_angle(angle_deg, self.pulse_min_us,
                                     self.pulse_max_us,
                                     self.angle_min, self.angle_max)
        span = self.pulse_max_us - self.pulse_min_us
        ratio = (pulse_us - self.pulse_min_us) / span if span else 0.0
        ratio = max(0.0, min(1.0, ratio))
        try:
            # gpiozero.Servo 的 value: -1 → 最小脉宽, +1 → 最大脉宽
            self._servo.value = ratio * 2.0 - 1.0
        except Exception as exc:  # noqa: BLE001
            raise ActuatorError(f"设置舵机角度失败：{exc}") from exc

    def _disable_hardware(self) -> None:
        try:
            self._servo.detach()
        except Exception:  # noqa: BLE001
            pass

    def close(self) -> None:
        """停止 PWM 并释放 GPIO。"""
        self.disable()
        try:
            self._servo.close()
        except Exception:  # noqa: BLE001
            pass

    def describe(self) -> str:
        return (f"LgpioServo(GPIO{self.gpio_pin} chip{self.chip} "
                f"{1e6/self.period_us:.0f}Hz "
                f"pulse={self.pulse_min_us:.0f}-{self.pulse_max_us:.0f}us)")


class BusyPwmServo(ServoBase):
    """
    舵机驱动 —— **忙等 PWM，抖动最小的软件方案（K1 实测推荐）**。

    为什么需要它
    ------------
    :class:`LgpioServo`（gpiozero 软件 PWM）用 ``time.sleep()`` 定时，
    而 Linux 的 sleep 唤醒精度只有几十微秒 —— 摊到 20ms 周期、1.5ms 脉宽上
    就是约 3° 的可见抖动。

    实测排查过程：

    ========================================  ==========
    尝试                                      结果
    ========================================  ==========
    ``sudo chrt -f 50``（实时调度优先级）      ❌ 无明显改善
    ``PR_SET_TIMERSLACK = 1ns``               ❌ 无明显改善
    CPU governor 设为 performance             ❌ 无明显改善
    **忙等 PWM（本类）**                       ✅ **明显更稳**
    ========================================  ==========

    原理
    ----
    · 脉冲的**高电平时间**用忙等（busy-wait）精确控制 → 误差约 1~3µs
    · 脉冲之间的**低电平时间**才用 sleep（周期略抖对舵机几乎无影响）
    · 把 GIL 切换间隔调大，避免忙等期间被其他 Python 线程抢占

    代价
    ----
    每个脉冲忙等约 1.5ms / 20ms = **约 7.5% 的一个 CPU 核**。
    K1 有 8 核，可以忽略。

    硬件 PWM 说明
    -------------
    K1 的 ``/sys/class/pwm/pwmchip0`` 和 ``pwmchip1`` 各只有 1 路，
    且设备树里 pwm14 已被屏幕背光占用，40Pin 上没有可用的硬件 PWM
    （``pinmux-functions`` 在 ``pinctrl-single`` 驱动下不存在，无法复用）。
    因此软件方案是本板的最优解；若将来需要多路高精度 PWM，可加 PCA9685。
    """

    def __init__(self, gpio_pin: int = 49, period_us: float = 20000.0,
                 pulse_min_us: float = 650.0, pulse_max_us: float = 1450.0,
                 chip: int = 0, switch_interval: float = 0.05, **kwargs):
        super().__init__(**kwargs)

        import sys
        import threading

        try:
            import lgpio
        except ImportError as exc:
            raise ActuatorError(
                "需要 lgpio： sudo apt install -y python3-lgpio"
            ) from exc

        self.gpio_pin = gpio_pin
        self.chip = chip
        self.period_us = period_us
        self.pulse_min_us = pulse_min_us
        self.pulse_max_us = pulse_max_us

        self._lgpio = lgpio
        self._pulse_us = pulse_min_us
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._jitter = []          # 最近若干次高电平定时误差（微秒）

        self._old_switch = sys.getswitchinterval()
        sys.setswitchinterval(switch_interval)

        try:
            self._h = lgpio.gpiochip_open(chip)
            lgpio.gpio_claim_output(self._h, gpio_pin)
            lgpio.gpio_write(self._h, gpio_pin, 0)
        except Exception as exc:  # noqa: BLE001
            sys.setswitchinterval(self._old_switch)
            raise ActuatorError(
                f"申请 GPIO{gpio_pin} 失败：{exc}\n"
                "  1. 确认该引脚在设备树里被配置为 GPIO（见 LgpioServo 文档）\n"
                "  2. 试试： sudo chmod a+rw /dev/gpiochip0"
            ) from exc

        self._thread = threading.Thread(target=self._loop,
                                        name="busy-pwm", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    # PWM 主循环
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        import time

        h, pin, lg = self._h, self.gpio_pin, self._lgpio
        period = self.period_us / 1e6
        write = lg.gpio_write
        clock = time.perf_counter

        while not self._stop.is_set():
            with self._lock:
                pulse = self._pulse_us / 1e6

            t0 = clock()
            write(h, pin, 1)

            # ★ 忙等：精确控制高电平时间
            target = t0 + pulse
            while clock() < target:
                pass
            t_off = clock()
            write(h, pin, 0)

            err = (t_off - target) * 1e6
            if len(self._jitter) >= 60:
                self._jitter.pop(0)
            self._jitter.append(err)

            # 低电平时间用 sleep（周期抖动对舵机影响很小）
            rest = period - (t_off - t0)
            if rest > 0:
                time.sleep(rest)

    # ------------------------------------------------------------------
    # ServoBase 接口
    # ------------------------------------------------------------------

    def _apply_angle(self, angle_deg: float) -> None:
        pulse = _pulse_from_angle(angle_deg, self.pulse_min_us,
                                  self.pulse_max_us,
                                  self.angle_min, self.angle_max)
        pulse = max(self.pulse_min_us, min(self.pulse_max_us, pulse))
        with self._lock:
            self._pulse_us = pulse

    def set_pulse_us(self, pulse_us: float) -> None:
        """
        直接设置脉宽（微秒），绕过角度映射。

        用于调试和标定：想直接给某个脉宽看机械响应时用它。
        会自动限制在 ``pulse_min_us`` ~ ``pulse_max_us`` 之间。
        """
        pulse = max(self.pulse_min_us, min(self.pulse_max_us, pulse_us))
        with self._lock:
            self._pulse_us = pulse

    def _disable_hardware(self) -> None:
        """卸力：停止输出脉冲（舵机失去保持力，不会堵转发热）。"""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def close(self) -> None:
        import sys

        self._disable_hardware()
        try:
            self._lgpio.gpio_write(self._h, self.gpio_pin, 0)
            self._lgpio.gpio_free(self._h, self.gpio_pin)
            self._lgpio.gpiochip_close(self._h)
        except Exception:  # noqa: BLE001
            pass
        try:
            sys.setswitchinterval(self._old_switch)
        except Exception:  # noqa: BLE001
            pass

    def jitter_stats(self) -> str:
        """高电平定时误差统计（用于评估抖动）。"""
        import statistics
        if len(self._jitter) < 5:
            return "样本不足"
        s = self._jitter
        return (f"平均 {statistics.mean(s):+.1f}us  "
                f"标准差 {statistics.pstdev(s):.1f}us  "
                f"最大 {max(abs(x) for x in s):.1f}us")

    def describe(self) -> str:
        return (f"BusyPwmServo(GPIO{self.gpio_pin} chip{self.chip} "
                f"{1e6/self.period_us:.0f}Hz "
                f"pulse={self.pulse_min_us:.0f}-{self.pulse_max_us:.0f}us "
                f"忙等定时 {self.jitter_stats()})")


def create_servo(cfg, adc=None):
    """根据配置创建舵机实例。"""
    if cfg.simulate:
        servo = SimulatedServo(
            adc=adc,
            angle_min=0.0,
            angle_max=180.0,
            max_rate_deg_per_s=cfg.get_float("servo.max_rate_deg_per_s", 200.0),
        )
        servo.set_pulse_range(
            cfg.get_float("servo.pulse_min_us", 500),
            cfg.get_float("servo.pulse_max_us", 2500),
        )
        servo.set_angle(cfg.get_float("servo.angle_home_deg", 10.0))
        return servo

    driver = cfg.get_str("servo.driver", "sysfs_pwm")
    common = dict(
        period_us=cfg.get_float("servo.period_us", 20000.0),
        pulse_min_us=cfg.get_float("servo.pulse_min_us", 500.0),
        pulse_max_us=cfg.get_float("servo.pulse_max_us", 2500.0),
        max_rate_deg_per_s=cfg.get_float("servo.max_rate_deg_per_s", 200.0),
    )

    if driver == "sysfs_pwm":
        return SysfsPwmServo(
            chip=cfg.get_int("servo.pwm_chip", 0),
            channel=cfg.get_int("servo.pwm_channel", 0),
            **common,
        )
    if driver == "gpio_soft":
        return SoftPwmServo(gpio_pin=cfg.get_int("servo.gpio_pin", 0), **common)
    if driver == "lgpio_pwm":
        return LgpioServo(
            gpio_pin=cfg.get_int("servo.gpio_pin", 49),
            chip=cfg.get_int("servo.gpio_chip", 0),
            **common,
        )
    if driver == "busy_pwm":
        return BusyPwmServo(
            gpio_pin=cfg.get_int("servo.gpio_pin", 49),
            chip=cfg.get_int("servo.gpio_chip", 0),
            **common,
        )

    raise ActuatorError(
        f"未知的 servo.driver：{driver!r}"
        "（可选 sysfs_pwm / gpio_soft / lgpio_pwm / busy_pwm / serial_bus）"
    )
