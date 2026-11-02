#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
忙等 PWM 舵机驱动 —— 用 CPU 空转换取微秒级定时精度。

原理
----
gpiozero 的软件 PWM 用 time.sleep() 定时，而 Linux 的 sleep 唤醒精度只有
几十微秒（受 timer slack 和调度影响）。对 20ms 周期、1.5ms 脉宽来说，
50µs 误差 ≈ 3° 的舵机抖动，肉眼可见。

本方案改成：
    · 脉冲的"高电平时间"用忙等（busy-wait）精确控制 → 精度约 1µs
    · 脉冲之间的"低电平时间"才用 sleep（周期略抖对舵机几乎无影响）

代价：每个脉冲忙等约 1.5ms / 20ms = 7.5% 的一个 CPU 核。

同时把 GIL 切换间隔调大，避免忙等期间被其他 Python 线程抢走。

用法::

    # 交互模式：手动输入脉宽，对比抖动
    sudo python3 apps/servo_busy_pwm.py --interactive

    # 对比模式：交替使用两种驱动，直观比较
    sudo python3 apps/servo_busy_pwm.py --compare

    # 直接驱动到某个脉宽并保持
    sudo python3 apps/servo_busy_pwm.py --pulse 1200 --hold 10
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

from grape_gripper.config import Config, default_config_path   # noqa: E402

try:
    import lgpio
except ImportError:
    print("❌ 缺少 lgpio： sudo apt install -y python3-lgpio")
    sys.exit(1)


class BusyPwmServo:
    """用忙等实现高精度软件 PWM 的舵机驱动。"""

    def __init__(self, pin: int, chip: int = 0,
                 period_us: float = 20000.0, pulse_us: float = 1500.0,
                 switch_interval: float = 0.05):
        self.pin = pin
        self.period_us = period_us
        self.pulse_us = pulse_us
        self._stop = threading.Event()
        self._lock = threading.Lock()

        # ★ 调大 GIL 切换间隔，避免忙等期间被其他线程抢走
        self._old_switch = sys.getswitchinterval()
        sys.setswitchinterval(switch_interval)

        self._h = lgpio.gpiochip_open(chip)
        lgpio.gpio_claim_output(self._h, pin)
        lgpio.gpio_write(self._h, pin, 0)

        # 定时精度统计
        self.jitter_us = []
        self._collect = 60      # 只保留最近 60 个样本

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def set_pulse(self, pulse_us: float) -> None:
        with self._lock:
            self.pulse_us = pulse_us

    def _loop(self) -> None:
        h, pin = self._h, self.pin
        period = self.period_us / 1e6
        while not self._stop.is_set():
            with self._lock:
                pulse = self.pulse_us / 1e6

            t0 = time.perf_counter()
            lgpio.gpio_write(h, pin, 1)

            # ★ 忙等：精确控制高电平时间
            target = t0 + pulse
            while time.perf_counter() < target:
                pass
            t_off = time.perf_counter()
            lgpio.gpio_write(h, pin, 0)

            err_us = (t_off - target) * 1e6
            if len(self.jitter_us) < self._collect:
                self.jitter_us.append(err_us)
            else:
                self.jitter_us.pop(0)
                self.jitter_us.append(err_us)

            # 低电平时间用 sleep（周期抖动对舵机影响很小）
            rest = period - (t_off - t0)
            if rest > 0:
                time.sleep(rest)

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        try:
            lgpio.gpio_write(self._h, self.pin, 0)
            lgpio.gpio_free(self._h, self.pin)
            lgpio.gpiochip_close(self._h)
        except Exception:  # noqa: BLE001
            pass
        try:
            sys.setswitchinterval(self._old_switch)
        except Exception:  # noqa: BLE001
            pass

    def jitter_stats(self) -> str:
        if len(self.jitter_us) < 5:
            return "样本不足"
        s = self.jitter_us
        return (f"平均 {statistics.mean(s):+.1f}us   "
                f"标准差 {statistics.pstdev(s):.1f}us   "
                f"最大 {max(abs(x) for x in s):.1f}us")


class GpiozeroServo:
    """对照组：gpiozero 的软件 PWM。"""

    def __init__(self, pin: int, pmin: float, pmax: float, period_us: float):
        from gpiozero import Device, Servo
        from gpiozero.pins.lgpio import LGPIOFactory

        Device.pin_factory = LGPIOFactory(chip=0)
        self.pmin, self.pmax = pmin, pmax
        self.servo = Servo(pin,
                           min_pulse_width=pmin / 1e6,
                           max_pulse_width=pmax / 1e6,
                           frame_width=period_us / 1e6,
                           initial_value=None)

    def set_pulse(self, pulse_us: float) -> None:
        p = max(self.pmin, min(self.pmax, pulse_us))
        ratio = (p - self.pmin) / (self.pmax - self.pmin)
        self.servo.value = ratio * 2.0 - 1.0

    def close(self) -> None:
        try:
            self.servo.detach()
        except Exception:  # noqa: BLE001
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description="忙等 PWM 舵机驱动")
    ap.add_argument("--config", default=None)
    ap.add_argument("--pin", type=int, default=None)
    ap.add_argument("--pulse", type=float, default=1200.0)
    ap.add_argument("--hold", type=float, default=10.0)
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--compare", action="store_true",
                    help="交替使用忙等 PWM 和 gpiozero，直观对比抖动")
    args = ap.parse_args()

    cfg = Config.load(args.config or default_config_path())
    if cfg.simulate:
        print("⚠️ config.yaml 里 hardware.simulate = true，请改成 false")
        return 1

    pin = args.pin if args.pin is not None else cfg.get_int("servo.gpio_pin", 49)
    pmin = cfg.get_float("servo.pulse_min_us", 650.0)
    pmax = cfg.get_float("servo.pulse_max_us", 1450.0)
    period = cfg.get_float("servo.period_us", 20000.0)

    print("=" * 70)
    print("忙等 PWM 舵机驱动")
    print("=" * 70)
    print(f"  GPIO{pin}   脉宽 {pmin:.0f}~{pmax:.0f}us   周期 {period:.0f}us")

    try:
        servo = BusyPwmServo(pin, chip=cfg.get_int("servo.gpio_chip", 0),
                             period_us=period, pulse_us=args.pulse)
    except Exception as exc:  # noqa: BLE001
        print(f"❌ 初始化失败：{exc}")
        return 1

    print("  ✅ 忙等 PWM 已启动")

    def clamp(p: float) -> float:
        return max(pmin, min(pmax, p))

    try:
        # ---------------- 对比模式 ----------------
        if args.compare:
            print()
            print("对比模式：交替使用两种驱动，观察抖动差别")
            print("  A = 忙等 PWM（新）")
            print("  B = gpiozero 软件 PWM（旧）")
            print("每个方式保持 6 秒，请注意听声音 / 看抖动")
            print()
            print("⚠️ 两种驱动不能同时占用同一个引脚，所以会依次建立和释放")
            print()
            input("按回车开始...")

            # 先释放主流程建好的忙等 PWM
            servo.close()
            time.sleep(0.5)

            busy_errors = []
            for rnd in range(2):
                # ---- A：忙等 PWM ----
                print(f"\n【第 {rnd+1} 轮】A 忙等 PWM  → {args.pulse:.0f}us")
                b = BusyPwmServo(pin, chip=cfg.get_int("servo.gpio_chip", 0),
                                 period_us=period,
                                 pulse_us=clamp(args.pulse))
                for i in range(6, 0, -1):
                    print(f"    {i}s   [定时误差 {b.jitter_stats()}]")
                    time.sleep(1.0)
                busy_errors.append(b.jitter_stats())
                b.close()
                time.sleep(0.6)

                # ---- B：gpiozero ----
                print(f"\n【第 {rnd+1} 轮】B gpiozero  → {args.pulse:.0f}us")
                try:
                    g = GpiozeroServo(pin, pmin, pmax, period)
                except Exception as exc:  # noqa: BLE001
                    print(f"    ❌ 无法建立 gpiozero 驱动：{exc}")
                    break
                g.set_pulse(clamp(args.pulse))
                for i in range(6, 0, -1):
                    print(f"    {i}s")
                    time.sleep(1.0)
                g.close()
                time.sleep(0.6)

            print()
            print("=" * 70)
            print("对比结束")
            print("=" * 70)
            for e in busy_errors:
                print(f"  忙等 PWM 定时误差：{e}")
            print()
            print("A 明显比 B 稳 → 用忙等 PWM；否则需要 PCA9685 硬件 PWM。")
            return 0

        # ---------------- 交互模式 ----------------
        if args.interactive:
            print()
            print(f"交互模式（当前 {args.pulse:.0f}us）。输入脉宽，q 退出")
            servo.set_pulse(clamp(args.pulse))
            cur = args.pulse
            while True:
                try:
                    line = input(f"  脉宽 [{pmin:.0f}~{pmax:.0f}] "
                                 f"（当前 {cur:.0f}）: ").strip()
                except EOFError:
                    break
                if line.lower() in ("q", "quit", "exit"):
                    break
                if line.lower() == "j":
                    print(f"    定时误差：{servo.jitter_stats()}")
                    continue
                if not line:
                    continue
                try:
                    cur = float(line)
                except ValueError:
                    print("    ❌ 请输入数字（或 j 看统计）")
                    continue
                actual = clamp(cur)
                servo.set_pulse(actual)
                print(f"    → {actual:.0f}us   定时误差：{servo.jitter_stats()}")
            return 0

        # ---------------- 保持模式 ----------------
        servo.set_pulse(clamp(args.pulse))
        print(f"\n  → {clamp(args.pulse):.0f}us，保持 {args.hold:.0f} 秒")
        print("     请观察抖动，并与 gpiozero 版本对比")
        t0 = time.time()
        while time.time() - t0 < args.hold:
            left = args.hold - (time.time() - t0)
            print(f"     {left:4.1f}s   定时误差：{servo.jitter_stats()}",
                  end="\r")
            time.sleep(0.5)
        print()
        return 0

    except KeyboardInterrupt:
        print("\n中断")
        return 0
    finally:
        servo.close()


if __name__ == "__main__":
    sys.exit(main())