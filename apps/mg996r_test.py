#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MG996R 舵机测试 —— 交互式、逐步试探，防止舵机失控。

优先使用 gpiozero + lgpio（官方 Bianbu 文档推荐的方式）。
如果 gpiozero 不可用，自动退回 sysfs PWM。

用法::

    # 先看板子上有哪些 GPIO 可用（不接舵机也能跑）
    python3 apps/mg996r_test.py --list

    # 打印引脚图（等价于 pinout 命令）
    python3 apps/mg996r_test.py --pinout

    # ★ 不知道信号线接哪个脚？自动逐个试，盯住舵机看哪个脚让它动
    python3 apps/mg996r_test.py --probe

    # 交互式测试（已知引脚号后）
    python3 apps/mg996r_test.py --pin 70

    # 只做一次扫描：从最小到最大脉宽走一遍，观察舵机行程
    python3 apps/mg996r_test.py --pin 70 --sweep

    # 直接转到某个角度（度，0~180）
    python3 apps/mg996r_test.py --pin 70 --angle 90

    # 直接指定脉宽微秒（更底层，用来找机械极限）
    python3 apps/mg996r_test.py --pin 70 --pulse 1500

⚠️ MG996R 安全须知
------------------
1. **必须独立 5V 供电（≥3A）**，绝不能从板子 5V 引脚取电 —— 堵转 2.5A 会把板子拉复位。
2. **舵机 GND 必须和板子 GND 共地**，否则信号无效。
3. 第一次测试前，**把舵机摇臂拆下来**（空载），避免扫到东西撞坏。
4. 出现持续嗡嗡声、发烫，立刻断电 —— 那是顶到极限了。
5. 棕=GND，红=+5V，橙=信号。
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass


# ---------------------------------------------------------------------------
# MG996R 的脉宽参数
# ---------------------------------------------------------------------------
# MG996R 数据手册：0° → 500us，180° → 2500us，周期 20ms（50Hz）
# 但实际每个舵机都有偏差，而且便宜舵机的有效范围常常只有 600~2400us。
# 所以下面给的是"保守起点"，用 --sweep 试探真实范围，再把安全值写进 config.yaml。
PULSE_MIN_US = 600      # 保守下限（比手册的 500 更安全，减少顶死风险）
PULSE_MAX_US = 2400     # 保守上限
PERIOD_US = 20000       # 50Hz


def list_gpio_resources() -> None:
    """打印板子上可用的 GPIO / PWM 资源，不需要接舵机。"""
    print("=" * 74)
    print("板的 GPIO / PWM 资源")
    print("=" * 74)

    print("\n--- gpiochip ---")
    base = "/sys/class/gpio"
    if os.path.isdir(base):
        found = False
        for name in sorted(os.listdir(base)):
            if not name.startswith("gpiochip"):
                continue
            found = True
            path = os.path.join(base, name)
            def rd(f: str) -> str:
                try:
                    with open(os.path.join(path, f), "r", encoding="ascii") as fh:
                        return fh.read().strip()
                except OSError:
                    return "?"
            print(f"  {name}: base={rd('base')} ngpio={rd('ngpio')} "
                  f"label={rd('label')}")
        if not found:
            print("  （无 gpiochipN 目录，使用 libgpiod 接口）")
    else:
        print("  无 /sys/class/gpio")

    print("\n--- gpioinfo（含 pwm 字样的引脚就是可用的硬件 PWM）---")
    if shutil.which("gpioinfo"):
        try:
            out = subprocess.run(["gpioinfo"], capture_output=True, text=True,
                                 timeout=10).stdout
            lines = out.splitlines()
            for line in lines[:70]:
                print("  " + line)
            if len(lines) > 70:
                print(f"  ...（共 {len(lines)} 行，已截断）")
            pwm_lines = [ln for ln in lines if "pwm" in ln.lower()]
            if pwm_lines:
                print("\n  ★ 名字里带 PWM 的引脚：")
                for ln in pwm_lines:
                    print("    " + ln.strip())
        except (subprocess.SubprocessError, OSError) as exc:
            print(f"  执行 gpioinfo 失败：{exc}")
    else:
        print("  gpioinfo 未安装 → sudo apt install -y gpiod")

    print("\n--- pwmchip ---")
    pwm_root = "/sys/class/pwm"
    if os.path.isdir(pwm_root):
        chips = [d for d in sorted(os.listdir(pwm_root)) if d.startswith("pwmchip")]
        if not chips:
            print("  无 pwmchip")
        for chip in chips:
            path = os.path.join(pwm_root, chip)
            try:
                with open(os.path.join(path, "npwm"), encoding="ascii") as fh:
                    npwm = fh.read().strip()
            except OSError:
                npwm = "?"
            try:
                real = os.path.realpath(path)
                platform = os.path.basename(os.path.dirname(real))
            except OSError:
                platform = "?"
            print(f"  {chip}: npwm={npwm}  控制器={platform}")
    else:
        print("  无 /sys/class/pwm")

    print("\n--- 设备树里的 PWM 节点 ---")
    dt = "/proc/device-tree"
    if os.path.isdir(dt):
        nodes = [n for n in os.listdir(dt) if "pwm" in n.lower()]
        if not nodes:
            print("  没有 pwm 节点")
        for node in nodes:
            npath = os.path.join(dt, node)
            status = compat = "?"
            try:
                with open(os.path.join(npath, "status"), "rb") as fh:
                    status = fh.read().decode("ascii", "replace").strip("\x00")
            except OSError:
                pass
            try:
                with open(os.path.join(npath, "compatible"), "rb") as fh:
                    compat = fh.read().decode("ascii", "replace").replace("\x00", " ").strip()
            except OSError:
                pass
            print(f"  {node}: status={status}")
            if compat != "?":
                print(f"      compatible={compat}")
    else:
        print("  读不到 /proc/device-tree")

    print("\n--- 提示 ---")
    print("  想知道哪个物理引脚能输出 PWM，运行： pinout")
    print("  （需要 sudo apt install -y python3-gpiozero python3-lgpio）")
    print()


# ---------------------------------------------------------------------------
# 后端 1：gpiozero + lgpio（推荐）
# ---------------------------------------------------------------------------

class GpiozeroServo:
    """用 gpiozero.Servo 输出舵机信号。"""

    name = "gpiozero+lgpio"

    def __init__(self, pin: int, pulse_min_us: int, pulse_max_us: int,
                 period_us: int):
        try:
            from gpiozero.pins.lgpio import LGPIOFactory
            from gpiozero import Device, Servo
        except ImportError as exc:
            raise RuntimeError(
                "没有 gpiozero / lgpio。安装：\n"
                "  sudo apt install -y python3-gpiozero python3-lgpio"
            ) from exc

        try:
            Device.pin_factory = LGPIOFactory(chip=0)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"初始化 lgpio 失败：{exc}\n"
                "试试： sudo chmod a+rw /dev/gpiochip0"
            ) from exc

        self._servo_cls = Servo
        self.pin = pin
        self.pulse_min_us = pulse_min_us
        self.pulse_max_us = pulse_max_us
        self.period_us = period_us

        self.servo = Servo(
            pin,
            min_pulse_width=pulse_min_us / 1e6,
            max_pulse_width=pulse_max_us / 1e6,
            frame_width=period_us / 1e6,
            initial_value=None,     # 先不动，避免上电瞬间猛转
        )
        self._current: float | None = None

    def set_angle(self, angle_deg: float) -> None:
        """角度 0~180 → gpiozero 的 -1~1。"""
        ratio = max(0.0, min(1.0, angle_deg / 180.0))
        value = ratio * 2.0 - 1.0
        self.servo.value = value
        self._current = angle_deg

    def set_pulse_us(self, pulse_us: float) -> None:
        """直接指定脉宽（微秒），用来精确探边界。"""
        span = self.pulse_max_us - self.pulse_min_us
        ratio = (pulse_us - self.pulse_min_us) / span if span else 0.0
        ratio = max(0.0, min(1.0, ratio))
        self.servo.value = ratio * 2.0 - 1.0
        self._current = ratio * 180.0

    def detach(self) -> None:
        """停止输出，舵机卸力（不再嗡嗡响、不发热）。"""
        try:
            self.servo.detach()
        except Exception:  # noqa: BLE001
            pass

    def close(self) -> None:
        self.detach()
        try:
            self.servo.close()
        except Exception:  # noqa: BLE001
            pass

    def describe(self) -> str:
        return (f"gpiozero.Servo(GPIO{pin}) "
                f"pulse={self.pulse_min_us}-{self.pulse_max_us}us "
                f"period={self.period_us}us")


# ---------------------------------------------------------------------------
# 后端 2：sysfs PWM（备用）
# ---------------------------------------------------------------------------

class SysfsServo:
    """用 /sys/class/pwm 输出。注意：pwmchip 编号和 GPIO 编号是两套体系。"""

    name = "sysfs_pwm"

    def __init__(self, chip: int, channel: int, pulse_min_us: int,
                 pulse_max_us: int, period_us: int):
        self.chip = chip
        self.channel = channel
        self.pulse_min_us = pulse_min_us
        self.pulse_max_us = pulse_max_us
        self.period_us = period_us

        self._chip_dir = f"/sys/class/pwm/pwmchip{chip}"
        self._pwm_dir = f"{self._chip_dir}/pwm{channel}"
        if not os.path.isdir(self._chip_dir):
            raise RuntimeError(f"找不到 {self._chip_dir}")
        if not os.path.isdir(self._pwm_dir):
            self._write_raw(f"{self._chip_dir}/export", str(channel))
            for _ in range(100):
                if os.path.isdir(self._pwm_dir):
                    break
                time.sleep(0.01)
            else:
                raise RuntimeError(f"导出后仍未出现 {self._pwm_dir}")
        self._write_raw(f"{self._pwm_dir}/period", str(period_us * 1000))
        self._write_raw(f"{self._pwm_dir}/enable", "1")

    def _write_raw(self, path: str, value: str) -> None:
        try:
            with open(path, "w", encoding="ascii") as fh:
                fh.write(value)
        except PermissionError as exc:
            raise RuntimeError(
                f"写 {path} 权限不足。sudo 运行，或：\n"
                "  sudo chmod -R a+rw /sys/class/pwm"
            ) from exc
        except OSError as exc:
            raise RuntimeError(f"写 {path} = {value} 失败：{exc}") from exc

    def _duty(self, pulse_us: float) -> None:
        self._write_raw(f"{self._pwm_dir}/duty_cycle", str(int(pulse_us * 1000)))

    def set_angle(self, angle_deg: float) -> None:
        ratio = max(0.0, min(1.0, angle_deg / 180.0))
        self._duty(self.pulse_min_us + ratio * (self.pulse_max_us - self.pulse_min_us))

    def set_pulse_us(self, pulse_us: float) -> None:
        pulse = max(self.pulse_min_us, min(self.pulse_max_us, pulse_us))
        self._duty(pulse)

    def detach(self) -> None:
        try:
            self._write_raw(f"{self._pwm_dir}/enable", "0")
        except RuntimeError:
            pass

    def close(self) -> None:
        self.detach()

    def describe(self) -> str:
        return (f"sysfs pwmchip{self.chip}/pwm{self.channel} "
                f"pulse={self.pulse_min_us}-{self.pulse_max_us}us")


def make_servo(args) -> object:
    """按参数创建舵机后端。"""
    if args.backend in ("auto", "gpiozero"):
        try:
            servo = GpiozeroServo(args.pin, args.pulse_min, args.pulse_max,
                                  args.period)
            print(f"后端：{servo.describe()}")
            return servo
        except RuntimeError as exc:
            if args.backend == "gpiozero":
                raise
            print(f"gpiozero 后端不可用：{exc}")
            print("改用 sysfs PWM 后端…")

    if args.backend in ("auto", "sysfs"):
        servo = SysfsServo(args.chip, args.channel, args.pulse_min,
                           args.pulse_max, args.period)
        print(f"后端：{servo.describe()}")
        return servo

    raise RuntimeError(f"未知后端：{args.backend}")


def do_list(args) -> int:
    list_gpio_resources()
    return 0


def do_sweep(servo, args) -> int:
    """从最小脉宽扫到最大脉宽，观察实际行程。"""
    print("\n开始扫描：观察舵机从一端转到另一端的完整行程。")
    print("注意听有没有「顶死」的嗡嗡声 —— 听到就记下那个脉宽，那是机械极限。\n")
    steps = 17
    for i in range(steps):
        pulse = args.pulse_min + (args.pulse_max - args.pulse_min) * i / (steps - 1)
        print(f"  脉宽 {pulse:6.0f}us … ", end="", flush=True)
        servo.set_pulse_us(pulse)
        time.sleep(0.6)
        print("ok")
    print("\n扫描结束。把「开始顶死」和「到不了」的脉宽记下来。")
    print("安全范围建议取：下限 +100us，上限 -100us。")
    if args.hold:
        print(f"\n保持最后位置 {args.hold}s（观察是否抖动/发热），然后卸力")
        time.sleep(args.hold)
    servo.detach()
    return 0


def do_interactive(servo, args) -> int:
    print("\n交互模式。输入角度（0~180）或脉宽（加 'u' 后缀，如 1500u）。")
    print("命令：  q=退出并卸力   d=卸力   h=回中位  s=扫描\n")
    while True:
        try:
            raw = input(f"角度[{servo._current if getattr(servo, '_current', None) is not None else '--'}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw:
            continue
        low = raw.lower()
        if low in ("q", "quit", "exit"):
            break
        if low == "d":
            servo.detach()
            print("  已卸力（舵机不再受力，可以手动扳动）")
            continue
        if low == "h":
            servo.set_angle(90.0)
            print("  转到中位 90°")
            time.sleep(0.4)
            continue
        if low == "s":
            do_sweep(servo, args)
            continue
        try:
            if low.endswith("u"):
                pulse = float(low[:-1])
                servo.set_pulse_us(pulse)
                print(f"  脉宽 {pulse:.0f}us")
            else:
                angle = float(low)
                if not 0 <= angle <= 180:
                    print("  角度要在 0~180 之间")
                    continue
                servo.set_angle(angle)
                print(f"  角度 {angle:.1f}°")
            time.sleep(0.4)
        except ValueError:
            print("  输入看不懂。示例： 90   或   1500u   或   q")
    servo.detach()
    print("已退出，舵机卸力。")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MG996R 舵机测试（K1 MusePi Pro）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法::")[1].split("⚠️")[0] if "用法::" in __doc__ else "",
    )
    p.add_argument("--list", action="store_true",
                   help="只列出板的 GPIO/PWM 资源，不驱动舵机")
    p.add_argument("--pinout", action="store_true",
                   help="打印 gpiozero 的引脚图（等价于 pinout 命令）")
    p.add_argument("--probe", action="store_true",
                   help="★ 自动探测哪个 GPIO 能驱动舵机：逐个引脚发脉冲，观察舵机是否动")
    p.add_argument("--pins", default=None,
                   help="--probe 时指定候选 GPIO，逗号分隔，例如 70,71,72 或 all")
    p.add_argument("--yes", action="store_true",
                   help="跳过确认提示（脚本化时用）")
    p.add_argument("--pin", type=int, default=70,
                   help="GPIO 编号（gpiozero 后端用），默认 70")
    p.add_argument("--chip", type=int, default=0,
                   help="pwmchip 编号（sysfs 后端用），默认 0")
    p.add_argument("--channel", type=int, default=0,
                   help="pwm 通道号（sysfs 后端用），默认 0")
    p.add_argument("--backend", choices=["auto", "gpiozero", "sysfs"],
                   default="auto", help="驱动后端，默认 auto（优先 gpiozero）")
    p.add_argument("--pulse-min", type=int, default=PULSE_MIN_US,
                   help=f"最小脉宽 us，默认 {PULSE_MIN_US}")
    p.add_argument("--pulse-max", type=int, default=PULSE_MAX_US,
                   help=f"最大脉宽 us，默认 {PULSE_MAX_US}")
    p.add_argument("--period", type=int, default=PERIOD_US,
                   help=f"PWM 周期 us，默认 {PERIOD_US}（50Hz）")
    p.add_argument("--angle", type=float, default=None,
                   help="直接转到指定角度（度）后退出")
    p.add_argument("--pulse", type=float, default=None,
                   help="直接输出指定脉宽（微秒）后退出")
    p.add_argument("--sweep", action="store_true",
                   help="做一次全行程扫描后退出")
    p.add_argument("--hold", type=float, default=0.0,
                   help="动作后保持多少秒再卸力")
    return p.parse_args()


#: 探测模式的候选 GPIO。
#: 来源：MUSE Pi 的 pinout 显示 GPIO70~78 在 40Pin 上，
#: 其中 GPIO70 被官方文档用作 PWM 示例。MUSE Pi Pro 编号可能不同，
#: 所以这里给一个"很可能"的候选集合，用 --pins 可以覆盖。
DEFAULT_PROBE_PINS = [70, 71, 72, 73, 74, 75, 76, 77, 78,
                      47, 48, 49, 50, 51, 52, 91, 92]


def _gpio_lines_with_pwm() -> list:
    """从 gpioinfo 里找出名字含 pwm 的 GPIO 编号，作为优先探测对象。"""
    if not shutil.which("gpioinfo"):
        return []
    try:
        out = subprocess.run(["gpioinfo"], capture_output=True, text=True,
                             timeout=15).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    found = []
    import re
    for line in out.splitlines():
        if "pwm" not in line.lower():
            continue
        m = re.search(r"line\s+(\d+)", line)
        if m:
            found.append(int(m.group(1)))
    return found


def print_pinout() -> int:
    """打印 gpiozero 的引脚图。"""
    if not shutil.which("pinout"):
        print("pinout 命令不存在。安装： sudo apt install -y python3-gpiozero")
        return 1
    try:
        out = subprocess.run(["pinout"], capture_output=True, text=True, timeout=20)
        print(out.stdout)
        if out.stderr.strip():
            print(out.stderr, file=sys.stderr)
        return out.returncode
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"执行 pinout 失败：{exc}", file=sys.stderr)
        return 2


def do_probe(args) -> int:
    """
    逐个候选 GPIO 发一小段脉冲，让用户观察舵机是否抖动/转动。

    原理：舵机只认信号线。给正确的引脚发 50Hz 脉冲，舵机就会动；
    给错误的引脚发脉冲，什么都不会发生（最多把一个普通 GPIO 拉高拉低，无害）。

    为什么要这一步：MUSE Pi Pro 的 GPIO 编号与 MUSE Pi 可能不同，
    与其猜，不如花 1 分钟实测。
    """
    print("=" * 74)
    print("舵机信号引脚自动探测")
    print("=" * 74)
    print("""
这一步会依次给一批 GPIO 引脚发送舵机脉冲，每个引脚约 1.5 秒。

【请盯住舵机】：观察到舵机出现抖动、转动或发出轻微响声的那一刻，
立刻按 Ctrl+C 停下，终端最后打印的引脚号就是正确的信号脚。
""")
    if not args.yes:
        try:
            input("舵机电源已接通、可以开始吗？按 Enter 开始（Ctrl+C 取消）… ")
        except (KeyboardInterrupt, EOFError):
            print("\n已取消")
            return 1

    # 决定候选列表：优先用 gpioinfo 里名字带 pwm 的
    if args.pins and args.pins.strip().lower() != "all":
        try:
            pins = [int(x.strip()) for x in args.pins.split(",") if x.strip()]
        except ValueError:
            print("--pins 格式错误。示例： --pins 70,71,72", file=sys.stderr)
            return 1
    else:
        hinted = _gpio_lines_with_pwm()
        if hinted:
            print(f"gpioinfo 里名字含 pwm 的引脚（优先探测）：{hinted}\n")
            pins = hinted + [p for p in DEFAULT_PROBE_PINS if p not in hinted]
        else:
            pins = list(DEFAULT_PROBE_PINS)

    print(f"候选引脚（{len(pins)} 个）：{pins}\n")

    backend_ok = False
    try:
        from gpiozero.pins.lgpio import LGPIOFactory
        from gpiozero import Device, Servo
        Device.pin_factory = LGPIOFactory(chip=0)
        backend_ok = True
    except Exception as exc:  # noqa: BLE001
        print(f"❌ 无法初始化 gpiozero/lgpio：{exc}", file=sys.stderr)
        print("   先执行： sudo apt install -y python3-gpiozero python3-lgpio",
              file=sys.stderr)
        print("   以及：   sudo chmod a+rw /dev/gpiochip0", file=sys.stderr)
        return 2

    if not backend_ok:
        return 2

    last_tried: int | None = None
    try:
        for idx, pin in enumerate(pins, 1):
            last_tried = pin
            print(f"[{idx}/{len(pins)}] 正在试 GPIO{pin} … ", end="", flush=True)
            servo = None
            try:
                servo = Servo(pin,
                              min_pulse_width=args.pulse_min / 1e6,
                              max_pulse_width=args.pulse_max / 1e6,
                              frame_width=args.period / 1e6,
                              initial_value=None)
                # 先回一个位置，再转到另一个位置 —— 这样舵机动作明显，容易观察
                servo.value = -0.6
                time.sleep(0.7)
                servo.value = 0.6
                time.sleep(0.7)
                print("试过")
            except Exception as exc:  # noqa: BLE001
                print(f"失败（{type(exc).__name__}: {exc}）")
            finally:
                if servo is not None:
                    try:
                        servo.detach()
                        servo.close()
                    except Exception:  # noqa: BLE001
                        pass
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\n\n★ 已中断。")
        if last_tried is not None:
            print(f"最后一个试的引脚是 GPIO{last_tried}")
            print(f"如果舵机在那一刻动了，就用它：")
            print(f"    python3 apps/mg996r_test.py --pin {last_tried}")
        print("\n如果从头到尾舵机都没动，检查：")
        print("  1. 信号线（橙色）是否真的接到了板子的引脚上")
        print("  2. 舵机电源是否接通（独立 5V）")
        print("  3. 舵机 GND 与板子 GND 是否共地  ← 最常见的问题")
        print("  4. 舵机本身是否损坏（拿 Arduino 试一下）")
        return 130

    print("\n探测完毕，所有候选引脚都试过了。")
    print("如果舵机全程没动，问题多半不在引脚号，而在接线：")
    print("  1. 橙色信号线是否接对   2. 独立 5V 是否供电")
    print("  3. GND 是否共地 ★       4. 舵机是否完好")
    return 0


def main() -> int:
    args = parse_args()

    if args.list:
        return do_list(args)
    if args.pinout:
        return print_pinout()
    if args.probe:
        return do_probe(args)

    print("=" * 74)
    print("MG996R 舵机测试")
    print("=" * 74)
    print("""
⚠️  开始前请确认：
    1. 舵机已接**独立 5V 电源（≥3A）**，不是从板子 5V 取电
    2. 舵机 GND 与板子 GND **已共地**
    3. 摇臂**已拆下**（空载测试），周围没有障碍物
    4. 棕=GND  红=+5V  橙=信号
    5. 随时准备切断舵机电源
""")
    if not args.yes:
        try:
            input("按 Enter 继续，Ctrl+C 取消… ")
        except (KeyboardInterrupt, EOFError):
            print("\n已取消")
            return 1

    try:
        servo = make_servo(args)
    except RuntimeError as exc:
        print(f"\n❌ 初始化舵机失败：{exc}", file=sys.stderr)
        print("\n先运行这个看看板子有什么资源：", file=sys.stderr)
        print("  python3 apps/mg996r_test.py --list", file=sys.stderr)
        return 2

    try:
        if args.sweep:
            return do_sweep(servo, args)
        if args.pulse is not None:
            servo.set_pulse_us(args.pulse)
            print(f"输出脉宽 {args.pulse:.0f}us")
            time.sleep(max(0.5, args.hold))
            servo.detach()
            return 0
        if args.angle is not None:
            if not 0 <= args.angle <= 180:
                print("角度要在 0~180 之间", file=sys.stderr)
                return 1
            servo.set_angle(args.angle)
            print(f"转到 {args.angle:.1f}°")
            time.sleep(max(0.5, args.hold))
            servo.detach()
            return 0
        return do_interactive(servo, args)
    finally:
        servo.close()


if __name__ == "__main__":
    sys.exit(main())
