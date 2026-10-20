#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
夹取测试 —— 单次或多次执行"张开 → 接近 → 力闭环 → 保持 → 释放"。

这是你调参时用得最多的工具。

用法::

    # 纯模拟（不需要任何硬件），先把流程和日志看懂
    python3 apps/grip_test.py --sim

    # 真实硬件，夹一次
    python3 apps/grip_test.py

    # 连续 20 次，统计成功率（调参用）
    python3 apps/grip_test.py -n 20

    # 每次先重新去皮（推荐：环境温度变化后零点会漂）
    python3 apps/grip_test.py --tare

    # 临时覆盖参数，快速试不同目标力
    python3 apps/grip_test.py --target-force 0.5
    python3 apps/grip_test.py --ramp-rate 0.5

    # 导出力曲线，用 Excel 画图看有没有过冲
    python3 apps/grip_test.py --csv logs/trace.csv

⚠️ 第一次上真机时，先夹**海绵或棉花糖**，不要直接上葡萄。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# 让脚本能直接 import 到 src/ 下的包
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

# 控制台编码兜底：某些终端（Windows GBK、串口终端）遇到 ⚠ ✅ 这类字符会直接崩。
# 显式重设为 UTF-8，无法编码的字符降级替换，而不是抛异常。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

from grape_gripper import (  # noqa: E402
    GripperController,
    load_default_config,
    validate_grip_config,
)
from grape_gripper.actuator import create_servo  # noqa: E402
from grape_gripper.safety import EmergencyStop, ForceWatchdog, HeartbeatWatchdog  # noqa: E402
from grape_gripper.sensor import create_sensor_pair  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="葡萄夹取力闭环测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default=None,
                        help="配置文件路径（默认项目根的 config.yaml）")
    parser.add_argument("--sim", action="store_true",
                        help="强制模拟模式，不使用真实硬件")
    parser.add_argument("--real", action="store_true",
                        help="强制真实模式（覆盖 config.yaml 里的 simulate）")
    parser.add_argument("-n", "--count", type=int, default=1,
                        help="连续夹取次数，默认 1")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="两次夹取之间的间隔秒数，默认 1.0")
    parser.add_argument("--tare", action="store_true",
                        help="每次夹取前先对传感器去皮（建议开启）")
    parser.add_argument("--target-force", type=float, default=None,
                        help="覆盖目标夹持力（N）")
    parser.add_argument("--max-force", type=float, default=None,
                        help="覆盖安全力上限（N）")
    parser.add_argument("--ramp-rate", type=float, default=None,
                        help="覆盖力上升速率（N/s），越小越安全")
    parser.add_argument("--approach-rate", type=float, default=None,
                        help="覆盖接近速度（度/秒）")
    parser.add_argument("--hold-time", type=float, default=None,
                        help="覆盖保持时间（秒）")
    parser.add_argument("--csv", default=None,
                        help="把最后一次的力曲线导出到 CSV")
    parser.add_argument("--csv-all", action="store_true",
                        help="每次夹取都导出一个 CSV（自动编号）")
    parser.add_argument("--hz", type=float, default=100.0,
                        help="控制循环频率，默认 100Hz")
    parser.add_argument("--quiet", action="store_true",
                        help="不打印每个周期的实时数据")
    return parser.parse_args()


def apply_overrides(cfg, args: argparse.Namespace) -> None:
    """把命令行参数覆盖到配置上。"""
    data = cfg.as_dict()
    grip = data.setdefault("grip", {})
    if args.target_force is not None:
        grip["target_force_n"] = args.target_force
    if args.max_force is not None:
        grip["force_max_n"] = args.max_force
    if args.ramp_rate is not None:
        grip["force_ramp_n_per_s"] = args.ramp_rate
    if args.approach_rate is not None:
        grip["approach_rate_deg_per_s"] = args.approach_rate
    if args.hold_time is not None:
        grip["hold_time_s"] = args.hold_time

    hw = data.setdefault("hardware", {})
    if args.sim:
        hw["simulate"] = True
    if args.real:
        hw["simulate"] = False


class Printer:
    """控制台输出：实时一行状态 + 换行日志。"""

    def __init__(self, quiet: bool = False):
        self.quiet = quiet
        self._last_line_len = 0

    def log(self, message: str) -> None:
        self._clear_line()
        print(message, flush=True)

    def status(self, text: str) -> None:
        if self.quiet:
            return
        pad = max(0, self._last_line_len - len(text))
        sys.stdout.write("\r" + text + " " * pad)
        sys.stdout.flush()
        self._last_line_len = len(text)

    def _clear_line(self) -> None:
        if self._last_line_len:
            sys.stdout.write("\r" + " " * self._last_line_len + "\r")
            sys.stdout.flush()
            self._last_line_len = 0


def run_once(ctrl: GripperController, sensors, servo, args,
             printer: Printer, csv_path: str | None,
             beat=None) -> object:
    """执行一次完整的夹取流程。"""
    ctrl.reset()

    # 上电后先张开到位（on_tick 喂看门狗：阻塞动作也会让主循环停止心跳）
    servo.move_to(ctrl.angle_open, timeout_s=1.5, on_tick=beat)
    time.sleep(0.15)
    sensors.reset_filter()
    # 丢掉前几个采样（滤波窗口填充期）
    for _ in range(10):
        sensors.update()
        time.sleep(0.005)

    if args.tare:
        printer.log("去皮中（请确保夹爪完全张开、没有接触任何东西）…")
        left_zero, right_zero = sensors.tare(samples=40)
        printer.log(f"去皮完成：左 zero={left_zero:.1f} 右 zero={right_zero:.1f}")

    ctrl.start_grip()
    period = 1.0 / max(1.0, args.hz)
    last_status = 0.0

    while not ctrl.finished:
        loop_start = time.monotonic()
        if beat is not None:
            beat()            # ★ 每个周期都喂看门狗，否则它会误判成主循环卡死
        ctrl.step()
        now = time.monotonic()

        if now - last_status > 0.1:
            last_status = now
            printer.status(
                f"  {ctrl.state.value:<9} "
                f"角度={servo.angle:6.2f}° "
                f"力={sensors.force_n:6.3f}N "
                f"(左{sensors.force_left_n:5.2f} 右{sensors.force_right_n:5.2f}) "
                f"目标={ctrl.target_force_n:.2f}N "
                f"偏={sensors.asymmetry:.2f}"
            )

        elapsed = time.monotonic() - loop_start
        if elapsed < period:
            time.sleep(period - elapsed)

    printer._clear_line()

    if csv_path:
        path = ctrl.dump_csv(csv_path)
        printer.log(f"力曲线已导出：{path}")

    return ctrl.result


def main() -> int:
    args = parse_args()

    # ---- 加载配置 ----
    if args.config:
        from grape_gripper.config import Config
        cfg = Config.load(args.config)
    else:
        cfg = load_default_config()
    apply_overrides(cfg, args)

    printer = Printer(quiet=args.quiet)

    print("=" * 78)
    print("葡萄无损夹取 —— 力闭环测试")
    print("=" * 78)

    # ---- 配置校验 ----
    warnings = validate_grip_config(cfg)
    for w in warnings:
        print(f"⚠️  {w}")
    print(f"模式：{'模拟（不驱动真实硬件）' if cfg.simulate else '真实硬件'}")
    print(f"目标力={cfg.get_float('grip.target_force_n')}N  "
          f"力上限={cfg.get_float('grip.force_max_n')}N  "
          f"上升速率={cfg.get_float('grip.force_ramp_n_per_s')}N/s")
    print(f"接近速度={cfg.get_float('grip.approach_rate_deg_per_s')}°/s  "
          f"保持={cfg.get_float('grip.hold_time_s')}s  "
          f"控制频率={args.hz}Hz")
    print()

    if not cfg.simulate:
        print("⚠️  真实硬件模式。请确认：")
        print("    1. 舵机已独立 5V 供电，且与 K1 共地")
        print("    2. 夹爪行程内**没有手指**，先拿海绵测试")
        print("    3. 随时准备断开舵机电源")
        print()
        try:
            input("按 Enter 继续，Ctrl+C 取消… ")
        except (KeyboardInterrupt, EOFError):
            print("\n已取消")
            return 1

    # ---- 创建硬件对象 ----
    sensors = create_sensor_pair(cfg)
    servo = create_servo(cfg, adc=sensors.left.driver)
    printer.log(f"传感器：{sensors.describe()}")
    printer.log(f"执行器：{servo.describe()}")

    ctrl = GripperController(cfg, sensors, servo, log=printer.log)

    # ---- 安全层 ----
    estop = EmergencyStop(log=printer.log).install()
    force_wd = ForceWatchdog(
        limit_n=ctrl.force_max_n,
        on_violation=lambda f: servo.disable(),
        log=printer.log,
    )
    heartbeat = HeartbeatWatchdog(
        timeout_s=cfg.get_float("safety.heartbeat_timeout_s", 0.5),
        on_timeout=lambda: servo.disable(),
        log=printer.log,
    ).start()

    results = []
    exit_code = 0
    try:
        for i in range(max(1, args.count)):
            if estop.requested:
                break
            if args.count > 1:
                printer.log(f"\n———— 第 {i + 1}/{args.count} 次 ————")

            csv_path = None
            if args.csv:
                csv_path = args.csv
            if args.csv_all:
                base, ext = os.path.splitext(args.csv or "logs/grip_trace.csv")
                csv_path = f"{base}_{i + 1:03d}{ext or '.csv'}"

            result = run_once(ctrl, sensors, servo, args, printer, csv_path,
                              beat=heartbeat.beat)
            results.append(result)
            force_wd.feed(result.peak_force_n)
            heartbeat.beat()

            if not result.success:
                exit_code = 2

            if i < args.count - 1:
                # 间隔期间保持张开，给葡萄/操作者一点时间
                servo.move_to(ctrl.angle_open, timeout_s=1.0,
                              on_tick=heartbeat.beat)
                time.sleep(args.delay)

    except KeyboardInterrupt:
        printer.log("\n用户中断")
        exit_code = 130
    finally:
        # 收尾：张开夹爪，避免一直夹着。
        # 顺序很重要 —— 先喂着看门狗完成收尾动作，最后才停掉它。
        try:
            servo.move_to(ctrl.angle_open, timeout_s=1.5,
                          on_tick=heartbeat.beat)
            servo.disable()
        except Exception as exc:  # noqa: BLE001
            printer.log(f"收尾失败：{exc}")
        heartbeat.stop()
        estop.uninstall()

    # ---- 汇总 ----
    print()
    print("=" * 78)
    if len(results) == 1:
        print(results[0].summary())
    else:
        ok = sum(1 for r in results if r.success)
        print(f"结果：{ok}/{len(results)} 成功  "
              f"（成功率 {100.0 * ok / max(1, len(results)):.1f}%）")
        for idx, r in enumerate(results, 1):
            print(f"  #{idx:03d} {r.summary()}")
        peaks = [r.peak_force_n for r in results]
        if peaks:
            print(f"峰值力：最小 {min(peaks):.3f}N  "
                  f"最大 {max(peaks):.3f}N  "
                  f"平均 {sum(peaks) / len(peaks):.3f}N")
        print()
        if len(results) >= 5:
            if ok == len(results):
                print("✅ 全部成功。可以尝试把 target_force_n 再降 0.1N，看能否更安全。")
            else:
                fails = [r for r in results if not r.success]
                reasons: dict = {}
                for r in fails:
                    reasons[r.reason.value] = reasons.get(r.reason.value, 0) + 1
                print(f"❌ 失败原因统计：{reasons}")
                if "over_force" in reasons:
                    print("   → 超力：降低 force_ramp_n_per_s（如 0.8），"
                          "或降低 target_force_n")
                if "slip" in reasons:
                    print("   → 滑移：葡萄可能在滚动，检查对中性 / 加硅胶垫 / "
                          "降低 approach_rate_deg_per_s")
                if "asymmetry" in reasons:
                    print("   → 偏心：夹爪中心没对准葡萄，检查相机到夹爪的标定")
                if "timeout" in reasons:
                    print("   → 超时：葡萄不在行程内，或夹爪行程/角度标定不对")
    print("=" * 78)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
