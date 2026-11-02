#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
力闭环实机测试（v2）—— 夹住物体并把夹持力控制到目标值。

v1 的教训（必须记录，否则会重蹈覆辙）
------------------------------------
1. **PD1 的有效采样率是 53Hz，不是 160Hz**
   传感器 160Hz 输出是在三个 PD 通道间轮流的，单个通道只占 1/3。
   所以"平均 0.3 秒"只有约 16 个独立样本，噪声 σ ≈ 0.67N
   —— 和目标力 0.8N 同量级，根本无法做单点判定。
   要把 σ 压到 0.15N 以下需要约 6 秒平均。

2. **接触阈值不能低于噪声**
   v1 用 0.30N 当接触判据，结果在 1444µs（几乎全开、根本没夹东西）
   就误判成"接触"。阈值至少要是平均后噪声的 3~4 倍。

3. **连续 PI 在噪声下会随机游走**
   每周期 kp×σ 的扰动会累积成位置漂移。

所以 v2 改成 **"测量 → 移动 → 再测量"** 的分步结构：

    每次先用 --window 秒（默认 1.0s，约 53 个独立样本）测出平均力，
    再根据误差决定移动多少微秒，然后停下来重新测量。

这样每一步的决策都建立在低噪声的测量上，不会随风飘。

状态机
------
    OPEN      张开
    APPROACH  以固定慢速闭合，直到平均力超过接触阈值（需连续 2 次确认）
    ADJUST    分步调整：测量 → 按误差移动 → 再测量（最多 N 次）
    HOLD      保持位置，统计稳定性
    RELEASE   张开卸力

三道安全防线
------------
    ① 力 > force_max_n（默认 1.5N）→ 立即反向张开并退出
    ② 脉宽不低于 pulse_min_us（650µs，实测两指不接触）
    ③ 每步移动量有上限，且总步数有限

用法::

    # 夹软物体（海绵/纸团），目标 0.8N
    sudo python3 apps/force_loop_test.py

    # 第一次试：更慢、目标更低、时间更短
    sudo python3 apps/force_loop_test.py --target 0.5 --approach 20 --hold 3

    # 只看过程（不动舵机）
    sudo python3 apps/force_loop_test.py --dry-run
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
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

# 单个 PD 通道的有效更新率 = 总输出率 / 3
PD_CHANNEL_HZ = 160.0 / 3.0     # ≈ 53 Hz


def main() -> int:
    ap = argparse.ArgumentParser(description="力闭环实机测试 v2")
    ap.add_argument("--config", default=None)
    ap.add_argument("--target", type=float, default=None,
                    help="目标夹持力（N），默认取配置 grip.target_force_n")
    ap.add_argument("--contact", type=float, default=None,
                    help="接触判定力（N）。默认自动取 max(0.5, 3×平均噪声)")
    ap.add_argument("--max-force", type=float, default=None,
                    help="硬上限（N），默认取配置 grip.force_max_n")
    ap.add_argument("--approach", type=float, default=25.0,
                    help="（已废弃，保留兼容）接近速率（微秒/秒）")
    ap.add_argument("--fast-step", type=float, default=30.0,
                    help="远离接触时的每步脉宽（微秒）")
    ap.add_argument("--slow-step", type=float, default=5.0,
                    help="接近接触时的每步脉宽（微秒）★关键安全参数")
    ap.add_argument("--fast-threshold", type=float, default=0.10,
                    help="力超过此值就从快步长切到慢步长（牛顿）")
    ap.add_argument("--window", type=float, default=0.8,
                    help="（保留兼容）基准测力窗口（秒）。实际用下面的远近两档")
    ap.add_argument("--window-far", type=float, default=0.25,
                    help="离目标远时的测力窗口（秒）——短一点跑得快")
    ap.add_argument("--window-near", type=float, default=2.0,
                    help="接近目标时的测力窗口（秒）——长一点噪声低 ★")
    ap.add_argument("--near-gap", type=float, default=0.45,
                    help="力低于 (目标 - 本值) 时算'远'（牛顿）")
    ap.add_argument("--max-move-far", type=float, default=25.0,
                    help="远距离时每步固定走多少（微秒）——大步快速接近")
    ap.add_argument("--max-move-near", type=float, default=6.0,
                    help="接近时单步最大移动（微秒）")
    ap.add_argument("--stop-pulse", type=float, default=700.0,
                    help="位置下限（微秒）。闭环到这里就不再闭合 —— "
                         "这是比力判据更可靠的保护，因为力噪声大而位置是确定的")
    ap.add_argument("--confirm", type=int, default=2,
                    help="需要连续几步都在容差内才算收敛（防噪声尖峰误判）")
    ap.add_argument("--max-time", type=float, default=90.0,
                    help="整个闭环的最长时长（秒），超时退出")
    ap.add_argument("--min-step", type=float, default=2.0,
                    help="最小步进（微秒），防止误差很小时原地不动")
    ap.add_argument("--max-iter", type=int, default=120,
                    help="最多闭环步数")
    ap.add_argument("--tol", type=float, default=0.15,
                    help="收敛容差（N）")
    ap.add_argument("--kp", type=float, default=15.0,
                    help="调整增益（微秒/牛）")
    ap.add_argument("--max-move", type=float, default=8.0,
                    help="（保留兼容）调整阶段单步最大移动量（微秒）")
    ap.add_argument("--hold", type=float, default=5.0, help="保持并统计的时长（秒）")
    ap.add_argument("--release", type=float, default=None, help="松开脉宽")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = Config.load(args.config or default_config_path())
    if cfg.simulate and not args.dry_run:
        print("⚠️ config.yaml 里 hardware.simulate = true，请改成 false")
        return 1

    target = args.target if args.target is not None else \
        cfg.get_float("grip.target_force_n", 0.8)
    max_force = args.max_force if args.max_force is not None else \
        cfg.get_float("grip.force_max_n", 1.5)
    pmin = cfg.get_float("servo.pulse_min_us", 650.0)
    pmax = cfg.get_float("servo.pulse_max_us", 1450.0)
    release = args.release if args.release is not None else pmax
    raw_zero = cfg.get_float("sensor.calib.left.raw_zero", 683457.0)
    raw_span = cfg.get_float("sensor.calib.left.raw_span", 15640.0)
    f_span = cfg.get_float("sensor.calib.left.force_max_n", 10.0)
    k = raw_span / f_span if f_span else 1564.0
    pin = cfg.get_int("servo.gpio_pin", 49)

    n_window = max(1.0, args.window * PD_CHANNEL_HZ)

    print("=" * 74)
    print("力闭环实机测试 v2（测量 → 移动 → 再测量）")
    print("=" * 74)
    print(f"  目标力 {target:.2f}N    硬上限 {max_force:.2f}N ★")
    print(f"  脉宽限位 {pmin:.0f}~{pmax:.0f}us    灵敏度 {k:,.0f} 计数/N")
    print(f"  测力窗口：远 {args.window_far:.1f}s → 近 {args.window_near:.1f}s"
          f"（PD1 有效更新率 {PD_CHANNEL_HZ:.0f}Hz）")
    print(f"  调整增益 {args.kp}us/N   步长上限 远 {args.max_move_far:.0f} "
          f"/ 近 {args.max_move_near:.0f}us")
    print()
    print("⚠️ 操作：把要夹的东西（海绵/纸团/小番茄）放进两指之间，")
    print("   手放在电源开关上。听到持续嗡嗡声立刻 Ctrl+C。")
    print()

    if args.dry_run:
        print("（--dry-run：不驱动硬件）")
        return 0

    # ---------------- 硬件 ----------------
    try:
        from grape_gripper.actuator import BusyPwmServo
        servo = BusyPwmServo(gpio_pin=pin, period_us=20000.0,
                             pulse_min_us=pmin, pulse_max_us=pmax,
                             angle_min=0.0, angle_max=180.0)
    except Exception as exc:  # noqa: BLE001
        print(f"❌ 舵机初始化失败：{exc}")
        return 1

    try:
        from grape_gripper.ps01 import Ps01Reader
        reader = Ps01Reader(cfg.get_str("sensor.ps01.port", "/dev/ttyUSB0"))
        reader.open()
    except Exception as exc:  # noqa: BLE001
        print(f"❌ 传感器初始化失败：{exc}")
        servo.close()
        return 1
    if not reader.wait_first_sample(timeout=3.0):
        print(f"❌ 传感器无数据：{reader.stats()}")
        servo.close()
        reader.close()
        return 1

    # ---------------- 测力 ----------------
    def measure(window: float):
        """
        返回 (平均力 N, 标准差 N, 轮询次数, ★独立样本数)。

        ⚠️ 独立样本数很关键：reader.channel() 每 2ms 轮询一次，但 PD1 实际
        只以约 53Hz 更新，所以 1 秒内 500 次轮询里只有约 53 个**不同**的值。
        用轮询次数去算"平均后噪声"会严重高估降噪效果（踩过这个坑）。
        """
        vals = []
        distinct = 0
        last = None
        t0 = time.time()
        while time.time() - t0 < window:
            v = reader.channel(1)
            if v is not None:
                vals.append(float(v))
                if v != last:
                    distinct += 1
                    last = v
            time.sleep(0.002)
        if len(vals) < 5:
            return 0.0, 0.0, 0, 0
        mu = statistics.mean(vals)
        sd = statistics.pstdev(vals)
        return (mu - raw_zero) / k, sd / k, len(vals), distinct

    pulse = pmax
    f_prev = 0.0
    try:
        input("按回车开始（先张开）...")
        servo.set_pulse_us(pmax)
        pulse = pmax
        print(f"\n[OPEN] 张开到 {pmax:.0f}us")
        time.sleep(1.5)

        f_base, sd_base, n_base, nd_base = measure(args.window)
        print(f"       空载：力 {f_base:+.3f}N   单点噪声 σ {sd_base:.2f}N   "
              f"（轮询 {n_base} 次，独立样本 {nd_base} 个 → 有效 "
              f"{nd_base/args.window:.0f}Hz）")

        # ★ 用**独立样本数**折算平均后的噪声（不是轮询次数！）
        sigma_avg = sd_base / (nd_base ** 0.5) if nd_base else sd_base
        print(f"       平均 {args.window:.1f}s 后噪声 σ ≈ {sigma_avg:.2f}N")
        print(f"       （闭环里靠近目标时会用 {args.window_near:.1f}s 窗口，"
              f"噪声会更低）")
        # 注意：这里不再用它算"接触阈值" —— 见下面的说明

        def force_now() -> float:
            f, _, _, _ = measure(args.window)
            return f - f_base

        # ---------------- 统一闭环 ----------------
        # 不再分 APPROACH / ADJUST 两段，也不用"力阈值"判断接触 ——
        # 因为 PD1 有效更新率仅约 22Hz，0.8s 平均后噪声仍有 0.61N，
        # 任何"力阈值"都会被噪声误触发，或者（更糟）设得比硬上限还高
        # 而永远无法满足（v3 就踩了这个坑：阈值 1.82N > 硬上限 1.50N）。
        #
        # 改成一个闭环：每步测力 → 按误差决定移动量 → 再测。
        # 接近目标时自动放慢步长并加长平均窗口（噪声按 √N 下降）。
        print(f"\n[CLOSED-LOOP] 单步闭环到 {target:.2f}N（容差 {args.tol:.2f}N）")
        print(f"   远距离：窗口 {args.window_far:.1f}s，最大步长 "
              f"{args.max_move_far:.0f}us")
        print(f"   接近时：窗口 {args.window_near:.1f}s，最大步长 "
              f"{args.max_move_near:.0f}us")
        print()
        t0 = time.time()
        converged = False
        in_tol_streak = 0
        stop_pulse = max(args.stop_pulse, pmin)
        for it in range(1, args.max_iter + 1):
            if time.time() - t0 > args.max_time:
                print(f"\n   ⏱ 超过最长时长 {args.max_time:.0f}s，退出闭环")
                break
            # 自适应窗口：力还低（离目标远）时用短窗口跑得快，
            #               接近目标时用长窗口压低噪声
            near = f_prev > (target - args.near_gap)
            win = args.window_near if near else args.window_far
            max_move = args.max_move_near if near else args.max_move_far

            f, sd, _, nd = measure(win)
            f -= f_base
            f_prev = f
            err = target - f

            if f > max_force:
                print(f"   第 {it:3d} 步  力 {f:+.3f}N  "
                      f"❌ 超硬上限 {max_force:.2f}N，紧急松开")
                servo.set_pulse_us(release)
                time.sleep(1.5)
                return 1

            mode = "近" if near else "远"
            print(f"   第 {it:3d} 步  {time.time()-t0:5.1f}s  "
                  f"脉宽 {pulse:6.0f}us  力 {f:+6.3f}N  "
                  f"误差 {err:+6.3f}N  [{mode} 窗口{win:.1f}s "
                  f"步≤{max_move:.0f}us]", end="")

            # ★ 需要连续 --confirm 步都在容差内，才算收敛 ——
            #   否则一次噪声尖峰就会让闭环"假收敛"（v4 踩过这个坑）
            if abs(err) <= args.tol:
                in_tol_streak += 1
                if in_tol_streak >= args.confirm:
                    print(f"   ✅ 已收敛（连续 {in_tol_streak} 步在容差内）")
                    converged = True
                    break
                print(f"   （容差内 {in_tol_streak}/{args.confirm}）")
            else:
                in_tol_streak = 0

            # ★ 远距离用固定大步长，不做比例控制
            #   踩过的坑：比例控制下 err≈0.5N 时 step = 0.5×15 = 7.5µs，
            #   明明上限是 20µs 却走不满，700µs 的行程花了 87 步 / 90 秒。
            if near:
                step = err * args.kp
                step = max(-max_move, min(max_move, step))
            else:
                step = max_move if err > 0 else -max_move
            # 保证最小步进，否则误差很小时会原地不动
            if 0 < step < args.min_step:
                step = args.min_step
            elif -args.min_step < step < 0:
                step = -args.min_step

            new_pulse = max(stop_pulse, min(pmax, pulse - step))
            if abs(new_pulse - pulse) < 0.1:
                # 已到位置下限：这时只能报告"位置到底了"
                print(f"   🛑 已到位置下限 {stop_pulse:.0f}us")
                print(f"      此时力 {f:+.3f}N —— "
                      f"{'已经夹住' if f > 0.15 else '仍未夹到东西'}")
                break
            print(f"  → {new_pulse - pulse:+.0f}us")
            pulse = new_pulse
            servo.set_pulse_us(pulse)
            time.sleep(0.15)

        if not converged:
            print(f"\n   ⚠️ 未在 {args.max_iter} 步内收敛到 "
                  f"{target:.2f}±{args.tol:.2f}N")

        # ---------------- HOLD ----------------
        print(f"\n[HOLD] 保持位置，统计 {args.hold:.0f}s")
        t0 = time.time()
        samples = []
        while time.time() - t0 < args.hold:
            f = force_now()
            samples.append(f)
            print(f"   {time.time()-t0:5.1f}s  力 {f:+6.3f}N   "
                  f"脉宽 {pulse:.0f}us")
            if f > max_force:
                print(f"   ❌ 力 {f:.2f}N 超硬上限，紧急松开")
                servo.set_pulse_us(release)
                time.sleep(1.5)
                return 1

        # ---------------- 统计 ----------------
        print()
        print("=" * 74)
        print("结果")
        print("=" * 74)
        if len(samples) >= 3:
            mu = statistics.mean(samples)
            sd = statistics.pstdev(samples)
            print(f"  稳态力   均值 {mu:.3f}N   标准差 {sd:.3f}N   "
                  f"范围 {min(samples):.3f} ~ {max(samples):.3f}N")
            print(f"  目标      {target:.3f}N   稳态误差 {mu-target:+.3f}N")
            print(f"  最终脉宽  {pulse:.0f}us")
            print()
            if abs(mu - target) <= args.tol and sd <= 0.25:
                print("  ✅ 闭环成功（均值误差在容差内，波动可接受）")
            elif abs(mu - target) <= 0.3:
                print("  🔶 基本可用（误差 ≤0.3N）")
            else:
                print("  ⚠️ 偏差较大 —— 检查安装或调整 --kp")
            print()
            print(f"  说明：传感器单通道噪声 σ≈{sigma_avg:.2f}N，"
                  f"所以±{2*sigma_avg:.2f}N 内的波动属正常。")
            print(f"        实际夹持力由'位置'决定，比读数稳定得多。")

        print()
        print("[RELEASE] 松开")
        servo.set_pulse_us(release)
        time.sleep(1.5)

    except KeyboardInterrupt:
        print("\n中断 —— 立即松开")
        try:
            servo.set_pulse_us(release)
            time.sleep(1.0)
        except Exception:  # noqa: BLE001
            pass
    finally:
        for obj in (servo, reader):
            try:
                obj.close()
            except Exception:  # noqa: BLE001
                pass

    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())