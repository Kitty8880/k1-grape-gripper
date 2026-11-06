#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脉宽扫描 + 拍照 —— 为「边缘断口判状态」提供标定数据。

为什么要扫描而不是"定位到刚好接触"
----------------------------------
直觉上应该：慢慢闭合，力刚出现就停，这样就是"贴住未夹紧"。

**但力传感器做不到这件事。** 实测数据：

    单样本噪声 σ ≈ 2.65 N（4,186 counts）
    0.6 s 窗口   σ ≈ 0.24 N
    2.0 s 窗口   σ ≈ 0.22 N

"刚好接触"对应的力大约 0.02~0.05 N，
而噪声是 **0.24 N** —— 信噪比不到 0.2。

也就是说：**力传感器分辨不出「贴住」和「没碰到」，
更分辨不出「贴住」和「差 2µs」。**

所以改成**逐档拍摄**：

    940µs → 930µs → 920µs → ... → 800µs
    每一档：等力稳定 → 记录力中位数 → 拍一张图

拍完之后：

    ① 看脉宽 → 力的曲线，找到力开始上升的拐点（≈ 接触点）
    ② 算每张图的 max_gap（极坐标断口），找到断口开始出现的档位
    ③ 两者的交集就是"贴住但未夹紧"的窄窗口

**这个扫描一次解决三个问题**：标定像素/毫米、定 Canny 阈值、
选出 b) 组（贴住未夹紧）的图。

用法::

    # 完整扫描 940 → 800，每 10µs 一档，每档拍一张
    sudo python3 apps/pose_sweep.py --out debug/sweep

    # 只扫接触附近（更快）
    sudo python3 apps/pose_sweep.py --from 920 --to 860 --step 5

    # 不拍照，只看力曲线
    sudo python3 apps/pose_sweep.py --no-capture

    # 停在某一档不动（手动摆位拍照用）
    sudo python3 apps/pose_sweep.py --hold 880

输出::

    debug/sweep/0880us_f1234.jpg     文件名里带脉宽和力
    debug/sweep/sweep.csv            脉宽, 力, σ, 样本数, 图片名

⚠️ 运行前先停服务：
    sudo systemctl stop grape-gripper
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
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

from grape_gripper.config import Config, default_config_path   # noqa: E402


def build_hardware(cfg):
    """按配置建舵机和力传感器，和 grasp_with_verify.py 完全一致。"""
    from grape_gripper.actuator import BusyPwmServo
    from grape_gripper.ps01 import Ps01Reader

    pin = cfg.get_int("servo.gpio", 49)
    pmin = cfg.get_float("servo.pulse_min_us", 750.0)
    pmax = cfg.get_float("servo.pulse_max_us", 1450.0)
    servo = BusyPwmServo(gpio_pin=pin, period_us=20000.0,
                         pulse_min_us=pmin, pulse_max_us=pmax,
                         angle_min=0.0, angle_max=180.0)

    port = cfg.get_str("sensor.ps01.port", "/dev/ttyUSB0")
    reader = Ps01Reader(port)
    reader.open()
    if not reader.wait_first_sample(timeout=3.0):
        raise RuntimeError(f"传感器无数据：{reader.stats()}")

    # 灵敏度用标定值
    try:
        k = cfg.get_float("sensor.calib.right.counts_per_n", 1564.0)
    except Exception:  # noqa: BLE001
        k = 1564.0

    # ⚠️ 空载基线**必须实测**，不能用配置里的常量。
    #
    #   2026-11-05 踩过：配置里 sensor.calib.right.raw_zero = 693,300
    #   是**传感器 B** 的基线，而通道 channel(1) 读的是**传感器 A**
    #   （A 约 682,700）。两者差 10,600 counts ≈ 6.8 N，
    #   于是张开状态下力读数直接显示 -6.844 N。
    #
    #   力控脚本 grasp_with_verify.py 的做法是对的：
    #   启动时开爪、测一段、把中位数当基线（日志里那行「空载力 +0.056N」）。
    #   这里照抄。
    print("  测空载基线（张开状态，约 1.2 s）...")
    servo.set_pulse_us(pmax)
    time.sleep(1.2)
    base_vals = []
    t0 = time.time()
    while time.time() - t0 < 1.2:
        v = reader.channel(1)
        if v is not None:
            base_vals.append(float(v))
        time.sleep(0.002)
    if len(base_vals) < 10:
        raise RuntimeError("基线采样不足，传感器可能没在输出")
    raw_zero = statistics.median(base_vals)
    print(f"  空载基线 = {raw_zero:,.0f} counts "
          f"（{len(base_vals)} 个样本）")

    return servo, reader, raw_zero, k, pmin, pmax


def measure_force(reader, raw_zero, k, window: float = 1.2):
    """
    在 window 秒内采样，返回 (力N, σN, 样本数)。

    ⚠️ 用**中位数**，不能用均值 —— PS01 每约 2 秒冒一个大尖峰
    （+11,000 counts ≈ +7 N），均值会被它整体拉高 1.2 N。
    """
    vals = []
    t0 = time.time()
    while time.time() - t0 < window:
        v = reader.channel(1)
        if v is not None:
            vals.append(float(v))
        time.sleep(0.002)
    if len(vals) < 8:
        return 0.0, 0.0, len(vals)
    med = statistics.median(vals)
    return ((med - raw_zero) / k,
            statistics.pstdev(vals) / k,
            len(vals))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="脉宽扫描 + 拍照（为边缘断口判状态做标定）")
    ap.add_argument("--from", dest="p_from", type=float, default=940.0,
                    help="起始脉宽（默认 940 = 预闭合位）")
    ap.add_argument("--to", dest="p_to", type=float, default=800.0,
                    help="结束脉宽（默认 800，安全下限以上）")
    ap.add_argument("--step", type=float, default=10.0,
                    help="每档变化量（默认 10µs）")
    ap.add_argument("--win", type=float, default=1.2,
                    help="每档测力窗口秒数（默认 1.2）")
    ap.add_argument("--settle", type=float, default=0.5,
                    help="移到位置后等多久再测力（默认 0.5）")
    ap.add_argument("--out", default="debug/sweep", help="图片输出目录")
    ap.add_argument("--no-capture", action="store_true", help="不拍照")
    ap.add_argument("--hold", type=float, default=None,
                    help="只移到这个脉宽并停住")
    ap.add_argument("--capture", type=int, default=0,
                    help="--hold 模式下自动连拍几张（默认 0 = 手动）")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = Config.load(args.config or default_config_path())
    print("=" * 72)
    print("脉宽扫描 + 拍照")
    print("=" * 72)

    try:
        servo, reader, raw_zero, k, pmin, pmax = build_hardware(cfg)
    except Exception as exc:  # noqa: BLE001
        print(f"❌ 硬件初始化失败：{exc}")
        print("   排查：sudo pkill -f product_mode; sudo systemctl stop grape-gripper")
        return 1

    cam = None
    if not args.no_capture:
        try:
            from k1_camera_capture import K1Camera
            outdir = os.path.join(_ROOT, args.out)
            os.makedirs(outdir, exist_ok=True)
            cam = K1Camera(use_sudo=False)
            print(f"  摄像头：cam-test（约 1.2 s/张）")
            print(f"  输出　：{outdir}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠️ 摄像头不可用（继续只测力）：{exc}")
            cam = None

    print(f"  舵机　：GPIO{cfg.get_int('servo.gpio', 49)}  "
          f"{pmin:.0f}~{pmax:.0f}µs")
    print(f"  传感器：{cfg.get_str('sensor.ps01.port', '/dev/ttyUSB0')}  "
          f"基线 {raw_zero:,.0f}  灵敏度 {k:,.0f} counts/N")
    print(f"  测力　：{args.win:.1f}s 窗口中位数")
    print()

    # ---------- 单独摆位模式 ----------
    if args.hold is not None:
        p = max(pmin, min(pmax, args.hold))
        print(f"移到 {p:.0f}µs 并保持 ...")
        servo.set_pulse_us(p)
        time.sleep(1.0)
        f, sg, n = measure_force(reader, raw_zero, k, args.win)
        print(f"  力 = {f:+.3f} N   σ = {sg:.3f} N   样本 {n}")

        # ★ 自动连拍 N 张（拍的时候位置不动）
        if args.capture > 0 and cam is not None:
            import cv2
            print(f"  连拍 {args.capture} 张（位置保持不动）...")
            for i in range(args.capture):
                try:
                    bgr = cam.capture()
                    name = (f"hold{p:06.0f}us_{i+1:02d}_f"
                            f"{f:+.3f}.jpg").replace("+", "p").replace("-", "m")
                    cv2.imwrite(os.path.join(outdir, name), bgr)
                    print(f"    [{i+1}/{args.capture}] {name}", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"    [{i+1}/{args.capture}] 失败：{exc}", flush=True)
        else:
            print("  现在可以手动拍照。回车退出。")
            try:
                input()
            except EOFError:
                time.sleep(30)

        servo.set_pulse_us(pmax)
        time.sleep(0.3)
        servo.close()
        reader.close()
        return 0

    # ---------- 扫描 ----------
    # ⚠️ 第一版这里写错了：先把 from/to 排序成 lo/hi，然后用 hi 当上界，
    #    结果从 940 往 850 扫时只生成 1 档（940 >= 940 成立，930 就不成立了）。
    #    正确做法：直接用带符号的步长，从 p_from 走到 p_to。
    signed = -abs(args.step) if args.p_from > args.p_to else abs(args.step)
    steps = []
    p = args.p_from
    while (p >= args.p_to - 1e-6) if signed < 0 else (p <= args.p_to + 1e-6):
        steps.append(round(p, 1))
        p += signed
        if len(steps) > 500:
            break
    print(f"  扫描 {len(steps)} 档：{steps[0]:.0f} → {steps[-1]:.0f} µs")
    print()

    # 先全开，确保出发点一致
    servo.set_pulse_us(pmax)
    time.sleep(1.0)

    rows = []
    try:
        print(f"  {'脉宽':>7} {'力(N)':>9} {'σ(N)':>7} {'样本':>6}  图片")
        print("  " + "-" * 62)
        for pulse in steps:
            servo.set_pulse_us(pulse)
            time.sleep(args.settle)
            f, s, n = measure_force(reader, raw_zero, k, args.win)

            img_name = ""
            if cam is not None:
                try:
                    bgr = cam.capture()
                    import cv2
                    img_name = f"{pulse:06.0f}us_f{f:+.3f}.jpg".replace("+", "p") \
                                                              .replace("-", "m")
                    cv2.imwrite(os.path.join(outdir, img_name), bgr)
                except Exception as exc:  # noqa: BLE001
                    img_name = f"（拍照失败：{exc}）"

            print(f"  {pulse:7.0f} {f:+9.3f} {s:7.3f} {n:6d}  {img_name}")
            rows.append((pulse, f, s, n, img_name))

        # ---------- 写 CSV ----------
        if cam is not None:
            import csv
            csv_path = os.path.join(outdir, "sweep.csv")
            with open(csv_path, "w", encoding="utf-8", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["pulse_us", "force_n", "sigma_n", "samples", "image"])
                w.writerows(rows)
            print()
            print(f"  CSV：{csv_path}")
    except KeyboardInterrupt:
        print("\n中断")
    finally:
        try:
            servo.set_pulse_us(pmax)
            time.sleep(0.3)
            servo.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            reader.close()
        except Exception:  # noqa: BLE001
            pass

    # ---------- 找接触拐点 ----------
    if rows:
        print()
        print("=" * 72)
        print("力曲线拐点分析")
        print("=" * 72)
        forces = [r[1] for r in rows]
        base = statistics.median(forces[:3]) if len(forces) >= 3 else forces[0]

        # ⚠️ 判据要用**中位数的标准误差**，不是单样本 σ。
        #
        #   2026-11-05 第一版写错了：直接用 pstdev/1564 得到的 σ≈2.58 N
        #   当噪声，于是阈值 = 基线 + 2×2.58 = +5.1 N，
        #   结果 850µs 明明已经 +0.42 N 了还判"没有上升"。
        #
        #   正确关系：n 个样本的中位数，其标准误差约为
        #       1.253 × σ_single / √n
        #   本机实测 n≈566、σ_single≈2.58 N → SE ≈ 0.14 N。
        #   所以判据应该用 2×SE ≈ 0.27 N，而不是 5.1 N。
        n_typ = int(statistics.median([r[3] for r in rows])) or 1
        sig_single = statistics.median([r[2] for r in rows]) or 0.2
        se = 1.253 * sig_single / (n_typ ** 0.5)
        print(f"  起始基线力 {base:+.3f} N")
        print(f"  单样本 σ {sig_single:.3f} N（n={n_typ}）")
        print(f"  → 中位数的标准误差 SE ≈ {se:.3f} N")
        print(f"  「力开始上升」判据：力 > 基线 + 2×SE = {base + 2*se:+.3f} N")
        hit = None
        for pulse, f, s, n, img in rows:
            if f > base + 2 * se:
                hit = (pulse, f)
                break
        if hit:
            print(f"  → 接触点约在 **{hit[0]:.0f} µs**（力 {hit[1]:+.3f} N）")
            print(f"  → 「贴住但未夹紧」的窗口在这之前 1~2 档："
                  f"{hit[0] + args.step:.0f}~{hit[0] + 2*args.step:.0f} µs")
        else:
            print("  → 扫描范围内力没有明显上升：葡萄比最紧那档还窄（或没放好）")
        print()
        print("  下一步：把带脉冲的图传到 PC，用 max_gap 分析找出")
        print("  「断口刚出现」的档位 —— 那才是真正的「贴住」点。")
    return 0


if __name__ == "__main__":
    sys.exit(main())