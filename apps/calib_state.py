#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
现场标定：在**实时流**上量出三档的扁平度，算出本机的阈值。

为什么必须现场标定
------------------
2026-11-06 实测发现一个系统偏差：

    存档照片（K1Camera 单张拍摄）    自由状态扁平度 中位 0.130
    实时流  （K1CameraStream）      自由状态扁平度 中位 0.147

差 0.017，而"自由→贴住"的间距只有 0.026 —— 用照片上标定的阈值
（0.1442）去判实时流，**自由状态会被判成"贴住"**。

原因：两条采集路径的自动曝光/白平衡行为不同，掩码边界因此偏移
1~2 像素，而 `approxPolyDP` 的最长边对这种边界变化很敏感。

**所以阈值必须在实际使用的那条路径上标定。**

标定方法（不需要人工判断"贴上没有"，靠力自动定位）
--------------------------------------------------
    ① 张开到最大脉宽 → 等稳定器收敛 → 记 flat_free
    ② 以 4 µs 步长逐步闭合，每档停 0.8 s 测力 + 采扁平度
    ③ 力首次超过 0.05 N 的那一档 = "贴住"，记 flat_touch
    ④ 力首次超过 0.90 N 的那一档 = "夹紧"，记 flat_grip
    ⑤ 阈值取相邻两档的中点（留最大余量）

用法::

    # 标定（会驱动舵机，葡萄要放好在两指之间）
    sudo python3 apps/calib_state.py

    # 只看不改（不驱动舵机，只用当前状态报数）
    sudo python3 apps/calib_state.py --dry-run
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
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
sys.path.insert(0, _HERE)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

from grape_gripper.config import Config, default_config_path   # noqa: E402
from grape_state import GrapeStateDetector, StateStabilizer    # noqa: E402

TOUCH_N = 0.05
GRIP_N = 0.90


class LiveVision:
    """实时流 + 稳定器：持续给出平滑后的扁平度。"""

    def __init__(self, det, stab, width=1920, height=1080):
        self.det = det
        self.stab = stab
        self.width = width
        self.height = height
        self.cam = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._flat = None
        self._state = None
        self._raw = None
        self.n = 0
        self.thread = None

    def start(self):
        from k1_camera_stream import K1CameraStream
        self.cam = K1CameraStream(width=self.width, height=self.height,
                                  use_sudo=False, verbose=False)
        self.cam.open()
        self._stop.clear()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def _loop(self):
        while not self._stop.is_set():
            t0 = time.time()
            try:
                bgr = self.cam.read_frame(timeout=4.0)
                if bgr is None:
                    time.sleep(0.1)
                    continue
                r = self.det.detect(bgr)
            except Exception:  # noqa: BLE001
                time.sleep(0.2)
                continue
            s = self.stab.update(r)
            with self._lock:
                self._raw = r.get("flatness")
                if s["stable"]:
                    self._flat = s["flatness"]
                    self._state = s["state"]
                    self.n += 1
            dt = time.time() - t0
            if dt < 0.30:
                time.sleep(0.30 - dt)
            else:
                time.sleep(0.02)

    def latest(self):
        with self._lock:
            return self._flat, self._state, self._raw

    def stop(self):
        self._stop.set()
        if self.thread:
            self.thread.join(timeout=3.0)
        try:
            if self.cam:
                self.cam.close()
        except Exception:  # noqa: BLE001
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description="在实时流上标定三档扁平度阈值")
    ap.add_argument("--dry-run", action="store_true", help="不驱动舵机")
    ap.add_argument("--step", type=float, default=10.0,
                    help="闭合步长(µs)。默认 10 —— 用 4 的话"
                         "从 1450 到 840 要 150 多步、跑两分钟太慢")
    ap.add_argument("--hold", type=float, default=0.5, help="每档停留(秒)")
    ap.add_argument("--settle", type=float, default=3.0, help="每档等稳定的秒数")
    args = ap.parse_args()

    cfg = Config.load(default_config_path())
    pmin = cfg.get_float("servo.pulse_min_us", 750.0)
    pmax = cfg.get_float("servo.pulse_max_us", 1450.0)
    k = cfg.get_float("sensor.calib.right.counts_per_n", 1564.0)

    print("=" * 74)
    print("现场标定：三档扁平度阈值（实时流）")
    print("=" * 74)
    print(f"  脉宽范围 {pmin:.0f}~{pmax:.0f}µs，步长 {args.step:.0f}µs，"
          f"每档等 {args.settle:.1f}s")

    det = GrapeStateDetector()
    vision = LiveVision(det, StateStabilizer(window=7, gate_px=60.0))
    vision.start()
    print("  视觉已启动，等稳定器收敛…", flush=True)
    time.sleep(2.0)

    servo = None
    reader = None
    raw_zero = None
    if not args.dry_run:
        try:
            from grape_gripper.actuator import BusyPwmServo
            servo = BusyPwmServo(gpio_pin=cfg.get_int("servo.gpio", 49),
                                 period_us=20000.0, pulse_min_us=pmin,
                                 pulse_max_us=pmax, angle_min=0.0,
                                 angle_max=180.0)
        except Exception as exc:  # noqa: BLE001
            print(f"❌ 舵机失败：{exc}")
            vision.stop()
            return 1
        try:
            from grape_gripper.ps01 import Ps01Reader
            reader = Ps01Reader(cfg.get_str("sensor.ps01.port", "/dev/ttyUSB0"))
            reader.open()
            if not reader.wait_first_sample(timeout=3.0):
                raise RuntimeError("传感器无数据")
            vals = []
            t0 = time.time()
            while time.time() - t0 < 1.2:
                v = reader.channel(1)
                if v is not None:
                    vals.append(float(v))
                time.sleep(0.002)
            raw_zero = statistics.median(vals)
            print(f"  力基线 {raw_zero:.0f} counts")
        except Exception as exc:  # noqa: BLE001
            print(f"❌ 传感器失败：{exc}")
            vision.stop()
            return 1

    def force(window=0.25):
        if reader is None or raw_zero is None:
            return 0.0
        vals = []
        t0 = time.time()
        while time.time() - t0 < window:
            v = reader.channel(1)
            if v is not None:
                vals.append(float(v))
            time.sleep(0.002)
        if len(vals) < 4:
            return 0.0
        vals.sort()
        return (vals[len(vals) // 2] - raw_zero) / k

    rows = []
    try:
        # ---- ① 张开 ----
        if servo is not None:
            servo.set_pulse_us(pmax)
        print(f"\n  ① 张开到 {pmax:.0f}µs，等 {args.settle:.1f}s…", flush=True)
        time.sleep(args.settle)
        f = force()
        flat, st, raw = vision.latest()
        rows.append((pmax, f, flat, st))
        print("     力 %+6.3fN   扁平度 %s   判定 %s"
              % (f, ("%.4f" % flat) if flat else "--", st or "--"))

        # ---- ② 逐步闭合 ----
        if servo is not None:
            print(f"\n  ② 逐步闭合（步长 {args.step:.0f}µs）…", flush=True)
            pulse = pmax - args.step
            while pulse >= pmin:
                servo.set_pulse_us(pulse)
                time.sleep(args.hold)
                flat, st, raw = vision.latest()
                f = force()
                rows.append((pulse, f, flat, st))
                print("     %4.0fµs  力 %+6.3fN  扁平度 %s  %s"
                      % (pulse, f, ("%.4f" % flat) if flat else "  --  ",
                         st or ""), flush=True)
                if f > 1.30:          # 快到位了，别把葡萄压坏
                    print("     力已到 1.30N，停止闭合")
                    break
                pulse -= args.step
    except KeyboardInterrupt:
        print("\n  中断")
    finally:
        try:
            if servo is not None:
                servo.set_pulse_us(pmax)
                time.sleep(0.3)
                servo.close()
        except Exception:  # noqa: BLE001
            pass
        vision.stop()
        try:
            if reader is not None:
                reader.close()
        except Exception:  # noqa: BLE001
            pass

    # ---- ③ 定位三档 ----
    print()
    print("=" * 74)
    print("标定结果")
    print("=" * 74)
    valid = [(p, f, fl) for p, f, fl, _ in rows if fl is not None]
    if len(valid) < 3:
        print("  ❌ 有效样本太少（扁平度没测到）。检查葡萄是否在画面里。")
        return 1

    free_ = [v for v in valid if abs(v[0] - pmax) < 1e-6]
    touch = [v for v in valid if v[1] > TOUCH_N]
    grip = [v for v in valid if v[1] > GRIP_N]

    flat_free = free_[0][2] if free_ else min(v[2] for v in valid)
    flat_touch = touch[0][2] if touch else None
    flat_grip = grip[0][2] if grip else None

    print(f"  自由（{pmax:.0f}µs）      扁平度 {flat_free:.4f}")
    if flat_touch is not None:
        print(f"  贴住（力首次 >{TOUCH_N}N） 扁平度 {flat_touch:.4f}")
    if flat_grip is not None:
        print(f"  夹紧（力首次 >{GRIP_N}N） 扁平度 {flat_grip:.4f}")

    if flat_touch is None or flat_grip is None:
        print()
        print("  ⚠️ 没测到完整的力上升过程：")
        print("     · 确认葡萄在两指之间")
        print("     · 确认传感器接好（打印的力应随闭合上升）")
        return 2

    t1 = (flat_free + flat_touch) / 2.0
    t2 = (flat_touch + flat_grip) / 2.0
    print()
    print("  ★ 建议阈值（取相邻两档中点，留最大余量）：")
    print(f"      FLAT_FREE_MAX  = {t1:.4f}")
    print(f"      FLAT_TOUCH_MAX = {t2:.4f}")
    print()
    print("  写入 apps/grape_state.py 的对应常量即可。")
    if t2 <= t1:
        print("  ⚠️ 但这次标定的顺序不正常（夹紧的扁平度没有更大）。")
        print("     可能原因：葡萄太硬、或力上升时夹爪把葡萄推走了。")
        print("     请检查葡萄是否真的在两指之间被夹住。")
    return 0


if __name__ == "__main__":
    sys.exit(main())