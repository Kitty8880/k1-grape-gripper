#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
抓取验证状态机 —— 夹取 → 提起 → 判力 → 成功/加力重试/放弃。

为什么需要它
------------
"夹住了"和"安全夹住了"是两回事。光把力控到目标值不够，还要回答：
    · 物体还在不在夹爪里？（滑没滑）
    · 力够不够吊住它？（会不会一提就掉）
    · 力会不会太大？（有没有压坏）

本脚本实现一条完整的判定链：闭合 → 保持 → 提起 → 复测 → 决策。

判据（核心）
------------
    提起前后各测一次平均力：
        力下降 < slip_threshold  → ✅ 夹牢了
        力下降 ≥ slip_threshold  → ⚠️ 在滑 → 加力重试（最多 N 次）
        加力后仍滑               → ❌ 放弃，报告原因

⚠️ 这个判据的精度受传感器噪声限制
--------------------------------
PS01 单通道有效更新率约 21Hz，噪声 σ：
    2.0s 窗口 → 0.22N
    4.0s 窗口 → 0.15N

"前后两次测量的差值"噪声 ≈ √2 × σ：
    4.0s 窗口 → 约 0.21N

所以 slip_threshold 默认取 0.5N（约 2.4σ）。
这意味着：**完全滑落（力掉到 0）能可靠检出；轻微滑动检不出来。**
对葡萄这种目标力只有 0.4N 的轻物体，判据偏紧 —— 这是传感器的物理限制，
不是软件问题。文档里必须如实说明。

用法::

    # 夹葡萄（力参数从 config.yaml 的 grasp_policy 读）
    sudo python3 apps/grasp_with_verify.py --object grape

    # 覆盖目标力
    sudo python3 apps/grasp_with_verify.py --object grape --target 0.3

    # 不驱动硬件，只打印状态流转（先验证逻辑）
    python3 apps/grasp_with_verify.py --object grape --dry-run

    # 跳过"请提起"的手动确认（自动流程里用）
    sudo python3 apps/grasp_with_verify.py --object grape --auto-lift
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

PD_CHANNEL_HZ = 160.0 / 3.0        # PD1 有效更新率约 53Hz（实测更低，约 21Hz）

# 默认力策略（config.yaml 里没有对应条目时用这个）
DEFAULT_POLICY = {
    "target_force_n": 0.2,
    "force_max_n": 0.6,
    "approach_pulse": 900.0,
    "grasp_pulse": 820.0,
}


# ---------------------------------------------------------------------------
# 力策略表
# ---------------------------------------------------------------------------

def load_policy(cfg: Config, name: str) -> dict:
    """
    从 config.yaml 的 grasp_policy.<name> 读力策略。

    ⚠️ 必须用 Config 的点号路径 getter，不能去读 cfg.data
       （Config 不把原始字典暴露成 .data，之前踩过这个坑，
        结果四类物体全都返回默认值 0.2/0.6/900）。
    """
    base = f"grasp_policy.{name}"
    return {
        "target_force_n": cfg.get_float(
            f"{base}.target_force_n", DEFAULT_POLICY["target_force_n"]),
        "force_max_n": cfg.get_float(
            f"{base}.force_max_n", DEFAULT_POLICY["force_max_n"]),
        "approach_pulse": cfg.get_float(
            f"{base}.approach_pulse", DEFAULT_POLICY["approach_pulse"]),
        "grasp_pulse": cfg.get_float(
            f"{base}.grasp_pulse", DEFAULT_POLICY["grasp_pulse"]),
    }


def print_policy_table(cfg: Config) -> None:
    names = ("grape", "egg", "pingpong", "unknown")
    print(f"{'物体':<12}{'目标力(N)':>10}{'硬上限(N)':>11}"
          f"{'预闭合(us)':>12}{'夹持(us)':>10}")
    print("-" * 58)
    for n in names:
        p = load_policy(cfg, n)
        print(f"{n:<12}{p['target_force_n']:>10.2f}"
              f"{p['force_max_n']:>11.2f}{p['approach_pulse']:>12.0f}"
              f"{p['grasp_pulse']:>10.0f}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="抓取验证状态机")
    ap.add_argument("--config", default=None)
    ap.add_argument("--object", default="unknown",
                    choices=["grape", "egg", "pingpong", "unknown"],
                    help="物体类别（决定力策略）")
    ap.add_argument("--target", type=float, default=None, help="覆盖目标力(N)")
    ap.add_argument("--max-force", type=float, default=None, help="覆盖硬上限(N)")
    ap.add_argument("--stop-pulse", type=float, default=None,
                    help="覆盖位置下限(us)")
    ap.add_argument("--pre-pulse", type=float, default=None,
                    help="快速预闭合到哪个脉宽(us)，默认取策略的 approach_pulse")
    ap.add_argument("--probe-step", type=float, default=3.0,
                    help="精调阶段的每步脉宽（微秒）")
    ap.add_argument("--probe-step-fast", type=float, default=10.0,
                    help="探测阶段的每步脉宽（微秒）。测得快，用于逼近接触点")
    ap.add_argument("--win-fast", type=float, default=0.4,
                    help="探测阶段的测力窗口（秒）")
    ap.add_argument("--win-fine", type=float, default=0.6,
                    help="精调阶段的测力窗口（秒）")
    ap.add_argument("--contact-force", type=float, default=0.4,
                    help="判定「已接触」的力（牛顿）—— 超过它就转入精调阶段")
    ap.add_argument("--probe-settle", type=float, default=0.12,
                    help="每步后的稳定等待（秒）")
    ap.add_argument("--grasp-pulse", type=float, default=None,
                    help="最终夹持位置(us)。默认取策略的 grasp_pulse")
    ap.add_argument("--verify", type=float, default=3.0,
                    help="提起前的测力时长（秒）")
    ap.add_argument("--recheck", type=float, default=4.0,
                    help="提起后的测力时长（秒）")
    ap.add_argument("--slip", type=float, default=0.5,
                    help="判定滑移的力下降阈值（牛顿）")
    ap.add_argument("--min-force", type=float, default=0.0,
                    help="夹持力最低要求（牛顿）。低于它一律判失败，"
                         "不论力有没有下降。默认 0 = 不检查。"
                         "葡萄建议设 0.8 —— 实测 0.4~0.5N 虽然看起来稳定，"
                         "但摩擦力不足以吊住葡萄，提起来会滑落")
    ap.add_argument("--retries", type=int, default=2, help="滑移后的重试次数")
    ap.add_argument("--boost", type=float, default=1.2,
                    help="重试时目标力放大倍数")
    ap.add_argument("--tighten", type=float, default=10.0,
                    help="滑移重试时夹持位置收紧多少微秒（脉宽减小=更紧）")
    ap.add_argument("--auto-lift", action="store_true",
                    help="不等待手动确认（自动流程用）")
    ap.add_argument("--settle-check", type=float, default=0.8,
                    help="达到目标后等多少秒再复测一次（默认 0.8）。"
                         "复测力低于目标就继续闭合 —— 用来排除"
                         "「噪声尖峰导致提前停止」。设 0 关闭")
    ap.add_argument("--settle-tol", type=float, default=0.10,
                    help="复测允许的低于目标的容差（默认 0.10N），"
                         "低于 目标-容差 才判为噪声尖峰")
    ap.add_argument("--plateau-n", type=float, default=0.12,
                    help="判定「力进入平台期」的阈值（牛顿）："
                         "最近 3 步的力跨度小于它就认为舵机已堵转")
    ap.add_argument("--auto-start", action="store_true",
                    help="跳过开始前的回车确认，直接开始闭合。"
                         "产品模式/服务模式必须加 —— 否则 stdin 是 "
                         "/dev/null，input() 会抛 EOFError")
    ap.add_argument("--show-policy", action="store_true", help="只打印策略表")
    ap.add_argument("--dry-run", action="store_true", help="不驱动硬件")
    args = ap.parse_args()

    cfg = Config.load(args.config or default_config_path())

    if args.show_policy:
        print("=" * 48)
        print("力策略表（config.yaml → grasp_policy）")
        print("=" * 48)
        print_policy_table(cfg)
        return 0

    pol = load_policy(cfg, args.object)
    target = args.target if args.target is not None else pol["target_force_n"]
    max_force = (args.max_force if args.max_force is not None
                 else pol["force_max_n"])
    pmin = cfg.get_float("servo.pulse_min_us", 650.0)
    pmax = cfg.get_float("servo.pulse_max_us", 1450.0)
    # ⚠️ stop_pulse（预闭合的下沿）不能低于机械安全下限，
    #    否则打印出来的"位置下限"会误导人（之前硬编码 700，比 pmin=750 还小）。
    stop_pulse = (args.stop_pulse if args.stop_pulse is not None
                  else pmin)
    pre_pulse = (args.pre_pulse if args.pre_pulse is not None
                 else pol.get("approach_pulse", 900.0))
    raw_zero = cfg.get_float("sensor.calib.left.raw_zero", 683457.0)
    raw_span = cfg.get_float("sensor.calib.left.raw_span", 15640.0)
    f_span = cfg.get_float("sensor.calib.left.force_max_n", 10.0)
    k = raw_span / f_span if f_span else 1564.0
    pin = cfg.get_int("servo.gpio_pin", 49)

    # ★ 最深位置 = 机械安全下限（两只传感器不碰上即可），与物体无关
    #   物体差异只体现在"力目标"上，不体现在"能闭多深"上。
    grasp_pulse = (args.grasp_pulse if args.grasp_pulse is not None
                   else pol.get("grasp_pulse", pmin))
    grasp_pulse = max(pmin, grasp_pulse)

    print("=" * 74)
    print(f"抓取验证状态机 —— 目标物体：{args.object}")
    print("=" * 74)
    print(f"  力策略：目标 {target:.2f}N   硬上限 {max_force:.2f}N   "
          f"预闭合 {pre_pulse:.0f}us   位置下限 {stop_pulse:.0f}us")
    print(f"  判定：提起前后力下降 ≥ {args.slip:.2f}N 判为滑移，"
          f"最多重试 {args.retries} 次")
    print(f"  测力：提起前 {args.verify:.1f}s   提起后 {args.recheck:.1f}s")
    n_v = max(1.0, args.verify * PD_CHANNEL_HZ)
    n_r = max(1.0, args.recheck * PD_CHANNEL_HZ)
    print(f"  噪声估算：σ ≈ 0.4N（{args.verify:.1f}s） / "
          f"0.35N（{args.recheck:.1f}s），差值判据约 2.4σ")
    print()

    if args.dry_run:
        print("── 干跑（不驱动硬件）──")
        print()
        for st in ("CLOSE", "VERIFY", "PROMPT", "RECHECK", "DECIDE", "RELEASE"):
            print(f"  [{st}]", end="")
            if st == "CLOSE":
                print(f" 张开 → 快速预闭合到 {pre_pulse:.0f}us → "
                      f"力闭环到 {target:.2f}N")
            elif st == "VERIFY":
                print(f" 保持 {args.verify:.1f}s，测平均力 F1")
            elif st == "PROMPT":
                print(" 提示用户提起物体，等待回车")
            elif st == "RECHECK":
                print(f" 再测 {args.recheck:.1f}s，得 F2")
            elif st == "DECIDE":
                print(f" 若 F1-F2 < {args.slip:.2f}N → 成功；"
                      f"否则加力 {args.boost:.1f}× 重试")
            else:
                print(" 张开卸力")
        print()
        print("✅ 状态流转正常")
        return 0

    if cfg.simulate:
        print("⚠️ config.yaml 里 hardware.simulate = true，请改成 false")
        return 1

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
        在 window 秒内采样，返回 (力, σ, 独立样本数)。

        ⚠️ 必须用**中位数**，不能用均值！
           PS01 会周期性冒出一个大尖峰（实测 A +11,300 / B +6,800 计数）。
           均值会被它拉高约 1,900 计数 = 1.2N，导致：
             · "空载力"虚高到 +0.6N（之前几次运行的输出里就能看到）
             · 每一步的力都被整体抬高/压低，判据失真
           中位数对孤立尖峰不敏感，是这个传感器唯一可用的统计量。
           （步进测试改用中位数后，曲线才第一次变干净）
        """
        vals, distinct, last = [], 0, None
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
            return 0.0, 0.0, 0
        vals.sort()
        med = vals[len(vals) // 2]
        return ((med - raw_zero) / k,
                statistics.pstdev(vals) / k, distinct)

    def measure_both(window: float):
        """同时读两只传感器，返回 (A的力, B的力, 平均, σ)。用于诊断。"""
        va, vb = [], []
        t0 = time.time()
        while time.time() - t0 < window:
            a = reader.channel(1)      # 传感器 A 的 PD1
            b = reader.channel(2)      # 传感器 A 的 PD2
            if a is not None:
                va.append(float(a))
            if b is not None:
                vb.append(float(b))
            time.sleep(0.002)
        if len(va) < 5:
            return 0.0, 0.0, 0.0, 0.0
        va.sort()
        vb.sort()
        ma = va[len(va) // 2]
        mb = vb[len(vb) // 2] if vb else ma
        fa = (ma - raw_zero) / k
        fb = (mb - raw_zero) / k
        return fa, fb, 0.5 * (fa + fb), statistics.pstdev(va) / k

    pulse = pmax
    result = {"ok": False, "reason": "", "pulse": pmax, "force": 0.0,
              "attempts": 0}
    try:
        # ⚠️ 产品模式下 stdin 是 /dev/null，input() 必然抛 EOFError。
        #    这里必须能把"读不到回车"当成"直接开始"，
        #    否则语音/视觉链路走到最后一步就崩（实测踩过）。
        if args.auto_start:
            print("（--auto-start：跳过开始确认）")
        else:
            try:
                input("按回车开始（请把物体放在两指之间，"
                      "手放在电源开关上）...")
            except EOFError:
                print("（stdin 不可读 —— 按无人值守模式直接开始）")

        # ============ 1. CLOSE ============
        print(f"\n[CLOSE] 张开 → 快速预闭合到 {pre_pulse:.0f}us")
        servo.set_pulse_us(pmax)
        time.sleep(1.0)
        pulse = max(pre_pulse, stop_pulse)
        servo.set_pulse_us(pulse)
        time.sleep(1.0)

        f_base, _, _ = measure(2.5)
        # ⚠️ 空载力应该在 ±0.3N 以内。超过 0.5N 说明 config 里的 raw_zero 过期了
        #    （传感器装到夹指上后零点会变）。虽然脚本会自动减去它，
        #    但偏得太多会让小力信号淹没在残差里。
        print(f"        空载力 {f_base:+.3f}N")
        if abs(f_base) > 0.5:
            print(f"        ⚠️ 空载力偏离 0 较多 —— config 的 sensor.calib.left.raw_zero"
                  f"（{raw_zero:,.0f}）可能过期，建议重新 tare")

        attempt = 0
        cur_target = target
        while True:
            attempt += 1
            result["attempts"] = attempt
            cur_max = min(max_force * (args.boost ** (attempt - 1)), 12.0)
            print(f"\n─── 第 {attempt} 次尝试：目标 {cur_target:.2f}N，"
                  f"上限 {cur_max:.2f}N ───")

            # ---- ★ 力到位就停（自适应物体尺寸）★ ----
            #
            # 控制结构的演进（每次都踩了坑，记录在此）：
            #
            #  ① 力闭环收敛到目标值  → 被噪声骗（900us 两指全开也报"力已到目标"）
            #     原因：噪声 σ 0.5N > 容差 0.15N
            #  ② 位置为主走到 grasp_pulse → 只能夹一个尺寸
            #     实测：同一个 856us，夹住了 A 葡萄，换一颗不同大小的就不行
            #     原因：固定位置只能对应一个直径
            #  ③ 力到位就停（当前）   → 自适应尺寸
            #
            # 为什么③可行，而①不可行：
            #   **判"有没有跨过阈值"比判"等不等于目标值"稳健得多。**
            #   ①是双边容差，噪声从两侧都能骗它；
            #   ③是单边阈值 + 连续 2 次确认，噪声只能造成一次孤立假触发。
            #
            # 为什么③必须用（物理原因）：
            #   葡萄的力-脉宽曲线极陡：860→850us 这 10us 内平均力从 0.24N 跳到 0.92N。
            #   陡 = 力对位置极其敏感 = **力是比位置灵敏得多的"到位"信号**。
            #
            # 控制器逻辑（两阶段，为了速度）：
            #   阶段① 快速探测：大步长 + 短窗口，力还很低时快速逼近接触点
            #   阶段② 慢速精调：小步长 + 中等窗口，力接近目标时精细控制
            #   力 ≥ 目标（连续 2 次）  → 夹紧了，停
            #   力 > 安全上限（连续 2 次）→ 立刻松开
            #   走到 deepest（最深允许）→ 停（说明力一直上不来，可能没夹到）
            #
            # 为什么要分两阶段：单阶段小步长太慢 ——
            #   从 940us 到 845us 用 2us 步长要走 48 步 × 1.25s = 60 秒（实测）。
            #   分两阶段后：70us 用 10us 步长（7 步）+ 15us 用 3us 步长（5 步）≈ 10 秒。
            t0 = time.time()
            converged = False
            reached_force = False
            over_count = 0
            hit_count = 0
            SIGMA_1S = 0.55
            deepest = max(pmin, min(grasp_pulse, pulse))   # 最深允许位置（安全兜底）
            #
            # ⚠️ deepest 必须钳到 servo.pulse_min_us（机械安全下限），
            #    不能比它更深 —— 那会让两只传感器对压。
            #
            #    踩过的坑：曾把 grasp_pulse 按物体设成 835us（葡萄），
            #    结果**小一点的葡萄夹不起来** —— 因为 835us 时力才 0.46N，
            #    而夹爪已经到"最深允许"停住了，没能继续闭合到力达标。
            #
            #    正确的分工：
            #      · 能闭多深  = 机械约束（750us，两只传感器不碰上即可）
            #      · 夹多大力  = 物体相关（葡萄 1.5N）
            #    所以 deepest 用机械下限，物体差异只体现在力的目标上。
            print(f"         力目标 {cur_target:.2f}N（连续 2 次确认）"
                  f"   最深允许 {deepest:.0f}us（机械安全下限）")
            print(f"         阶段① 探测：每步 {args.probe_step_fast:.0f}us / "
                  f"{args.win_fast:.1f}s 窗口，直到力 ≥ {args.contact_force:.2f}N")
            print(f"         阶段② 精调：每步 {args.probe_step:.0f}us / "
                  f"{args.win_fine:.1f}s 窗口，直到力 ≥ {cur_target:.2f}N")
            print(f"         {'时间':>7} {'脉宽':>7} {'原始计数':>11} "
                  f"{'力(N)':>8}  说明")

            phase = 1
            prev_pulse = None
            prev_f = None
            cur_step = args.probe_step_fast
            force_hist = []        # 最近几拍的力，用于判断"平台期/舵机堵转"
            stall_count = 0        # 连续多少次测力"数值完全不变"→ 数据流断裂
            while pulse > deepest:
                # ★ 自适应步长：按观察到的"力对位置的梯度"决定走多大一步
                #
                #   为什么需要：固定 3us 步长在"力上升很慢"的葡萄上会走很多步 ——
                #   实测踩过：从 840us 到 750us 走了 30 步、耗时 27 秒。
                #   梯度小说明还远，用大步长快速逼近；梯度大才用小步长防止过冲。
                #
                #   梯度 = Δ力 / Δ脉宽（N/us）
                #     实测参考：某一颗葡萄 860→850us 涨 1.3N → 0.13 N/us（很陡）
                #               另一颗     810→750us 涨 0.7N → 0.012 N/us（很缓）
                if phase == 1:
                    cur_step = args.probe_step_fast
                else:
                    rise = (f - prev_f) if prev_f is not None else 0.0

                    # ★ 刹车优先于预测
                    #
                    #   2026-11-03 实测踩坑：力曲线在接触前是平的（梯度≈0），
                    #   一接触就陡。用「上一步梯度」预测下一步，
                    #   在拐点处必然预测错 —— 平的那一步之后恰好就是最陡的地方。
                    #
                    #   实测：890us +1.024N → 880us +1.023N（梯度 0，选了最大步长）
                    #         → 870us +1.889N（一步跳 0.87N）
                    #         → 860us +2.258N（目标 1.20N，过冲到接近硬上限 2.50N）
                    #
                    #   所以先看「刚才涨得快不快」：涨得快就立刻小步刹车，
                    #   不去管梯度表算出什么。
                    if rise > 0.30:
                        cur_step = 1.5      # 刚才跳了一大截 → 立刻刹车
                    elif rise > 0.12:
                        cur_step = 2.0
                    elif prev_f is not None and prev_pulse is not None \
                            and prev_pulse != pulse:
                        grad = rise / abs(prev_pulse - pulse)
                        if grad > 0.05:
                            cur_step = 2.0
                        elif grad > 0.02:
                            cur_step = 4.0
                        elif grad > 0.005:
                            cur_step = 8.0
                        else:
                            cur_step = 12.0   # 很缓，大步推进
                    else:
                        cur_step = args.probe_step

                    # ★ 第二道保护：**离目标越近，步长越小**
                    #
                    #   光靠"上一步涨得快就刹车"不够 —— 因为刹车总是在
                    #   已经跳完之后才生效。真正要防的是"在离目标很近时
                    #   还敢跨一大步"。
                    #
                    #   实测 880us 时力已到 1.023N、目标 1.20N，
                    #   只剩 0.18N，却选了 12us 的步长 →
                    #   一步跨到 1.889N，直接过冲。
                    #
                    #   按剩余距离设上限：
                    #       剩 >1.0N : 最多 12us
                    #       剩 >0.6N : 最多 8us
                    #       剩 >0.3N : 最多 4us
                    #      剩 ≤0.3N : 最多 2us
                    remaining = cur_target - f
                    if remaining <= 0.3:
                        cur_step = min(cur_step, 2.0)
                    elif remaining <= 0.6:
                        cur_step = min(cur_step, 4.0)
                    elif remaining <= 1.0:
                        cur_step = min(cur_step, 8.0)

                    # ⚠️ 这里**只能**用绝对上下限，绝不掺 probe_step 当下限。
                    #    早期写成 max(args.probe_step, ...)（probe_step 默认 3.0），
                    #    导致「2µs 防过冲」那条分支永远不生效，等于没写。
                    cur_step = max(1.5, min(12.0, cur_step))

                win = args.win_fast if phase == 1 else args.win_fine
                pulse = max(deepest, pulse - cur_step)
                servo.set_pulse_us(pulse)
                time.sleep(args.probe_settle)

                f, _, n_distinct = measure(win)
                raw_med = f * k + raw_zero          # 还原成原始计数，便于比对
                f -= f_base
                result["force"] = f

                # ★★★ 传感器数据停滞检测
                #
                #   2026-11-03 实测踩到：19 步的原始计数全是 682,194，
                #   一个数都没变。真实读数永远在抖（σ≈4,199 计数），
                #   所以"完全不变"说明串口流断了（读线程死了 / USB 掉了 /
                #   端口没释放），**不是力为 0，也不是夹不住**。
                #
                #   不加这个检测的话，程序会把故障当成"力一直是 0"，
                #   最后报「力不足」—— 诊断方向完全错了，
                #   排查时会白白怀疑机械结构和葡萄大小。
                if n_distinct <= 1:
                    stall_count += 1
                else:
                    stall_count = 0
                if stall_count >= 3:
                    print(f"   ❌ 传感器数据停滞：连续 {stall_count} 次测力，"
                          f"原始计数一直是 {raw_med:.0f}（一次都没变）")
                    print(f"      真实读数应当持续抖动（σ≈4,199 计数）。"
                          f"完全不变 = 串口数据流已断。")
                    print(f"      排查顺序：")
                    print(f"        1) ls -l /dev/ttyUSB*        ← 串口还在吗")
                    print(f"        2) sudo pkill -f product_mode; sudo pkill -f grasp")
                    print(f"        3) sudo python3 scripts/ps01_dual_test.py --seconds 8")
                    print(f"        4) 仍无数据 → 拔插两个 USB 转接头")
                    result["reason"] = "sensor_stalled"
                    result["ok"] = False
                    # 立即张开卸力，不要带着未确认的夹持状态停在原地
                    try:
                        servo.set_pulse_us(pmax)
                        time.sleep(1.0)
                    except Exception:  # noqa: BLE001
                        pass
                    return 1

                sigma_win = SIGMA_1S / (win ** 0.5)
                limit_eff = max(cur_max, cur_target + 3.0 * sigma_win)
                if f > limit_eff:
                    over_count += 1
                else:
                    over_count = 0
                # ★ 单边阈值：力达到目标就计数（不再要求"接近"目标）
                if f >= cur_target:
                    hit_count += 1
                else:
                    hit_count = 0

                # 阶段①→② 切换：力开始上升就减速
                if phase == 1 and f >= args.contact_force:
                    phase = 2
                    note = "★ 检测到接触 → 转入精调"
                elif phase == 1:
                    note = "探测中"
                else:
                    note = f"精调 确认{hit_count}/2"

                print(f"         {time.time()-t0:6.1f}s {pulse:6.0f}us "
                      f"{raw_med:>11,.0f} {f:>+8.3f}  {note}")

                if over_count >= 2:
                    print(f"   ❌ 力连续 2 次超上限 {limit_eff:.2f}N，松开")
                    result["reason"] = "overforce"
                    servo.set_pulse_us(pmax)
                    time.sleep(1.5)
                    return 1
                if hit_count >= 2:
                    # ★★ 确认之后必须**复测**，防止噪声尖峰导致提前停
                    #
                    #   2026-11-03 用户实测反馈：夹爪在精调过程中仍有抖动，
                    #   有时力读数偶然冲到 1.2N 两次 → 判定"到位"停止闭合，
                    #   但稳态力其实只有 1.05N，结果夹不牢。
                    #
                    #   机制：真实力 1.05N 时，σ≈0.24N 的噪声要连出两次
                    #   ≥1.2N（约 +1.5σ，单次概率 ~20%）才能骗过
                    #   "连续 2 次"判据。听着概率低，但要走 10~20 步，
                    #   累积起来经常发生。
                    #
                    #   加确认次数只能降低概率，治不了根。真正的判据是
                    #   **「停下来之后力还在不在」** —— 所以确认后再等
                    #   settle_check 秒测一次，掉下去就继续闭合。
                    if args.settle_check > 0:
                        time.sleep(args.settle_check)
                        f_chk, _, _ = measure(args.win_fine)
                        f_chk -= f_base
                        if f_chk < cur_target - args.settle_tol:
                            print(f"   ⚠️ 复测 {f_chk:.2f}N < 目标 "
                                  f"{cur_target:.2f}N（差 "
                                  f"{cur_target-f_chk:.2f}N）—— 刚才那次是"
                                  f"噪声尖峰，继续闭合")
                            hit_count = 0
                            prev_f = f_chk
                            prev_pulse = pulse
                            continue

                    print(f"   ✅ 力达到目标 {cur_target:.2f}N"
                          f"（连续 2 次，复测 {f_chk:.2f}N 确认），"
                          f"停止闭合 —— 夹持力到位"
                          if args.settle_check > 0 else
                          f"   ✅ 力达到目标 {cur_target:.2f}N（连续 2 次），"
                          f"停止闭合 —— 夹持力到位")
                    converged = True
                    reached_force = True
                    break

                # 记录本次用于下一轮梯度计算
                prev_f = f
                prev_pulse = pulse
                force_hist.append(f)
                if len(force_hist) > 6:
                    force_hist.pop(0)

            if not reached_force:
                if pulse <= deepest:
                    # ★ 走到最深允许位置时，**先判断这是不是舵机堵转**
                    #
                    #   为什么这个区分很关键：
                    #     "脉宽 → 开口"的映射只在**空载**时成立。
                    #     葡萄挡住夹爪时，MG996R 扭矩不够就推不动，
                    #     实际夹爪停在葡萄表面，而命令脉宽一直在减小。
                    #     此时力会进入**平台期**（脉宽再减，力也不涨）。
                    #
                    #   实测数据（2026-11-02）：
                    #     780us +0.748N → 770us +0.706N → 760us +0.717N
                    #     → 750us +0.758N      脉宽走了 30us，力只变 0.05N
                    #   这不是"力不足"，是**已经到舵机的物理极限**。
                    #
                    #   所以判据改成：
                    #     力进入平台期 且 力 ≥ 最低要求  → 判定夹住（这就是最大夹持力）
                    #     力进入平台期 但 力 < 最低要求  → 判定太弱
                    min_ok = max(args.min_force, cur_target * 0.5)
                    plateau = (len(force_hist) >= 3 and
                               (max(force_hist[-3:]) -
                                min(force_hist[-3:])) < args.plateau_n)
                    spread = (max(force_hist[-3:]) - min(force_hist[-3:])
                              if len(force_hist) >= 3 else 999.0)

                    if plateau:
                        print(f"   ⚠️ 力进入平台期（最近 3 步变化仅 "
                              f"{spread:.2f}N）→ **舵机已堵转**，"
                              f"这已是最大夹持力")
                        print(f"      当前力 {f:.2f}N（目标 {cur_target:.2f}N）")
                        if f >= min_ok:
                            print(f"   ✅ 力 {f:.2f}N ≥ 最低要求 {min_ok:.2f}N"
                                  f" —— 判定为已夹住")
                            print(f"      （目标 {cur_target:.2f}N 超出了舵机"
                                  f"能给的力，但夹持力足够吊住物体）")
                            converged = True
                            reached_force = True
                            result["note"] = "servo_stall"
                        else:
                            print(f"   ❌ 力不足（{f:.2f}N < {min_ok:.2f}N）")
                            print(f"      → 物体太小 / 不在两指之间 / 被挤出去了")
                    else:
                        print(f"   ⚠️ 到最深允许位置 {deepest:.0f}us，"
                              f"力 {f:.2f}N（目标 {cur_target:.2f}N），"
                              f"且力仍在变化（最近3步跨度 {spread:.2f}N）")
                        if f >= min_ok:
                            print(f"   ✅ 力 ≥ 最低要求 {min_ok:.2f}N "
                                  f"—— 判定为已夹住")
                            converged = True
                            reached_force = True
                            result["note"] = "hit_mechanical_limit"
                        else:
                            print(f"   ❌ 力不足（{f:.2f}N < {min_ok:.2f}N）")
                converged = True

            if not converged:
                print("   ⚠️ 闭环未完成，但继续验证")

            # ============ 2. VERIFY ============
            print(f"\n[VERIFY] 保持 {args.verify:.1f}s，测平均力 F1")
            f1, sd1, nd1 = measure(args.verify)
            f1 -= f_base
            print(f"         F1 = {f1:+.3f}N  （σ {sd1:.2f}N，"
                  f"独立样本 {nd1}）")
            if f1 < 0.05:
                print("         ⚠️ 力接近 0 —— 可能没夹到东西")
                result["reason"] = "no_contact"

            # ============ 3. PROMPT ============
            print(f"\n[PROMPT] ★ 请用手把夹爪（带物体）提起来 3~5 cm")
            if args.auto_lift:
                print("         （--auto-lift：跳过确认，直接进入复测）")
                time.sleep(1.5)
            else:
                try:
                    input("         提起后按回车继续...")
                except EOFError:
                    pass

            # ============ 4. RECHECK ============
            print(f"\n[RECHECK] 再测 {args.recheck:.1f}s，得 F2")
            f2, sd2, nd2 = measure(args.recheck)
            f2 -= f_base
            print(f"          F2 = {f2:+.3f}N  （σ {sd2:.2f}N，"
                  f"独立样本 {nd2}）")

            # ============ 5. DECIDE ============
            drop = f1 - f2
            min_f = max(args.min_force, cur_target * 0.5)
            print(f"\n[DECIDE] 力下降 {drop:+.3f}N   "
                  f"（滑移阈值 {args.slip:.2f}N）")
            print(f"         夹持力 {f2:+.3f}N   "
                  f"（最低要求 {min_f:.2f}N）")

            # ★ 先查"力够不够大"，再查"力有没有掉"
            #   实测教训：0.47N 的夹持力在提起后确实会滑落，但"力下降"只有
            #   0.05N，看起来像稳定 —— 光看下降量会误判成功。
            #   物理原因：摩擦力 = 2μN，0.47N 时余量不足以吊住 12g 的葡萄。
            if f2 < min_f:
                print(f"         ❌ 夹持力不足（{f2:.3f}N < {min_f:.2f}N）"
                      f"—— 摩擦力吊不住物体")
                result["reason"] = "weak_grip"
                if attempt > args.retries:
                    print(f"         已重试 {args.retries} 次，放弃")
                    break
            elif drop < args.slip:
                print("         ✅ 夹牢了（力稳定且足够）")
                result.update(ok=True, reason="held", pulse=pulse, force=f2)
                break
            else:
                print("         ⚠️ 检测到滑移")
                result["reason"] = "slip"

            if attempt > args.retries:
                print(f"         已重试 {args.retries} 次，放弃")
                break
            # ★ 重试必须"夹得更紧"，不是"张开"！
            #   脉宽减小 = 夹爪更闭合。之前写成 pulse + 20（张开）是反的。
            #   同时把目标力也抬高一点，让"提前停止"的门槛更晚触发。
            grasp_pulse = max(stop_pulse, grasp_pulse - args.tighten)
            cur_target = min(cur_target * args.boost, cur_max * 0.8)
            print(f"         夹持位置收紧到 {grasp_pulse:.0f}us"
                  f"（-{args.tighten:.0f}us），目标力提到 {cur_target:.2f}N，重试")
            pulse = pre_pulse
            servo.set_pulse_us(pulse)
            time.sleep(0.6)

        # ============ 6. RELEASE ============
        print("\n[RELEASE] 张开卸力")
        servo.set_pulse_us(pmax)
        time.sleep(1.5)

    except KeyboardInterrupt:
        print("\n中断 —— 立即松开")
        try:
            servo.set_pulse_us(pmax)
            time.sleep(1.0)
        except Exception:  # noqa: BLE001
            pass
        result["reason"] = "interrupted"
    finally:
        for obj in (servo, reader):
            try:
                obj.close()
            except Exception:  # noqa: BLE001
                pass

    # ---------------- 总结 ----------------
    print()
    print("=" * 74)
    print("结果")
    print("=" * 74)
    print(f"  物体      {args.object}")
    print(f"  尝试次数  {result['attempts']}")
    print(f"  最终脉宽  {result['pulse']:.0f}us")
    print(f"  最终力    {result['force']:+.3f}N")
    print()
    if result["ok"]:
        print("  ✅ 抓取成功（提起后力稳定）")
    else:
        reasons = {
            "slip": "检测到滑移，加力后仍失败 —— 物体太重或太滑",
            "overforce": "力超过上限，已紧急松开",
            "no_contact": "没夹到东西",
            "weak_grip": "夹持力不足，摩擦力吊不住物体（提起来会滑落）",
            "interrupted": "用户中断",
        }
        print(f"  ❌ 抓取失败：{reasons.get(result['reason'], result['reason'])}")
    print("=" * 74)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())