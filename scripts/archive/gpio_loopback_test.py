#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPIO 回环测试 —— 不用万用表，纯软件判断"引脚到底有没有输出"。

需要一根杜邦线，把「输出脚」和「读取脚」短接起来。

测试内容：
    第 1 步：验证「读取脚」的 GPIO 输入通路是否正常
              （用内部上拉/下拉自测，不需要外接线）
    第 2 步：驱动「输出脚」高低翻转，读「读取脚」能否跟上
    第 3 步：反向再测一遍（互换角色）

用法::

    # 引脚 32(GPIO34) 输出，引脚 36(GPIO35) 读取
    sudo python3 gpio_loopback_test.py --drive 34 --read 35

    # 只做第 1 步（不用接线，检查读取脚本身）
    sudo python3 gpio_loopback_test.py --read 35 --step 1

    # 扫描多个候选脚，找出哪些能做输入
    sudo python3 gpio_loopback_test.py --scan-input
"""
from __future__ import annotations

import argparse
import sys
import time

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass


# 官方引脚图里标注为纯 GPIO 的脚（只有 GPIO_xx 一个功能）
PLAIN_GPIO_CANDIDATES = [
    (34, "引脚 32"), (35, "引脚 36"), (37, "引脚 40"), (46, "引脚 38"),
    (33, "引脚 37"), (70, "引脚 7"), (71, "引脚 11"), (72, "引脚 13"),
    (73, "引脚 15"), (74, "引脚 12"), (91, "引脚 16"), (92, "引脚 18"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="GPIO 回环测试")
    ap.add_argument("--drive", type=int, default=34, help="输出脚（GPIO 编号）")
    ap.add_argument("--read", type=int, default=35, help="读取脚（GPIO 编号）")
    ap.add_argument("--chip", type=int, default=0)
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--scan-input", action="store_true",
                    help="扫描哪些脚能做输入")
    args = ap.parse_args()

    try:
        import lgpio
    except ImportError:
        print("缺少 lgpio：sudo apt install -y python3-lgpio")
        return 1

    def want(step: int) -> bool:
        return args.step == 0 or args.step == step

    print("=" * 68)
    print("GPIO 回环测试")
    print("=" * 68)

    try:
        h = lgpio.gpiochip_open(args.chip)
    except Exception as exc:  # noqa: BLE001
        print(f"❌ 打不开 gpiochip{args.chip}：{exc}")
        print("   试试： sudo chmod a+rw /dev/gpiochip0")
        return 1

    # ================= 扫描模式 =================
    if args.scan_input:
        print("\n扫描哪些引脚能做「输入」（用内部上拉/下拉自测）")
        print("内部上拉时读到 1、下拉时读到 0 → 该脚的 GPIO 输入通路正常\n")
        print(f"{'GPIO':>6}  {'物理脚':<8} {'上拉读数':>8} {'下拉读数':>8}  判定")
        print("-" * 60)
        for gpio, phys in PLAIN_GPIO_CANDIDATES:
            up = down = None
            try:
                lgpio.gpio_free(h, gpio)
            except Exception:  # noqa: BLE001
                pass
            try:
                lgpio.gpio_claim_input(h, gpio, lgpio.SET_PULL_UP)
                time.sleep(0.05)
                up = lgpio.gpio_read(h, gpio)
                lgpio.gpio_free(h, gpio)
                lgpio.gpio_claim_input(h, gpio, lgpio.SET_PULL_DOWN)
                time.sleep(0.05)
                down = lgpio.gpio_read(h, gpio)
                lgpio.gpio_free(h, gpio)
            except Exception as exc:  # noqa: BLE001
                print(f"{gpio:>6}  {phys:<8} {'--':>8} {'--':>8}  申请失败: {exc}")
                continue

            if up == 1 and down == 0:
                verdict = "✅ 可作输入"
            elif up == down:
                verdict = "❌ 读到固定值（未复用为 GPIO）"
            else:
                verdict = "⚠️ 异常"
            print(f"{gpio:>6}  {phys:<8} {up:>8} {down:>8}  {verdict}")

        lgpio.gpiochip_close(h)
        print("\n" + "=" * 68)
        return 0

    # ================= 常规回环 =================
    drive, read = args.drive, args.read
    print(f"输出脚： GPIO{drive}      读取脚： GPIO{read}")

    # ---- 第 1 步：验证读取脚的输入通路 ----
    if want(1):
        print()
        print("【第 1 步】验证读取脚的 GPIO 输入通路（不需要接线）")
        ok = True
        try:
            lgpio.gpio_free(h, read)
        except Exception:  # noqa: BLE001
            pass
        try:
            lgpio.gpio_claim_input(h, read, lgpio.SET_PULL_UP)
            time.sleep(0.05)
            up = lgpio.gpio_read(h, read)
            lgpio.gpio_free(h, read)
            lgpio.gpio_claim_input(h, read, lgpio.SET_PULL_DOWN)
            time.sleep(0.05)
            down = lgpio.gpio_read(h, read)
            lgpio.gpio_free(h, read)
        except Exception as exc:  # noqa: BLE001
            print(f"  ❌ 申请 GPIO{read} 失败：{exc}")
            lgpio.gpiochip_close(h)
            return 1

        print(f"  内部上拉时读到： {up}")
        print(f"  内部下拉时读到： {down}")
        if up == 1 and down == 0:
            print("  ✅ 读取脚的 GPIO 输入通路正常")
        else:
            print("  ❌ 读取脚读不到内部上拉/下拉 → 该脚未复用为 GPIO")
            print("     换一个脚试试： --scan-input 会列出可用的脚")
            ok = False

        if not ok:
            lgpio.gpiochip_close(h)
            print("\n" + "=" * 68)
            return 1

    # ---- 第 2 步：正面回环 ----
    if want(2):
        print()
        print("【第 2 步】请用杜邦线短接：")
        print(f"         GPIO{drive}（输出） ←→ GPIO{read}（读取）")
        input("  接好后按回车继续...")

        try:
            lgpio.gpio_free(h, drive)
            lgpio.gpio_free(h, read)
        except Exception:  # noqa: BLE001
            pass
        lgpio.gpio_claim_output(h, drive)
        lgpio.gpio_claim_input(h, read)

        print()
        matched = 0
        print(f"  {'次数':>4} {'输出':>6} {'读到':>6}  结果")
        print("  " + "-" * 34)
        for i in range(8):
            level = i % 2          # 0,1,0,1,...
            lgpio.gpio_write(h, drive, level)
            time.sleep(0.3)
            got = lgpio.gpio_read(h, read)
            same = (got == level)
            matched += int(same)
            print(f"  {i+1:>4} {level:>6} {got:>6}  {'✅' if same else '❌'}")

        print()
        if matched == 8:
            print("  🎉 全部匹配 → 输出脚和读取脚的 GPIO 通路都正常")
            print("     说明「引脚能输出」→ 扫不清问题就在舵机侧（电源/接线）")
        elif matched == 0:
            print("  ❌ 完全没跟上 → 输出脚没有真正输出")
            print("     可能是引脚复用没设成 GPIO")
        else:
            print(f"  ⚠️ 只匹配 {matched}/8 → 信号不稳定，检查杜邦线接触")

        lgpio.gpio_free(h, drive)
        lgpio.gpio_free(h, read)

    # ---- 第 3 步：反向回环 ----
    if want(3):
        print()
        print(f"【第 3 步】反向测试：GPIO{read} 输出，GPIO{drive} 读取")
        input("  把杜邦线两端互换（或者不用换，同一根线就行），按回车继续...")
        try:
            lgpio.gpio_free(h, drive)
            lgpio.gpio_free(h, read)
        except Exception:  # noqa: BLE001
            pass
        lgpio.gpio_claim_output(h, read)
        lgpio.gpio_claim_input(h, drive)

        matched = 0
        for i in range(8):
            level = i % 2
            lgpio.gpio_write(h, read, level)
            time.sleep(0.3)
            got = lgpio.gpio_read(h, drive)
            matched += int(got == level)
            print(f"  {i+1:>4} 输出={level}  读到={got}  "
                  f"{'✅' if got == level else '❌'}")

        print()
        print(f"  反向匹配 {matched}/8")
        lgpio.gpio_free(h, drive)
        lgpio.gpio_free(h, read)

    lgpio.gpiochip_close(h)
    print()
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())