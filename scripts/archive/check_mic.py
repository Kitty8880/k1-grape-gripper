#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USB 麦克风检查 —— 录音并分析，确认麦克风真的能收到声音。

为什么要分析而不是只看文件大小
------------------------------
arecord 录到"全静音"也会生成一个大小正常的 wav 文件，
所以必须看波形的幅度（RMS / 峰值）才能确认麦克风真的在工作。

用法::

    # 自动列出声卡并要求指定
    python3 scripts/check_mic.py

    # 指定声卡
    python3 scripts/check_mic.py --card 1 --seconds 3

    # 只列出设备，不录音
    python3 scripts/check_mic.py --list
"""
from __future__ import annotations

import argparse
import math
import os
import re
import struct
import subprocess
import sys
import wave

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass


def run(cmd: list) -> tuple:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError:
        return 127, "", f"命令不存在：{cmd[0]}"
    except Exception as exc:  # noqa: BLE001
        return 1, "", str(exc)


def list_cards() -> None:
    print("── /proc/asound/cards ──")
    try:
        with open("/proc/asound/cards", encoding="utf-8") as fh:
            print(fh.read().rstrip())
    except OSError as exc:
        print(f"  ❌ 读不到：{exc}")

    print()
    print("── arecord -l（录音设备）──")
    rc, out, err = run(["arecord", "-l"])
    if rc == 127:
        print("  ❌ 没装 alsa-utils，请运行： sudo apt install -y alsa-utils")
    else:
        print(out.rstrip() or "  （无录音设备）")

    print()
    print("── lsusb（USB 设备）──")
    rc, out, _ = run(["lsusb"])
    if rc == 0:
        for line in out.splitlines():
            if re.search(r"audio|mic|sound|webcam", line, re.I):
                print(f"  ★ {line}")
        else:
            print("  （上面只列出疑似音频设备；完整列表请单独运行 lsusb）")


def main() -> int:
    ap = argparse.ArgumentParser(description="USB 麦克风检查")
    ap.add_argument("--list", action="store_true", help="只列出设备")
    ap.add_argument("--card", type=int, default=None, help="声卡号（默认自动探测）")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--rate", type=int, default=16000)
    ap.add_argument("--out", default="/tmp/mic_check.wav")
    args = ap.parse_args()

    print("=" * 70)
    print("USB 麦克风检查")
    print("=" * 70)
    print()

    list_cards()

    if args.list:
        return 0

    print()
    print("=" * 70)

    # 自动挑一个录音设备
    card = args.card
    if card is None:
        rc, out, _ = run(["arecord", "-l"])
        m = re.search(r"card (\d+):", out or "")
        if m:
            card = int(m.group(1))
            print(f"自动选中声卡 {card}")
        else:
            print("❌ 没有找到录音设备。请检查：")
            print("   1. 麦克风是否插好（换个 USB 口试试）")
            print("   2. lsusb 里能不能看到它")
            return 1

    dev = f"plughw:{card},0"
    print()
    print(f"录音测试：设备 {dev}，{args.seconds:.0f} 秒，{args.rate} Hz 单声道")
    print("★ 请对着麦克风正常说话（要说满整个时长）")
    input("   准备好后按回车开始...")

    rc, out, err = run(["arecord", "-D", dev, "-f", "S16_LE",
                        "-r", str(args.rate), "-c", "1",
                        "-d", str(int(args.seconds)), args.out])
    if rc != 0:
        print(f"❌ 录音失败：{err or out}")
        print(f"   可以试试别的声卡号：--card 0 / --card 1 / --card 2")
        return 1

    if not os.path.exists(args.out):
        print("❌ 没有生成录音文件")
        return 1

    size = os.path.getsize(args.out)
    print(f"✅ 录音完成：{args.out}（{size:,} 字节）")

    # ---- 分析波形 ----
    try:
        with wave.open(args.out, "rb") as w:
            nch = w.getnchannels()
            width = w.getsampwidth()
            rate = w.getframerate()
            n = w.getnframes()
            data = w.readframes(n)
    except Exception as exc:  # noqa: BLE001
        print(f"❌ 读 wav 失败：{exc}")
        return 1

    if width != 2:
        print(f"⚠️ 采样位宽是 {width*8} bit，本脚本按 16 bit 分析")
        return 1

    samples = struct.unpack(f"<{len(data)//2}h", data)
    if not samples:
        print("❌ 没有采样数据")
        return 1

    peak = max(abs(x) for x in samples)
    rms = math.sqrt(sum(x * x for x in samples) / len(samples))
    # 削顶比例
    clipped = sum(1 for x in samples if abs(x) >= 32700) / len(samples) * 100

    print()
    print("── 波形分析 ──")
    print(f"  声道 {nch}    采样率 {rate} Hz    时长 {n/rate:.2f}s")
    print(f"  峰值 {peak:>6}  （满量程 32767）")
    print(f"  RMS  {rms:>6.1f}")
    print(f"  削顶比例 {clipped:.2f}%")
    print()

    # 判定
    if peak < 50:
        print("  ❌ 几乎全静音 —— 麦克风没在工作，或者录到了错误的设备")
        print("     排查：")
        print("       1. 换 USB 口")
        print("       2. 用 --card 指定别的声卡号")
        print("       3. 检查系统音量： alsamixer")
        return 1
    elif peak < 500:
        print("  ⚠️ 声音很小 —— 可能录到了远处的声音，或麦克风增益太低")
        print("     可以调增益： alsamixer")
    elif clipped > 5:
        print("  ⚠️ 削顶严重 —— 声音过大，调低增益： alsamixer")
    else:
        print("  ✅ 麦克风工作正常，电平合适")

    print()
    print("── 回放（如果板子接了音箱/HDMI 音频）──")
    rc, out, err = run(["aplay", "-D", f"plughw:{card},0", args.out])
    if rc == 0:
        print("  已回放，听听是不是你刚才说的话")
    else:
        print(f"  回放不可用（{err.strip()[:60]}），可把文件拷到电脑上听：")
        print(f"    scp daiweikai@<板子IP>:{args.out} .")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())