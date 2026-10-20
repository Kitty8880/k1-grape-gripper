#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USB 麦克风检测与录音测试（K1 MusePi Pro）

目的：确认买的 USB 麦克风在 K1 上能不能用，并找出正确的设备编号，
      这个编号要写进 ASR 配置里。

用法::

    # 1. 只看设备列表，不录音（零风险，先跑这个）
    python3 apps/mic_test.py --list

    # 2. 自动挑一个麦克风，录 3 秒并存成 wav
    python3 apps/mic_test.py --record 3

    # 3. 指定设备录 5 秒
    python3 apps/mic_test.py --record 5 --device plughw:1,0

    # 4. 录完自动统计音量（判断是不是"录到了声音"还是"一片静音"）
    python3 apps/mic_test.py --record 3 --analyze

为什么用 arecord 而不是 Python 库：
    arecord 是 ALSA 官方工具，系统自带，不依赖 sounddevice/pyaudio
    这些在 RISC-V 上编译麻烦的库。先确认硬件能用，再上 Python 集成。
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import struct
import subprocess
import sys
import wave

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass


def run(cmd: list, timeout: float = 15.0) -> tuple:
    """执行命令，返回 (返回码, 标准输出, 标准错误)。"""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr
    except FileNotFoundError:
        return 127, "", f"命令不存在：{cmd[0]}"
    except subprocess.SubprocessError as exc:
        return 1, "", str(exc)


# ---------------------------------------------------------------------------
# 设备列举
# ---------------------------------------------------------------------------

def list_devices() -> int:
    print("=" * 76)
    print("麦克风 / 音频设备检测")
    print("=" * 76)

    # ---- arecord -l：录音设备列表（最关键）----
    print("\n【1】录音设备列表（arecord -l）  ← 最关键的一条")
    print("-" * 76)
    if shutil.which("arecord"):
        code, out, err = run(["arecord", "-l"])
        text = (out or "") + (err or "")
        print(text.rstrip() if text.strip() else "  （无输出）")
        cards = re.findall(r"card\s+(\d+):\s*(\S+)\s*\[([^\]]*)\]", text)
        if cards:
            print("\n  ★ 检测到的录音设备：")
            for num, short, full in cards:
                print(f"     card {num}: {short} [{full}]")
                print(f"       对应设备名： plughw:{num},0")
            print("\n  把上面某个 plughw:N,0 记下来，ASR 配置要用它。")
        else:
            print("\n  ⚠️ 没有检测到任何录音设备！")
            print("     如果你已经插了 USB 麦克风，排查：")
            print("       1. 换个 USB 口再试（优先 USB2.0 口）")
            print("       2. dmesg | tail -20  看内核有没有报错")
            print("       3. 某些 USB3.0 口对 UAC 设备兼容性差")
    else:
        print("  arecord 未安装 → sudo apt install -y alsa-utils")

    # ---- arecord -L：所有可用 PCM 设备 ----
    print("\n【2】所有可用音频设备（arecord -L 前 40 行）")
    print("-" * 76)
    if shutil.which("arecord"):
        code, out, err = run(["arecord", "-L"])
        lines = ((out or "") + (err or "")).splitlines()
        for line in lines[:40]:
            print("  " + line)
        if len(lines) > 40:
            print(f"  ...（共 {len(lines)} 行）")

    # ---- 声卡信息 ----
    print("\n【3】声卡详细信息（/proc/asound/cards）")
    print("-" * 76)
    if os.path.isfile("/proc/asound/cards"):
        with open("/proc/asound/cards", encoding="utf-8", errors="replace") as fh:
            print(fh.read().rstrip())
    else:
        print("  读不到 /proc/asound/cards")

    # ---- USB 层面 ----
    print("\n【4】USB 设备（看麦克风有没有被识别为 USB 设备）")
    print("-" * 76)
    if shutil.which("lsusb"):
        code, out, err = run(["lsusb"])
        print((out or "").rstrip() or "  （无输出）")
    else:
        print("  lsusb 未安装 → sudo apt install -y usbutils")

    # ---- 内核日志里的音频相关 ----
    print("\n【5】内核日志中的音频/USB 音频信息")
    print("-" * 76)
    code, out, err = run(["dmesg"])
    if code == 0 and out:
        keywords = ("audio", "snd", "usb", "mic", "uac")
        hits = [ln for ln in out.splitlines()
                if any(k in ln.lower() for k in keywords)]
        for line in hits[-25:]:
            print("  " + line)
        if not hits:
            print("  （没有相关日志）")
    else:
        # dmesg 需要 root
        code, out, err = run(["sudo", "-n", "dmesg"])
        if code == 0:
            print(out[-2000:])
        else:
            print("  dmesg 需要权限： sudo dmesg | grep -iE 'audio|snd|usb' | tail -25")

    # ---- 音量/静音状态 ----
    print("\n【6】当前音量与静音状态（amixer）")
    print("-" * 76)
    if shutil.which("amixer"):
        code, out, err = run(["amixer", "scontrols"])
        print(((out or "") + (err or "")).rstrip() or "  （无控件）")
        print("\n  ⚠️ 如果 Capture 是 [off] 或被静音，录音会一片安静：")
        print("     amixer sset Capture 80% unmute")
        print("     amixer -c 1 sset Mic 80% unmute     # 如果是 card 1")
    else:
        print("  amixer 未安装（在 alsa-utils 包里）")

    print("\n" + "=" * 76)
    print("下一步：python3 apps/mic_test.py --record 3 --analyze")
    print("=" * 76)
    return 0


# ---------------------------------------------------------------------------
# 录音
# ---------------------------------------------------------------------------

def auto_pick_device() -> str | None:
    """从 arecord -l 里自动挑一个录音设备。"""
    if not shutil.which("arecord"):
        return None
    code, out, err = run(["arecord", "-l"])
    text = (out or "") + (err or "")
    m = re.search(r"card\s+(\d+):", text)
    if m:
        return f"plughw:{m.group(1)},0"
    return None


def record(device: str, seconds: float, out_path: str,
           rate: int = 16000, channels: int = 1) -> int:
    """
    用 arecord 录音。

    默认 16000Hz 单声道 —— 这是 ASR（语音识别）的标准输入格式，
    直接录成模型要的样子，省掉后面重采样。
    """
    if not shutil.which("arecord"):
        print("❌ arecord 未安装 → sudo apt install -y alsa-utils")
        return 1

    cmd = [
        "arecord",
        "-D", device,
        "-f", "S16_LE",        # 16 位小端
        "-r", str(rate),       # 采样率
        "-c", str(channels),   # 声道数
        "-d", str(int(seconds)),  # 时长（秒）
        "-t", "wav",
        out_path,
    ]
    print(f"录音中… {seconds} 秒（对着麦克风说几句话）")
    print(f"  命令：{' '.join(cmd)}")
    sys.stdout.flush()

    code, out, err = run(cmd, timeout=seconds + 15)
    if code != 0:
        print(f"\n❌ 录音失败（返回码 {code}）")
        print((err or out or "").rstrip())
        print("\n排查：")
        print("  1. 设备名对不对？先跑 --list 看 plughw:N,0")
        print("  2. 麦克风被其他程序占用？（如桌面录音软件）")
        print("  3. 试试 plughw:1,0 换成 plughw:0,0 或 hw:1,0")
        return 2

    size = os.path.getsize(out_path) if os.path.isfile(out_path) else 0
    print(f"\n✅ 录音完成：{os.path.abspath(out_path)}  ({size} 字节)")
    if size < 1024:
        print("⚠️ 文件太小，可能没录到东西")
        return 3
    return 0


def analyze_wav(path: str) -> int:
    """
    分析 wav 文件的音量，判断是真录到声音还是静音。

    为什么需要：录音"成功"不代表录到了声音 —— 麦克风静音、
    增益为 0、或者录的是错误设备时，仍然会生成一个合法的（但是静音的）wav。
    这个检查能立刻区分这两种情况。
    """
    if not os.path.isfile(path):
        print(f"文件不存在：{path}")
        return 1

    try:
        with wave.open(path, "rb") as wf:
            n_channels = wf.getnchannels()
            width = wf.getsampwidth()
            rate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
    except (wave.Error, OSError) as exc:
        print(f"读 wav 失败：{exc}")
        return 1

    print("\n" + "=" * 76)
    print("录音质量分析")
    print("=" * 76)
    print(f"  声道数  : {n_channels}")
    print(f"  位深    : {width * 8} bit")
    print(f"  采样率  : {rate} Hz")
    print(f"  帧数    : {n_frames}  时长: {n_frames / max(1, rate):.2f} 秒")

    if width != 2:
        print(f"  （只支持 16 位样本的分析，当前 {width * 8} 位）")
        return 0

    samples = struct.unpack(f"<{len(raw) // 2}h", raw[: (len(raw) // 2) * 2])
    if not samples:
        print("  ⚠️ 没有采样数据")
        return 1

    peak = max(abs(s) for s in samples)
    rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
    peak_ratio = peak / 32768.0
    rms_ratio = rms / 32768.0

    print(f"  峰值    : {peak}  ({peak_ratio * 100:.1f}% 满量程)")
    print(f"  RMS     : {rms:.1f}  ({rms_ratio * 100:.2f}% 满量程)")
    # 大致换算成 dBFS
    import math
    if rms > 0:
        dbfs = 20 * math.log10(rms_ratio)
        print(f"  电平    : {dbfs:.1f} dBFS")

    print()
    if peak == 0:
        print("  ❌ 完全静音（所有采样都是 0）")
        print("     这是最典型的问题：设备选对了但没录到声音。排查：")
        print("       1. amixer sset Capture 80% unmute")
        print("       2. amixer -c 1 sset Mic 80% unmute   # card 号按实际情况改")
        print("       3. 确认麦克风没被物理静音（有些带开关）")
    elif peak_ratio < 0.01:
        print("  ⚠️ 信号非常弱（峰值 < 1% 满量程）")
        print("     可能：麦克风增益太低 / 离得太远 / 接错设备（录到了别的声卡）")
    elif peak_ratio > 0.99:
        print("  ⚠️ 削顶了（峰值接近满量程）")
        print("     声音会失真，ASR 识别率会下降。降低增益：")
        print("       amixer sset Capture 60%")
    else:
        print("  ✅ 电平正常，麦克风工作良好")

    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="USB 麦克风检测与录音测试（K1 MusePi Pro）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--list", action="store_true",
                   help="只列出音频设备，不录音")
    p.add_argument("--record", type=float, default=None, metavar="SEC",
                   help="录音秒数，例如 --record 3")
    p.add_argument("--device", default=None,
                   help="ALSA 设备名，如 plughw:1,0（默认自动挑选）")
    p.add_argument("--rate", type=int, default=16000,
                   help="采样率，默认 16000（ASR 标准）")
    p.add_argument("--channels", type=int, default=1,
                   help="声道数，默认 1（单声道，ASR 推荐）")
    p.add_argument("--out", default="mic_test.wav", help="输出 wav 路径")
    p.add_argument("--analyze", action="store_true",
                   help="录完自动分析电平（强烈建议开启）")
    p.add_argument("--analyze-only", default=None, metavar="WAV",
                   help="只分析已有的 wav 文件，不录音")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.analyze_only:
        return analyze_wav(args.analyze_only)

    if args.list or args.record is None:
        code = list_devices()
        if args.record is None:
            return code

    device = args.device or auto_pick_device()
    if not device:
        print("❌ 找不到录音设备。先跑 --list 看看。")
        return 1
    print(f"\n使用设备：{device}\n")

    code = record(device, args.record, args.out,
                  rate=args.rate, channels=args.channels)
    if code != 0:
        return code

    if args.analyze:
        analyze_wav(args.out)

    print("\n把 wav 拷到电脑上听一下（可选）：")
    print(f"  scp daiweikai@<板子IP>:~/{args.out} .")
    return 0


if __name__ == "__main__":
    sys.exit(main())
