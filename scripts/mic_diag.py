#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
麦克风诊断 —— 把每个输入设备都试一遍，找出哪个能录音。

为什么需要它
------------
`Device unavailable [Errno -9985]` 有三种完全不同的原因，
光看报错分不清：

    ① 设备索引变了          → 逐个试就能找到新的索引
    ② 设备被别的进程占用     → 所有索引都失败，但 arecord 能用
    ③ 采样率/声道不支持      → 大部分索引失败，换参数就行

本脚本逐个设备、逐个采样率地试，直接给出结论。

用法::

    python3 scripts/mic_diag.py
    python3 scripts/mic_diag.py --seconds 1.5
"""
from __future__ import annotations

import argparse
import os
import sys
import time

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description="麦克风诊断")
    ap.add_argument("--seconds", type=float, default=1.0,
                    help="每个组合试录多久")
    ap.add_argument("--rates", default="44100,48000,16000",
                    help="要试的采样率（逗号分隔）")
    ap.add_argument("--channels", default="1,2", help="要试的声道数")
    args = ap.parse_args()

    try:
        import numpy as np
        import pyaudio
    except ImportError as exc:
        print(f"❌ 需要 pyaudio 和 numpy：{exc}")
        return 1

    print("=" * 74)
    print("麦克风诊断")
    print("=" * 74)

    pa = pyaudio.PyAudio()

    # ---------- ① 列出所有输入设备 ----------
    print("\n[1/3] 输入设备列表")
    print("-" * 74)
    inputs = []
    n = pa.get_device_count()
    for i in range(n):
        try:
            d = pa.get_device_info_by_index(i)
        except Exception:  # noqa: BLE001
            continue
        ch = int(d.get("maxInputChannels", 0))
        if ch <= 0:
            continue
        inputs.append(i)
        tag = ""
        try:
            if i == pa.get_default_input_device_info()["index"]:
                tag = "   ← 系统默认输入"
        except Exception:  # noqa: BLE001
            pass
        print(f"  index {i:2d} | {d['name'][:48]:<48} | "
              f"{int(d['defaultSampleRate']):>6} Hz | {ch} ch{tag}")

    if not inputs:
        print("  ❌ 没有任何输入设备！")
        print("     检查： lsusb   看 USB 麦克风在不在")
        pa.terminate()
        return 1

    # ---------- ② 逐个组合试录音 ----------
    print(f"\n[2/3] 试录音（每个组合 {args.seconds:.1f} 秒）")
    print("-" * 74)
    rates = [int(x) for x in args.rates.split(",") if x.strip()]
    chans = [int(x) for x in args.channels.split(",") if x.strip()]

    working = []
    for idx in inputs:
        name = pa.get_device_info_by_index(idx)["name"][:36]
        for rate in rates:
            for ch in chans:
                label = f"index {idx} @ {rate}Hz x{ch}ch"
                try:
                    st = pa.open(format=pyaudio.paInt16, channels=ch,
                                 rate=rate, input=True,
                                 input_device_index=idx,
                                 frames_per_buffer=1024)
                except Exception as exc:  # noqa: BLE001
                    print(f"  ❌ {label:<34} {type(exc).__name__}: "
                          f"{str(exc)[:40]}")
                    continue

                # 录一小段，算电平和 RMS
                frames = []
                need = int(rate * args.seconds)
                got = 0
                try:
                    while got < need:
                        data = st.read(1024, exception_on_overflow=False)
                        frames.append(data)
                        got += len(data) // (2 * ch)
                except Exception as exc:  # noqa: BLE001
                    print(f"  ⚠️ {label:<34} 打开成功但读取失败：{exc}")
                    try:
                        st.close()
                    except Exception:  # noqa: BLE001
                        pass
                    continue
                finally:
                    try:
                        st.stop_stream()
                        st.close()
                    except Exception:  # noqa: BLE001
                        pass

                buf = b"".join(frames)
                a = np.frombuffer(buf, dtype=np.int16)
                if a.size == 0:
                    print(f"  ⚠️ {label:<34} 读到 0 字节")
                    continue
                peak = int(np.abs(a).max())
                rms = float(np.sqrt((a.astype(np.float64) ** 2).mean()))
                lvl = float(np.abs(a).mean())
                bar = "█" * int(min(28, lvl / 60))
                # 峰值打满 = ADC 削顶，通常意味着输入悬空（没接麦克风）
                # 或存在直流偏置 —— 这不是"拾到声音"，是假信号。
                clipped = peak >= 32000
                mark = "  ← ⚠️ 削顶，不是真麦克风" if clipped else ""
                print(f"  {'⚠️' if clipped else '✅'} {label:<34} "
                      f"峰值 {peak:>6}  RMS {rms:>7.0f}  "
                      f"电平 {lvl:>6.0f} {bar}{mark}")
                working.append((idx, rate, ch, name, lvl, peak))

    # ---------- ③ 结论 ----------
    print("\n" + "=" * 74)
    print("[3/3] 结论")
    print("=" * 74)
    if not working:
        print("  ❌ 所有组合都失败 —— 麦克风被别的进程占用，或硬件问题")
        print()
        print("  排查：")
        print("    1. 看有没有残留进程占着麦克风：")
        print("         ps aux | grep -E 'python3|arecord' | grep -v grep")
        print("         sudo pkill -f voice_grasp.py")
        print("    2. 绕开 PyAudio 直接试：")
        print("         arecord -d 3 -f S16_LE -r 44100 -c 1 /tmp/t.wav")
        print("    3. 看内核有没有报错： dmesg | tail -20")
        pa.terminate()
        return 1

    # 选设备的三条规则（顺序很重要）：
    #   ① 排除削顶的（峰值打满 = 输入悬空，不是真麦克风）
    #   ② 优先名字里带 USB 的（真实外接麦克风）
    #   ③ 在剩下的里选电平最高的
    #
    # 踩过的坑：一开始只按"电平最高"选，结果选到了板载声卡
    # （输入悬空 → 峰值 32767 → 平均电平虚高 351），完全误导。
    clean = [w for w in working if w[5] < 32000]
    usb = [w for w in clean if "usb" in w[3].lower()]
    pool = usb or clean or working
    best = max(pool, key=lambda t: t[4])

    print(f"  可用的组合共 {len(working)} 个"
          f"（其中 {len(working) - len(clean)} 个是削顶的假信号）：")
    for idx, rate, ch, name, lvl, peak in working:
        flag = "  ← 削顶，排除" if peak >= 32000 else ""
        print(f"    index {idx:2d}  {rate:>6}Hz  {ch}ch  "
              f"电平 {lvl:>6.0f}  峰值 {peak:>6}  {name[:30]}{flag}")
    print()
    if usb:
        print(f"  ★ 推荐用： --device {best[0]} --rate {best[1]}"
              f"   （USB 设备，电平 {best[4]:.0f}）")
    else:
        print(f"  ★ 推荐用： --device {best[0]} --rate {best[1]}"
              f"   ⚠️ 没找到 USB 麦克风，用的是 {best[3][:30]}")
    print()
    if best[4] < 100:
        print("  ⚠️ 但所有组合的电平都 < 100 ——")
        print("     说明设备能打开但**基本没拾到声音**。可能是：")
        print("       · 麦克风静音（alsamixer -c <卡号>，F4 切到 capture）")
        print("       · 麦克风插头/线材问题")
        print("       · 说话时离麦克风太远")
    else:
        print(f"  ✅ 电平 {best[4]:.0f} 说明确实拾到声音了")
    pa.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())