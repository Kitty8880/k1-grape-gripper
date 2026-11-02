#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASR 语音识别测试 —— 适配本项目硬件的版本。

为什么要自己写而不是直接用 03_asr_demo.py
------------------------------------------
官方示例里有两处硬编码不适合我们的硬件：

    rec_audio = RecAudioVadThread(..., rate=48000, device_index=3, ...)
                                              ↑ 我们的麦克风是 44100
                                                          ↑ 应该是 0

官方示例的设备列表（`01_search_device.py` 实测）：

    索引 0: USB PnP Sound Device: Audio (hw:0,0)  ← ★ 我们的 USB 麦克风，44100Hz
    索引 1: snd-es8326 (hw:1,0)                   板载声卡
    索引 2: sysdefault
    索引 3: spdif                                 ← 示例写死的这个，是错的
    索引 4: pipewire
    索引 5: default

本脚本把设备号和采样率都做成参数，并额外支持 --once / --device。

依赖（已装）
------------
    sudo apt install -y libportaudio2 python3-pyaudio python3-scipy \
                        python3-numpy onnxruntime python3-spacemit-ort

VAD 模型会在首次运行时自动从 archive.spacemit.com 下载到 ~/.cache/sensevoice/。

用法::

    # 交互式：回车开始说话，说完自动识别
    python3 apps/asr_test.py

    # 只识别一次就退出
    python3 apps/asr_test.py --once

    # 指定设备和采样率
    python3 apps/asr_test.py --device 0 --rate 44100

    # 单次录音最长 8 秒
    python3 apps/asr_test.py --max-time 8
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

# 官方 demo 的包（spacemit_asr / spacemit_audio）不在本项目里，
# 需要把它的目录加进搜索路径。
#
# ⚠️ 不能用 os.path.expanduser("~")：用 sudo 运行时 HOME 会变成 /root，
#    于是去找 /root/spacemit-demo（不存在），
#    而且 ASR 的模型缓存 ~/.cache/sensevoice 也会指错地方（会重下 229MB）。
#    用 __file__ 反推主目录，无论是否 sudo 都正确。
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HOME = os.path.dirname(_ROOT)
if os.path.isdir(os.path.join(_HOME, "spacemit-demo")):
    os.environ["HOME"] = _HOME

_NLP_DIR = os.path.join(_HOME, "spacemit-demo", "examples", "NLP")
if not os.path.isdir(_NLP_DIR):
    print(f"❌ 找不到官方示例目录：{_NLP_DIR}")
    print("   请先克隆： git clone https://gitee.com/bianbu/spacemit-demo.git ~/spacemit-demo")
    sys.exit(1)
sys.path.insert(0, _NLP_DIR)


def save_wav(path: str, samples, rate: int = 16000) -> bool:
    """
    把录音数组存成 16kHz 单声道 wav，用于人工回放诊断。

    这是排查"识别结果不对"最直接的手段：
    如果回放听起来正常 → 问题在模型/参数
    如果回放听起来太快/太慢/是噪音 → 问题在录音或重采样
    """
    import wave
    try:
        data = samples
        if hasattr(data, "astype"):
            data = data.astype("<i2")
        raw = data.tobytes() if hasattr(data, "tobytes") else bytes(data)
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(raw)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️ 保存 wav 失败：{exc}")
        return False


def audio_level(samples) -> str:
    """返回录音的电平描述，用于判断音量是否合适。"""
    try:
        import numpy as np
        a = np.asarray(samples, dtype=np.float64)
        if a.size == 0:
            return "空"
        peak = float(np.max(np.abs(a)))
        rms = float(np.sqrt(np.mean(a * a)))
    except Exception:  # noqa: BLE001
        return "无法分析"
    verdict = ("❌ 几乎静音" if peak < 200 else
               "⚠️ 声音偏小" if peak < 2000 else
               "✅ 电平正常" if peak < 30000 else
               "⚠️ 削顶")
    return f"峰值 {peak:,.0f}  RMS {rms:,.0f}  {verdict}"


def main() -> int:
    ap = argparse.ArgumentParser(description="ASR 语音识别测试")
    ap.add_argument("--device", type=int, default=0,
                    help="PyAudio 设备号（默认 0 = USB 麦克风）")
    ap.add_argument("--rate", type=int, default=44100,
                    help="录音采样率（我们的 USB 麦克风默认 44100）")
    ap.add_argument("--max-time", type=int, default=5, help="单次最长录音秒数")
    ap.add_argument("--sld", type=float, default=1.0, help="静音判定长度（秒）")
    ap.add_argument("--once", action="store_true", help="只识别一次就退出")
    ap.add_argument("--list", action="store_true", help="只列设备")
    ap.add_argument("--save", default="/tmp/asr_rec.wav",
                    help="把每次录音存成 wav，便于回放诊断")
    ap.add_argument("--no-save", action="store_true", help="不保存录音")
    args = ap.parse_args()

    print("=" * 70)
    print("ASR 语音识别测试")
    print("=" * 70)

    # ---- 列设备 ----
    try:
        import pyaudio
        pa = pyaudio.PyAudio()
        print("\n可用的录音设备：")
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                mark = "  ★" if i == args.device else "   "
                print(f"{mark} 索引 {i}: {info['name']}  "
                      f"（{int(info['maxInputChannels'])} 通道, "
                      f"{int(info['defaultSampleRate'])} Hz）")
        pa.terminate()
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️ 列设备失败：{exc}")

    if args.list:
        return 0

    print(f"\n使用：设备 {args.device}，采样率 {args.rate} Hz，"
          f"最长 {args.max_time}s，静音阈值 {args.sld}s")

    # ---- 加载模型 ----
    print("\n[1/2] 加载 ASR 模型（首次运行会下载，约 229MB）...")
    t0 = time.time()
    try:
        from spacemit_asr import ASRModel
        asr = ASRModel()
    except Exception as exc:  # noqa: BLE001
        print(f"❌ ASR 模型加载失败：{exc}")
        print("   排查：")
        print("     1. 网络是否能访问 archive.spacemit.com")
        print("     2. 磁盘空间： df -h ~")
        return 1
    print(f"      ✅ 用时 {time.time()-t0:.1f}s")

    # ---- 录音线程 ----
    print("[2/2] 初始化录音（首次运行会下载 VAD 模型）...")
    try:
        from spacemit_audio import RecAudioVadThread
        rec = RecAudioVadThread(sld=args.sld, max_time=args.max_time,
                                channels=1, rate=args.rate,
                                device_index=args.device,
                                trig_on=0.30, trig_off=0.35)
    except Exception as exc:  # noqa: BLE001
        print(f"❌ 录音初始化失败：{exc}")
        print("   试试别的设备号： python3 apps/asr_test.py --list")
        return 1
    print("      ✅ 就绪")

    print()
    print("=" * 70)
    print("准备好了。每次按回车开始说话，说完停顿约 1 秒会自动结束。")
    print("Ctrl+C 退出")
    print("=" * 70)

    try:
        while True:
            print("\n按回车开始说话...")
            try:
                input()
            except EOFError:
                break

            print("🎤 请说话...")
            t0 = time.time()
            try:
                audio_ret = rec.record_audio()
            except Exception as exc:  # noqa: BLE001
                print(f"❌ 录音失败：{exc}")
                continue

            # ★ record_audio() 返回的是 numpy 数组（int16 音频数据）或文件路径，
            #   不能直接 `if not audio_ret` —— numpy 数组的真值判断会抛异常
            empty = False
            if audio_ret is None:
                empty = True
            elif hasattr(audio_ret, "size"):        # numpy 数组
                empty = audio_ret.size == 0
            elif isinstance(audio_ret, str):
                empty = not audio_ret

            if empty:
                print("⚠️ 没录到有效音频（可能是没说话，或音量太低）")
                if args.once:
                    return 1
                continue

            if hasattr(audio_ret, "size"):          # numpy 数组
                n = int(audio_ret.size)
                print(f"    录音完成（{time.time()-t0:.1f}s）："
                      f"{n} 采样点 ≈ {n/16000:.2f}s @16kHz，"
                      f"dtype={audio_ret.dtype}")
                print(f"    音频电平：{audio_level(audio_ret)}")
                if not args.no_save:
                    if save_wav(args.save, audio_ret):
                        print(f"    已保存：{args.save}"
                              f"（拷到电脑上听，判断录音是否正常）")
            else:
                print(f"    录音完成（{time.time()-t0:.1f}s）：{audio_ret}")

            t1 = time.time()
            try:
                text = asr.generate(audio_ret)
            except Exception as exc:  # noqa: BLE001
                print(f"❌ 识别失败：{exc}")
                continue
            dt = time.time() - t1

            print()
            print(f"  ┌─ 识别结果 ─────────────────────────")
            print(f"  │ {text}")
            print(f"  └─ 耗时 {dt:.2f}s")

            if args.once:
                break

    except KeyboardInterrupt:
        print("\n中断")
    finally:
        try:
            rec.cleanup()
        except Exception:  # noqa: BLE001
            pass

    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())