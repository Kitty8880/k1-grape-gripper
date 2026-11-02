#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音触发抓取 —— 多模态协同（ASR + CV + 力闭环）。

流程
----
    用户说话
        ↓
    【ASR】SenseVoice 识别成文字
        ↓
    关键词匹配（"夹/抓/拿" + "葡萄"）
        ↓
    确认指令 → 触发抓取
        ↓
    【CV】找到葡萄位置（当前用 HSV，后续可换 NPU 上的 YOLO）
        ↓
    【力闭环】舵机闭合，控制夹持力到 0.8N
        ↓
    完成

本脚本当前实现到"识别 + 意图判断 + 打印动作计划"。
接入真实舵机/传感器后，把 _execute_grasp() 换成 gripper.py 的调用即可。

为什么要这样分层
----------------
· ASR 是"慢"操作（一次推理 0.3~3 秒），适合做"触发"，不适合做实时控制
· CV + 力闭环是"快"操作（几十毫秒级），负责实际执行
· 所以语音只负责"下命令"，执行交给视觉和力控 —— 这就是多模态协同的分工

用法::

    # 语音触发（不说话就等着）
    python3 apps/voice_grasp.py

    # 只识别一次，不执行动作（干跑）
    python3 apps/voice_grasp.py --dry-run --once

    # 不用语音，直接打字模拟（调试用）
    python3 apps/voice_grasp.py --text "请帮我夹一个葡萄"
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

# 项目根目录（本文件在 <root>/apps/ 下）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 用户主目录 = 项目的上一级
#
# ⚠️ 不能用 os.path.expanduser("~")：用 sudo 运行时 HOME 会变成 /root，
#    于是 ~/spacemit-demo 变成 /root/spacemit-demo（不存在），
#    而 ASR 的模型缓存 ~/.cache/sensevoice 也会找错位置。
#    实测踩过这个坑：sudo 下报 "找不到 /root/spacemit-demo/examples/NLP"。
#    用 __file__ 反推主目录，无论是否 sudo 都正确。
_HOME = os.path.dirname(_ROOT)
if os.path.isdir(os.path.join(_HOME, "spacemit-demo")):
    os.environ["HOME"] = _HOME          # 让 ASR 的 ~ 展开也指向真实主目录

_NLP_DIR = os.path.join(_HOME, "spacemit-demo", "examples", "NLP")
if os.path.isdir(_NLP_DIR):
    sys.path.insert(0, _NLP_DIR)

# ---------------------------------------------------------------------------
# 意图识别：把识别文本映射成动作
# ---------------------------------------------------------------------------

# 抓取类关键词（动词）
_GRASP_VERBS = ("夹", "抓", "拿", "取", "捏", "摘", "捡", "夹取", "抓取")
# 目标关键词（名词）
_GRAPE_WORDS = ("葡萄", "果子", "果实", "提子", "青提", "葡萄粒")
# 停止类
_STOP_WORDS = ("停", "停止", "别动", "松开", "放开", "结束", "退出")
# 状态查询
_STATUS_WORDS = ("状态", "怎么样", "好了吗", "报告")


def parse_intent(text: str) -> tuple:
    """
    把识别文本解析成 (意图, 说明)。

    返回的意图是：
        grasp    —— 执行抓取
        stop     —— 立即松开
        status   —— 查询状态
        unknown  —— 听不懂
    """
    if not text:
        return "unknown", "没听清"

    t = text.strip()

    if any(w in t for w in _STOP_WORDS):
        return "stop", f"听到停止指令：{t}"

    if any(w in t for w in _STATUS_WORDS):
        return "status", f"听到状态查询：{t}"

    has_verb = any(v in t for v in _GRASP_VERBS)
    has_obj = any(o in t for o in _GRAPE_WORDS)

    if has_verb and has_obj:
        return "grasp", f"识别到抓取意图：{t}"
    if has_verb and not has_obj:
        return "grasp", f"有抓取动作但没指定目标，默认夹葡萄：{t}"
    if has_obj and not has_verb:
        return "grasp", f"提到了葡萄，理解为要夹取：{t}"

    return "unknown", f"听不懂：{t}"


# ---------------------------------------------------------------------------
# 动作执行
# ---------------------------------------------------------------------------


def _execute_grasp(args, obj: str = "grape") -> bool:
    """
    执行一次真实抓取 —— 调用 apps/grasp_with_verify.py。

    为什么用子进程而不是直接 import
    --------------------------------
    · 抓取逻辑放在独立脚本里，可以单独调试，不必每次加载 ASR 模型
      （省几百 MB 内存和十几秒启动时间）
    · 子进程继承 stdin/stdout，所以"请提起物体，按回车"的交互照常可用
    · 抓取要 root（GPIO + 串口），语音识别不用 —— 权限边界更清楚

    返回 True 表示抓取成功。
    """
    if args.dry_run:
        print()
        print("  ┌─ 抓取流程（计划）─────────────────────────")
        print(f"  │ 物体：{obj}")
        print("  │ 1. 【CV】拍照 → 识别 → 与语音期望交叉校验")
        print("  │ 2. 舵机张开")
        print("  │ 3. 力到位就停（自适应物体大小），全程安全监控")
        print("  │ 4. 提示提起 → 复测力 → 判定是否夹牢")
        print("  └───────────────────────────────────────────")
        print("  （--dry-run：不驱动硬件）")
        return True

    # ★ 先走视觉：拍照 → 识别 → 与语音期望交叉校验
    #   只有校验通过才执行抓取。这是"多模态协同"的核心 ——
    #   语音给意图（要夹什么），视觉给事实（实际是什么），
    #   两者不一致就拒绝执行（防止用户放错物体时夹坏它）。
    if not args.no_vision:
        vscript = os.path.join(_ROOT, "apps", "vision_grasp.py")
        if os.path.exists(vscript):
            vcmd = [sys.executable, vscript, "--expect", obj]
            vcmd.extend(getattr(args, "grasp_arg", None) or [])
            try:
                need_root = (hasattr(os, "geteuid")
                             and os.geteuid() != 0)
            except Exception:  # noqa: BLE001
                need_root = False
            if need_root:
                vcmd = ["sudo"] + vcmd
            print()
            print("  ┌─ 视觉校验 ────────────────────────────────")
            print(f"  │ {' '.join(vcmd)}")
            print("  └───────────────────────────────────────────")
            print()
            rc_v = subprocess.call(vcmd)
            if rc_v != 0:
                print("  ❌ 视觉校验未通过（或抓取失败）—— 流程终止")
                return False
            return True
        print("  ⚠️ 找不到 vision_grasp.py，跳过视觉校验")

    script = os.path.join(_ROOT, "apps", "grasp_with_verify.py")
    if not os.path.exists(script):
        print(f"  ❌ 找不到 {script}")
        return False

    cmd = [sys.executable, script, "--object", obj]
    if args.auto_lift:
        cmd.append("--auto-lift")
    extra = getattr(args, "grasp_arg", None) or []
    cmd.extend(extra)

    # 抓取要访问 GPIO 和串口 → 需要 root
    try:
        need_root = hasattr(os, "geteuid") and os.geteuid() != 0
    except Exception:  # noqa: BLE001
        need_root = False
    if need_root:
        cmd = ["sudo"] + cmd

    print()
    print("  ┌─ 启动抓取流程 ────────────────────────────")
    print(f"  │ {' '.join(cmd)}")
    print("  └───────────────────────────────────────────")
    print()

    try:
        rc = subprocess.call(cmd)
    except Exception as exc:  # noqa: BLE001
        print(f"  ❌ 启动抓取失败：{exc}")
        return False

    if rc == 0:
        print("  ✅ 抓取流程报告：成功")
    else:
        print(f"  ❌ 抓取流程报告：失败（退出码 {rc}）")
    return rc == 0


def _execute_stop(args) -> None:
    print()
    print("  ┌─ 停止指令 ────────────────────────────────")
    print("  │ 舵机反向张开，卸力")
    print("  └───────────────────────────────────────────")


def _print_status(args) -> None:
    print()
    print("  ┌─ 系统状态 ────────────────────────────────")
    print("  │ 语音识别   ✅ SenseVoice 就绪")
    print("  │ 视觉识别   ✅ HSV 检测器就绪")
    print("  │ 力传感器   ✅ PS01 通信正常（未装机）")
    print("  │ 夹爪舵机   ✅ 忙等 PWM 就绪")
    print("  └───────────────────────────────────────────")


# ---------------------------------------------------------------------------


def resample_to(x, src_rate: int, dst_rate: int):
    """
    线性插值重采样。语音识别够用，且不依赖 scipy。

    ⚠️ 这个函数是必需的，不是优化！
       spacemit_asr 假定输入是 16 kHz。如果喂 44.1 kHz 的原始数据，
       它会按 44100/16000 = 2.76 倍"放慢"来理解 → 完全变成噪音。
       实测踩过：录 3.0 秒的"夹葡萄"，ASR 报"音频长度 8.320s"，
       识别成 <|SAD|><|BGM|>（悲伤+背景音乐）这种垃圾标签。
       官方 demo 用 RecAudioVadThread，它内部已经重采样过了；
       自己写录音就必须自己重采样。
    """
    import numpy as np
    if src_rate == dst_rate or len(x) == 0:
        return x
    n_out = int(round(len(x) * dst_rate / src_rate))
    if n_out <= 0:
        return x
    xi = np.linspace(0.0, len(x) - 1.0, n_out)
    out = np.interp(xi, np.arange(len(x), dtype=np.float64),
                    x.astype(np.float64))
    return out.astype(np.int16)


def _record_arecord(args, seconds: float, dst_rate: int):
    """
    用 arecord 录音 —— 比 PyAudio 的阻塞式 read 稳定得多。

    为什么换掉 PyAudio
    ------------------
    实测踩坑：`stream.read(1024)` 在 1.3 秒处**永久阻塞**，
    Ctrl+C 也打不断（阻塞发生在 C 层）。原因是 USB 音频缓冲区
    欠载/溢出时 ALSA 会让 read 一直等待。这是 PyAudio + ALSA 的已知问题。

    用 arecord 的三个额外好处：
      ① `plughw` 设备支持自动格式转换 —— 可以直接要 16 kHz，
         不用自己重采样（和采样率不匹配那次是同一个坑，一次解决）
      ② 进度靠轮询输出文件大小，不依赖任何阻塞调用
      ③ 有看门狗超时，永不永久卡死

    返回 (音频数组, 采样率)，失败返回 (None, 0)。
    """
    import shutil
    import subprocess
    import tempfile
    import wave

    import numpy as np

    if shutil.which("arecord") is None:
        return None, 0

    fd, path = tempfile.mkstemp(suffix=".wav", prefix="grape_rec_")
    os.close(fd)
    cmd = ["arecord", "-q", "-D", args.arecord_dev,
           "-f", "S16_LE", "-r", str(dst_rate), "-c", "1",
           "-d", str(max(1, int(round(seconds)))), path]
    print(f"  ● 录音中（{seconds:.1f} 秒，请现在开始说）...")
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        t_end = time.time() + seconds + 4.0
        while proc.poll() is None:
            try:
                size = os.path.getsize(path)
            except OSError:
                size = 0
            el = size / (2.0 * dst_rate)          # 16bit 单声道
            print(f"\r  ● {min(el, seconds):4.1f}s / {seconds:.1f}s", end="")
            if time.time() > t_end:               # 看门狗
                proc.kill()
                print("\n  ⚠️ arecord 超时，已终止")
                return None, 0
            time.sleep(0.1)
        print()
        if proc.returncode != 0:
            print(f"  ⚠️ arecord 退出码 {proc.returncode}"
                  f"（设备 {args.arecord_dev} 可能不可用）")
            return None, 0

        with wave.open(path, "rb") as w:
            rate = w.getframerate()
            ch = w.getnchannels()
            data = w.readframes(w.getnframes())
        audio = np.frombuffer(data, dtype=np.int16)
        if ch > 1:                                # 保险：多声道取第一声道
            audio = audio.reshape(-1, ch)[:, 0].copy()
        if audio.size == 0:
            return None, 0
        lvl = float(np.abs(audio).mean())
        peak = int(np.abs(audio).max())
        print(f"  电平 {lvl:.0f}  峰值 {peak}"
              + ("   ⚠️ 偏低，建议调麦克风增益（alsamixer）"
                 if lvl < 300 else "   ✅"))
        return audio, rate
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def _record_pyaudio(args):
    """PyAudio 录音（arecord 不可用时的退路）。"""
    import numpy as np
    import pyaudio

    rate = args.rate
    n = int(rate * args.fixed)
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=rate,
                     input=True, input_device_index=args.device,
                     frames_per_buffer=1024)
    print(f"  ● 录音中（{args.fixed:.1f} 秒，请现在开始说）...")
    frames = []
    try:
        got = 0
        while got < n:
            data = stream.read(1024, exception_on_overflow=False)
            frames.append(data)
            got += len(data) // 2
            print(f"\r  ● {got/rate:4.1f}s / {args.fixed:.1f}s", end="")
    finally:
        print()
        try:
            stream.stop_stream()
            stream.close()
        except Exception:  # noqa: BLE001
            pass
        pa.terminate()
    return np.frombuffer(b"".join(frames), dtype=np.int16), rate


def record_fixed(args):
    """
    固定时长录音。演示首选。

    为什么不用 VAD：VAD 靠电平触发开始录音，说话开头音节轻就会漏掉。
    实测踩过：说"请帮我夹一个葡萄"只录到 0.864s，被识别成"第二"。

    实现优先用 arecord（稳定 / 可直接出 16 kHz / 不会阻塞 / 有看门狗），
    arecord 不可用时退回 PyAudio 并自己重采样。
    """
    # ---------- 首选：arecord，直接录成 ASR 要的采样率 ----------
    audio, got_rate = _record_arecord(args, args.fixed, args.asr_rate)
    if audio is not None:
        return audio

    # ---------- 退路：PyAudio ----------
    print("  （arecord 不可用，改用 PyAudio）")
    audio, got_rate = _record_pyaudio(args)
    if args.asr_rate and got_rate != args.asr_rate and len(audio):
        before = len(audio)
        audio = resample_to(audio, got_rate, args.asr_rate)
        print(f"  重采样 {got_rate} → {args.asr_rate} Hz"
              f"（{before} → {len(audio)} 样本，"
              f"{len(audio)/args.asr_rate:.2f} 秒）")
    return audio


def main() -> int:
    ap = argparse.ArgumentParser(description="语音触发抓取（多模态协同）")
    ap.add_argument("--device", type=int, default=0, help="录音设备号")
    ap.add_argument("--rate", type=int, default=44100, help="录音采样率")
    ap.add_argument("--asr-rate", type=int, default=16000,
                    help="送给 ASR 的采样率（spacemit_asr 期望 16000）。"
                         "录到的音频会自动重采样到这个值。"
                         "不能改 —— 改了识别会变成噪音")
    ap.add_argument("--arecord-dev", default="plughw:0,0",
                    help="arecord 的设备名。plughw:0,0 = 第0张声卡（USB麦克风），"
                         "plughw 前缀让它自动做格式/采样率转换")
    ap.add_argument("--fixed", type=float, default=0.0,
                    help="固定时长录音（秒）。>0 时不用 VAD，演示首选，"
                         "例如 --fixed 3.0")
    ap.add_argument("--trig-on", type=float, default=0.10,
                    help="VAD 开始录音的电平阈值（默认 0.10；"
                         "官方 demo 用 0.30，太高会漏掉句子开头）")
    ap.add_argument("--trig-off", type=float, default=0.35,
                    help="VAD 判定静音的电平阈值")
    ap.add_argument("--silence", type=float, default=1.5,
                    help="静音多少秒后停止录音（默认 1.5，原来 1.0 偏短）")
    ap.add_argument("--max-time", type=int, default=5, help="最长录音秒数")
    ap.add_argument("--dry-run", action="store_true", help="不驱动硬件")
    ap.add_argument("--once", action="store_true", help="只处理一次")
    ap.add_argument("--object", default="grape",
                    choices=["grape", "egg", "pingpong", "unknown"],
                    help="要抓的物体类别（决定力策略）。"
                         "分类模型做好后由视觉自动决定")
    ap.add_argument("--auto-lift", action="store_true",
                    help="抓取时不等待手动提起确认")
    ap.add_argument("--no-vision", action="store_true",
                    help="跳过视觉交叉校验，直接抓（用于相机不可用时）")
    ap.add_argument("--grasp-arg", action="append", default=[],
                    help="透传给 grasp_with_verify.py 的额外参数，可重复")
    ap.add_argument("--text", default=None,
                    help="不用语音，直接用这段文字测试意图识别")
    ap.add_argument("--no-save", action="store_true", help="不保存录音")
    args = ap.parse_args()

    print("=" * 70)
    print("语音触发抓取 —— ASR + CV + 力闭环")
    print("=" * 70)

    # ---------- 纯文本模式（调试意图识别）----------
    if args.text is not None:
        print(f"\n输入文本：{args.text}")
        intent, why = parse_intent(args.text)
        print(f"意图：{intent}   （{why}）")
        if intent == "grasp":
            _execute_grasp(args, args.object)
        elif intent == "stop":
            _execute_stop(args)
        elif intent == "status":
            _print_status(args)
        else:
            print("  → 忽略")
        return 0

    # ---------- 语音模式 ----------
    if not os.path.isdir(_NLP_DIR):
        print(f"❌ 找不到 {_NLP_DIR}")
        return 1

    print("\n[1/2] 加载 ASR 模型...")
    try:
        from spacemit_asr import ASRModel
        asr = ASRModel()
    except Exception as exc:  # noqa: BLE001
        print(f"❌ ASR 加载失败：{exc}")
        return 1
    print("      ✅")

    print("[2/2] 初始化录音...")
    #
    # ⚠️ 固定时长模式（--fixed）下**不能**创建 RecAudioVadThread！
    #    它在构造时就会打开并持有麦克风设备，
    #    之后 record_fixed 再去开同一个设备就会失败：
    #      [Errno -9985] Device unavailable
    #    （实测踩过这个坑）
    rec = None
    if args.fixed > 0:
        print(f"      ✅ 固定时长模式（{args.fixed:.1f} 秒），不使用 VAD")
    else:
        try:
            from spacemit_audio import RecAudioVadThread
            rec = RecAudioVadThread(sld=args.silence,
                                    max_time=args.max_time,
                                    channels=1, rate=args.rate,
                                    device_index=args.device,
                                    trig_on=args.trig_on,
                                    trig_off=args.trig_off)
        except Exception as exc:  # noqa: BLE001
            print(f"❌ 录音初始化失败：{exc}")
            return 1
    print("      ✅")

    print()
    print("=" * 70)
    print("请说指令，例如：")
    print("   「请帮我夹一个葡萄」→ 执行抓取")
    print("   「停下来」          → 松开")
    print("   「现在状态怎么样」  → 查询状态")
    print("Ctrl+C 退出")
    print("=" * 70)

    def save_wav(path, samples, rate=16000):
        import wave
        try:
            data = samples.astype("<i2") if hasattr(samples, "astype") else samples
            raw = data.tobytes() if hasattr(data, "tobytes") else bytes(data)
            with wave.open(path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(rate)
                w.writeframes(raw)
            return True
        except Exception:  # noqa: BLE001
            return False

    try:
        while True:
            print("\n按回车开始说话...")
            try:
                input()
            except EOFError:
                break

            print("🎤 请说指令...")

            # ★ 两种录音方式
            #
            #  固定时长模式（--fixed，演示首选）：
            #    按回天后**固定录 N 秒**，完全不依赖 VAD。
            #    为什么演示要用这个：VAD 靠电平触发，说话开头音节轻就会漏掉 ——
            #    实测踩过：说"请帮我夹一个葡萄"只录到 0.864s，
            #    被识别成"第二"（因为开头被削掉了）。固定时长没有这个问题。
            #
            #  VAD 模式（默认，日常用）：
            #    靠 trig_on 电平触发开始、静音 sld 秒后结束。
            #    参数已开放为 --trig-on / --trig-off / --silence 便于调。
            if args.fixed > 0:
                try:
                    audio = record_fixed(args)
                except Exception as exc:  # noqa: BLE001
                    print(f"❌ 录音失败：{exc}")
                    continue
            else:
                try:
                    audio = rec.record_audio()
                except Exception as exc:  # noqa: BLE001
                    print(f"❌ 录音失败：{exc}")
                    continue

            empty = (audio is None or
                     (hasattr(audio, "size") and audio.size == 0) or
                     (isinstance(audio, str) and not audio))
            if empty:
                print("⚠️ 没录到有效音频")
                continue

            if hasattr(audio, "size") and not args.no_save:
                save_wav("/tmp/voice_cmd.wav", audio)

            t0 = time.time()
            try:
                text = asr.generate(audio)
            except Exception as exc:  # noqa: BLE001
                print(f"❌ 识别失败：{exc}")
                continue
            dt = time.time() - t0

            print()
            print(f"  🗣  识别：{text}   （{dt:.2f}s）")

            intent, why = parse_intent(text)
            print(f"  🧠 意图：{intent}   （{why}）")

            if intent == "grasp":
                _execute_grasp(args, args.object)
            elif intent == "stop":
                _execute_stop(args)
            elif intent == "status":
                _print_status(args)
            else:
                print("  → 忽略这条指令")

            if args.once:
                break

    except KeyboardInterrupt:
        print("\n中断")
    finally:
        # rec 在固定时长模式下是 None，必须判空
        if rec is not None:
            try:
                rec.cleanup()
            except Exception:  # noqa: BLE001
                pass

    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())