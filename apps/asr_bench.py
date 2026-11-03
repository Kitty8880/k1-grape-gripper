#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音识别率测试 —— 自动计数、自动判对错、自动出报告。

为什么需要专门写一个脚本
------------------------
"识别率 90%" 这个数字要经得起评审追问。手记 20 次很容易出错：
说的内容不一致、记错了、算错分母。本脚本把流程固定下来：

    · 固定要说的句子（可配置）
    · 固定重复次数
    · 自动录音、自动识别、自动判对错
    · 自动算三个指标并出表

三个指标（都写进测试报告）
--------------------------
    ① 句准确率   识别文本与期望完全一致（去标点后）
    ② 字准确率   1 − 字错率(CER)，编辑距离 / 期望长度
                  —— 这是中文 ASR 的标准指标
    ③ 意图准确率 解析出的意图是否正确（抓取/停止/查询）
                  —— 这是**产品实际关心的**指标

    ⚠️ 三个数字可能差很多。举例：
       期望"夹葡萄" → 识别"夹 葡萄"，句准确率 0 但字准确率 100%
       期望"夹葡萄" → 识别"加葡萄"，字准确率 67% 但意图准确率 100%
       （因为意图只认关键词"葡萄"）

用法::

    # 测单个句子 20 次（最常用）
    sudo python3 apps/asr_bench.py --phrase 葡萄 --times 20

    # 测多个句子（脚本轮流提示）
    sudo python3 apps/asr_bench.py --times 20

    # 用 VAD 模式（更接近产品实际行为）
    sudo python3 apps/asr_bench.py --phrase 葡萄 --times 20 --mode vad

    # 结果存文件，便于贴进报告
    sudo python3 apps/asr_bench.py --phrase 葡萄 --times 20 --csv asr_result.csv
"""
from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import tempfile
import time
import wave

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

_HOME = os.path.dirname(_ROOT)
if os.path.isdir(os.path.join(_HOME, "spacemit-demo")):
    os.environ["HOME"] = _HOME
_NLP_DIR = os.path.join(_HOME, "spacemit-demo", "examples", "NLP")
if os.path.isdir(_NLP_DIR):
    sys.path.insert(0, _NLP_DIR)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

# 默认测试这几句 —— 覆盖不同的说法长度
DEFAULT_PHRASES = ["葡萄", "夹葡萄", "请帮我夹一个葡萄"]

# 与 product_mode.py 保持一致，保证测的就是产品实际用的解析逻辑
_STRIP_TOKENS = (
    "<|zh|>", "<|en|>", "<|NEUTRAL|>", "<|HAPPY|>", "<|SAD|>",
    "<|ANGRY|>", "<|EMO_UNKNOWN|>", "<|Speech|>", "<|BGM|>",
    "<|Applause|>", "<|Laughter|>", "<|Cry|>", "<|woitn|>",
    "<|withitn|>", "<|nospeech|>", "<|SAD|>",
)
GRASP_WORDS = ("葡萄", "提子", "夹", "抓", "拿", "开始", "来一个")
STOP_WORDS = ("停下", "停", "松开", "放开", "别夹", "停止")
STATUS_WORDS = ("状态", "怎么样", "好了吗", "进度")


def clean(raw: str) -> str:
    """
    把识别结果规整成可比较的纯文本。

    ⚠️ 2026-11-03 实测踩坑：SenseVoice 在某些配置下会把情感输出成
       **emoji**（如「葡萄😔」），而不是 `<|SAD|>` 这种标签。
       原实现只过滤标签，导致 20 次测试里有 13 次「葡萄😔」被误判为
       识别错误 —— 句准确率显示 15%，实际是 80%。
       **一个字符串清洗的疏漏，把成绩抹掉了一大半。**

    现在改为「只保留中文字符和字母数字」，一次性解决：
        emoji、标点、空格、标签残留、全角符号、零宽连接符
    """
    import re
    t = raw
    for tok in _STRIP_TOKENS:
        t = t.replace(tok, "")
    return re.sub(r"[^\u4e00-\u9fff\u3400-\u4dbfA-Za-z0-9]", "", t)


def parse_intent(text: str) -> str:
    for w in STOP_WORDS:
        if w in text:
            return "stop"
    for w in STATUS_WORDS:
        if w in text:
            return "status"
    for w in GRASP_WORDS:
        if w in text:
            return "grasp"
    return "none"


def levenshtein(a: str, b: str) -> int:
    """编辑距离（用于算字错率）。"""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1,          # 删除
                           cur[j - 1] + 1,        # 插入
                           prev[j - 1] + (ca != cb)))   # 替换
        prev = cur
    return prev[-1]


# ---------------------------------------------------------------------------
# 录音
# ---------------------------------------------------------------------------

def record_fixed(path: str, seconds: float, device: str,
                 rate: int = 16000) -> bool:
    """用 arecord 录固定时长 —— 比 PyAudio 可靠（后者会永久阻塞）。"""
    cmd = ["arecord", "-q", "-D", device, "-f", "S16_LE",
           "-r", str(rate), "-c", "1", "-d", str(int(round(seconds))), path]
    try:
        rc = subprocess.call(cmd, timeout=seconds + 8,
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        return rc == 0 and os.path.exists(path) and os.path.getsize(path) > 1000
    except subprocess.TimeoutExpired:
        return False


def wav_to_float(path: str):
    import numpy as np
    with wave.open(path, "rb") as wf:
        n = wf.getnframes()
        raw = wf.readframes(n)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="语音识别率测试")
    ap.add_argument("--phrase", action="append", default=None,
                    help="要测试的句子，可重复指定。不指定则用默认三句")
    ap.add_argument("--times", type=int, default=20,
                    help="每句重复次数（默认 20）")
    ap.add_argument("--mode", choices=["fixed", "vad"], default="fixed",
                    help="fixed=固定时长录音（测 ASR 本身，默认）；"
                         "vad=语音端点检测（更接近产品实际）")
    ap.add_argument("--seconds", type=float, default=3.0,
                    help="fixed 模式的录音时长")
    ap.add_argument("--device", default="plughw:0,0")
    ap.add_argument("--rate", type=int, default=16000)
    ap.add_argument("--countdown", type=float, default=3.0,
                    help="每次录音前的准备倒计时秒数")
    ap.add_argument("--expect-intent", default="grasp",
                    help="这些句子期望解析出的意图")
    ap.add_argument("--csv", default=None, help="结果另存为 CSV")
    args = ap.parse_args()

    phrases = args.phrase or DEFAULT_PHRASES
    total = len(phrases) * args.times

    # ---------- 加载 ASR ----------
    print("=" * 70)
    print("语音识别率测试")
    print("=" * 70)
    print(f"  测试句子：{len(phrases)} 句 × {args.times} 次 = {total} 次")
    for p in phrases:
        print(f"    · 「{p}」")
    print(f"  模式　　：{args.mode}"
          f"（{'固定 ' + str(args.seconds) + ' 秒录音' if args.mode == 'fixed' else 'VAD 自动端点'}）")
    print(f"  录音设备：{args.device}  {args.rate} Hz")
    print()

    print("加载 ASR 模型 ...")
    try:
        from spacemit_asr import ASRModel
        asr = ASRModel()
        print("  ✅")
    except Exception as exc:  # noqa: BLE001
        print(f"  ❌ {exc}")
        return 1

    # VAD 模式才需要录音线程
    rec = None
    if args.mode == "vad":
        try:
            from spacemit_audio import RecAudioVadThread
            rec = RecAudioVadThread(sld=1.5, max_time=6, channels=1,
                                    rate=44100, device_index=0,
                                    trig_on=0.12, trig_off=0.30)
            print("  ✅ VAD 就绪")
        except Exception as exc:  # noqa: BLE001
            print(f"  ❌ VAD 初始化失败：{exc}（改用 --mode fixed）")
            return 1

    print()
    print("=" * 70)
    print("说明：听到提示后请正常语速说出屏幕上的句子")
    print("     每句之间会自动停顿，不用按键")
    print("=" * 70)

    tmpdir = tempfile.mkdtemp(prefix="asrbench_")
    rows = []
    idx = 0

    try:
        for phrase in phrases:
            print()
            print(f"───── 测试句子：「{phrase}」─────")
            for k in range(args.times):
                idx += 1
                print()
                print(f"[{idx}/{total}] 请说：「{phrase}」")

                # 倒计时
                d = args.countdown
                while d > 0:
                    print(f"   {d:.0f} ...", end="", flush=True)
                    time.sleep(min(1.0, d))
                    d -= 1.0
                print("  🎤 开始说话")

                t0 = time.time()
                if args.mode == "fixed":
                    wav = os.path.join(tmpdir, f"a{idx}.wav")
                    ok = record_fixed(wav, args.seconds, args.device,
                                      args.rate)
                    if not ok:
                        print("   ❌ 录音失败，跳过")
                        rows.append({"idx": idx, "phrase": phrase,
                                     "heard": "", "exact": 0, "cer": 1.0,
                                     "intent_ok": 0, "ms": 0,
                                     "note": "录音失败"})
                        continue
                    audio = wav_to_float(wav)
                    rec_sec = len(audio) / args.rate
                else:
                    audio = rec.record_audio()
                    if audio is None or (hasattr(audio, "size")
                                         and audio.size == 0):
                        print("   ⚠️ 未录到音频，跳过")
                        rows.append({"idx": idx, "phrase": phrase,
                                     "heard": "", "exact": 0, "cer": 1.0,
                                     "intent_ok": 0, "ms": 0,
                                     "note": "无音频"})
                        continue
                    rec_sec = audio.size / 44100.0

                t1 = time.time()
                try:
                    raw = asr.generate(audio)
                except Exception as exc:  # noqa: BLE001
                    print(f"   ❌ 识别异常：{exc}")
                    continue
                infer_ms = (time.time() - t1) * 1000

                if isinstance(raw, (list, tuple)):
                    raw = raw[0] if raw else ""
                heard = clean(str(raw))
                want = clean(phrase)

                exact = int(heard == want)
                dist = levenshtein(want, heard)
                cer = dist / max(1, len(want))
                intent = parse_intent(heard)
                intent_ok = int(intent == args.expect_intent)

                mark = "✅" if exact else ("🔸" if intent_ok else "❌")
                print(f"   识别：{heard or '（空）'}   "
                      f"{mark}  字错率 {cer*100:.0f}%   "
                      f"意图={intent}   推理 {infer_ms:.0f}ms")

                rows.append({"idx": idx, "phrase": phrase, "heard": heard,
                             "exact": exact, "cer": cer,
                             "intent_ok": intent_ok,
                             "ms": infer_ms, "note": "",
                             "rec_sec": rec_sec})
                time.sleep(0.6)

    except KeyboardInterrupt:
        print("\n\n用户中断，用已采集的数据出报告。")
    finally:
        if rec is not None:
            try:
                rec.cleanup()
            except Exception:  # noqa: BLE001
                pass

    # ---------- 报告 ----------
    valid = [r for r in rows if r["heard"]]
    print()
    print()
    print("=" * 70)
    print("测试结果")
    print("=" * 70)
    if not valid:
        print("没有有效样本。")
        return 1

    n = len(valid)
    exact_rate = sum(r["exact"] for r in valid) / n
    cer_avg = statistics.mean(r["cer"] for r in valid)
    intent_rate = sum(r["intent_ok"] for r in valid) / n
    lat = [r["ms"] for r in valid]

    print(f"  有效样本：{n} / {len(rows)} 次")
    print()
    print(f"  ① 句准确率　　{exact_rate*100:5.1f}%   "
          f"（{sum(r['exact'] for r in valid)}/{n}）")
    print(f"  ② 字准确率　　{(1-cer_avg)*100:5.1f}%   "
          f"（平均字错率 {cer_avg*100:.1f}%）")
    print(f"  ③ 意图准确率　{intent_rate*100:5.1f}%   "
          f"（{sum(r['intent_ok'] for r in valid)}/{n}）")
    print()
    if lat:
        print(f"  推理耗时：平均 {statistics.mean(lat):.0f}ms　"
              f"中位 {statistics.median(lat):.0f}ms　"
              f"最小 {min(lat):.0f}ms　最大 {max(lat):.0f}ms")

    # 按句子拆分
    print()
    print("  按句子拆分")
    print("  " + "-" * 60)
    for phrase in phrases:
        sub = [r for r in valid if r["phrase"] == phrase]
        if not sub:
            continue
        print(f"   「{phrase}」  {len(sub)} 次   "
              f"句准确 {sum(r['exact'] for r in sub)/len(sub)*100:5.1f}%   "
              f"意图 {sum(r['intent_ok'] for r in sub)/len(sub)*100:5.1f}%")

    # 错误样本
    bad = [r for r in valid if not r["exact"]]
    if bad:
        print()
        print(f"  识别错误的样本（{len(bad)} 条，供分析）")
        print("  " + "-" * 60)
        for r in bad[:15]:
            print(f"   期望「{r['phrase']}」 → 识别「{r['heard']}」"
                  f"   字错率 {r['cer']*100:.0f}%")

    # 评分对照
    print()
    print("  " + "-" * 60)
    ref = max(exact_rate, intent_rate)
    if ref >= 0.90:
        grade = "≥90% → 对应评分档「优秀」(9 分)"
    elif ref >= 0.80:
        grade = "≥80% → 对应评分档「良好」(6 分)"
    elif ref >= 0.70:
        grade = "≥70% → 对应评分档「一般」(3 分)"
    else:
        grade = "<70% → 0 分"
    print(f"  比赛评分参考：{grade}")
    print(f"  （比赛看的是「ASR 识别准确率」，建议报告里"
          f"同时给出句准确率和意图准确率）")

    # CSV
    if args.csv:
        try:
            with open(args.csv, "w", encoding="utf-8-sig") as fh:
                fh.write("序号,期望,识别,句准确,字错率,意图正确,推理ms,备注\n")
                for r in rows:
                    fh.write(f"{r['idx']},{r['phrase']},{r['heard']},"
                             f"{r['exact']},{r['cer']:.3f},"
                             f"{r['intent_ok']},{r['ms']:.0f},"
                             f"{r.get('note','')}\n")
            print(f"\n  明细已存：{args.csv}")
        except OSError as exc:
            print(f"\n  ⚠️ 写 CSV 失败：{exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())