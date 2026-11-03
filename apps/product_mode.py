#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
产品模式 —— 上电即待命，说一句话就干活。全程不需要键盘、显示器、终端。

和开发脚本的区别
----------------
`voice_grasp.py` 是**调试用**的：每轮都要按回车、要指定 --object、
出错就退出。适合开发，不适合演示和交付。

本脚本是**产品用**的：
    · 无限循环常听，**完全不需要按任何键**
    · 只听「唤醒词 + 动作」，其他话一律忽略
    · 出错不退出（相机挂了、夹取失败都继续待命）
    · 全程无阻塞输入（input() 一次都不用）
    · 适合做成 systemd 服务开机自启

工作流程
--------
    上电 → 服务自动启动 → 常听
      ↓
    说「夹葡萄」
      ↓
    ASR 识别 → 意图=抓取
      ↓
    拍照 → 视觉确认是葡萄 → 力控夹取 → 力验证
      ↓
    回到常听（循环）

支持的指令
----------
    「夹葡萄」/「夹一下」/「开始」   → 执行抓取
    「停下」/「停」/「松开」         → 立刻松开夹爪
    「状态」                         → 打印状态
    其他                             → 忽略，继续听

用法::

    # 前台跑（调试用，能看到输出）
    sudo python3 apps/product_mode.py

    # 只测语音、不驱动硬件
    sudo python3 apps/product_mode.py --dry-run

    # 校准：打印识别到的每一句话（调识别率时用）
    sudo python3 apps/product_mode.py --echo

    # 装成开机自启的服务
    sudo bash deploy/install_service.sh
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

# 用户主目录 = 项目的上一级
#
# ⚠️ 不能用 os.path.expanduser("~")：产品模式用 sudo 运行，HOME 会变成 /root，
#    于是 ~/spacemit-demo 变成 /root/spacemit-demo（不存在），
#    而 ASR 模型缓存 ~/.cache/sensevoice 也会指错地方（会重下 229MB）。
#    用 __file__ 反推主目录，无论是否 sudo 都正确。
_HOME = os.path.dirname(_ROOT)
if os.path.isdir(os.path.join(_HOME, "spacemit-demo")):
    os.environ["HOME"] = _HOME          # 让 ASR 的 ~ 展开也指向真实主目录

_NLP_DIR = os.path.join(_HOME, "spacemit-demo", "examples", "NLP")
if os.path.isdir(_NLP_DIR):
    sys.path.insert(0, _NLP_DIR)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

from grape_gripper.config import Config, default_config_path   # noqa: E402

# ---------------------------------------------------------------------------
# 唤醒词表（口语里常见的几种说法都收进来）
# ---------------------------------------------------------------------------

GRASP_WORDS = (
    # ★ 只要句子里出现"葡萄"就夹 —— 语音识别不稳定，
    #   与其要求说出完整的"夹葡萄"，不如放宽到关键词匹配。
    #   实测"抓葡萄"能识别对，但"请帮我夹一个葡萄"常被识别成别的，
    #   而只要中间有"葡萄"两个字就够判断意图了。
    "葡萄", "提子",
    "夹", "抓", "拿", "开始", "来一个",
)
STOP_WORDS = ("停下", "停", "松开", "放开", "别夹", "停止")
STATUS_WORDS = ("状态", "怎么样", "好了吗", "进度")

_STRIP_TOKENS = (
    "<|zh|>", "<|en|>", "<|NEUTRAL|>", "<|HAPPY|>", "<|SAD|>",
    "<|ANGRY|>", "<|Speech|>", "<|BGM|>", "<|Applause|>", "<|Laughter|>",
    "<|Cry|>", "<|woitn|>", "<|withitn|>", "<|nospeech|>",
)


def clean_text(raw: str) -> str:
    """
    去掉 SenseVoice 输出的情感/事件标签、标点、以及 **emoji**。

    ⚠️ 2026-11-03 实测：模型有时把情感输出成 emoji（「葡萄😔」）而不是
       标签（`<|SAD|>`）。只过滤标签会让「葡萄😔」匹配不上关键词表，
       但这里用的是 `in` 判断，所以「葡萄😔」其实仍能命中"葡萄"。
       不过为了日志干净、也为了和 asr_bench 的判定口径一致，
       统一按「只保留中文和字母数字」处理。
    """
    import re
    t = raw
    for tok in _STRIP_TOKENS:
        t = t.replace(tok, "")
    return re.sub(r"[^\u4e00-\u9fff\u3400-\u4dbfA-Za-z0-9]", "", t)


def parse_intent(raw: str) -> tuple[str, str]:
    """返回 (intent, cleaned_text)。intent ∈ grasp/stop/status/none。"""
    t = clean_text(raw)
    if not t:
        return "none", ""
    for w in STOP_WORDS:
        if w in t:
            return "stop", t
    for w in STATUS_WORDS:
        if w in t:
            return "status", t
    for w in GRASP_WORDS:
        if w in t:
            return "grasp", t
    return "none", t


# ---------------------------------------------------------------------------

class ProductRunner:
    def __init__(self, args):
        self.args = args
        self.cfg = Config.load(args.config or default_config_path())
        self.asr = None
        self.rec = None
        self._stop = False
        self.runs = 0

    # ---------------- 生命周期 ----------------

    def setup(self) -> bool:
        print("=" * 70, flush=True)
        print(" 葡萄抓取机器人 —— 产品模式", flush=True)
        print("=" * 70, flush=True)

        # 1) ASR
        print("[1/2] 加载语音识别模型 ...", flush=True)
        try:
            from spacemit_asr import ASRModel
            self.asr = ASRModel()
            print("      ✅", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"      ❌ {exc}", flush=True)
            return False

        # 2) 常听录音器
        print("[2/2] 初始化麦克风（常听模式）...", flush=True)
        try:
            from spacemit_audio import RecAudioVadThread
            # trig_on 调得很低：说话开头音节轻也要能触发
            #   （官方 demo 用 0.30，实测会漏掉句首，识别成"第二"）
            self.rec = RecAudioVadThread(
                sld=self.args.silence,
                max_time=self.args.max_time,
                channels=1,
                rate=self.args.rate,
                device_index=self.args.device,
                trig_on=self.args.trig_on,
                trig_off=self.args.trig_off,
            )
            print("      ✅", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"      ❌ {exc}", flush=True)
            return False

        print(flush=True)
        print("=" * 70, flush=True)
        print(" 已进入待命状态 —— 不需要按任何键", flush=True)
        print("   说「夹葡萄」  → 执行抓取", flush=True)
        print("   说「停下」    → 立刻松开", flush=True)
        print("   说「状态」    → 查询", flush=True)
        print("=" * 70, flush=True)
        print(flush=True)
        return True

    def shutdown(self) -> None:
        print("\n正在退出 ...", flush=True)
        if self.rec is not None:
            try:
                self.rec.cleanup()
            except Exception:  # noqa: BLE001
                pass
        print("已退出。", flush=True)

    # ---------------- 动作 ----------------

    def _run(self, argv: list) -> int:
        """跑一个子脚本，等待完成。子脚本的键盘输入会被 /dev/null 替代。"""
        import subprocess
        cmd = [sys.executable] + argv
        if self.args.dry_run:
            print(f"      （--dry-run）跳过：{' '.join(cmd)}", flush=True)
            return 0
        print(f"      $ {' '.join(cmd)}", flush=True)
        try:
            # stdin 接 /dev/null —— 产品模式下不该有任何键盘交互
            with open(os.devnull, "r") as devnull:
                return subprocess.call(cmd, stdin=devnull)
        except Exception as exc:  # noqa: BLE001
            print(f"      ❌ 启动失败：{exc}", flush=True)
            return 1

    def do_grasp(self) -> None:
        print(f"\n▶ 第 {self.runs + 1} 次抓取", flush=True)
        script = os.path.join(_HERE, "vision_grasp.py")
        # ⚠️ 不要给 vision_grasp.py 传 --grasp-arg！
        #    它是 action="append"，argparse 会把后面的 --min-force 当成选项，
        #    报 "argument --grasp-arg: expected one argument"（踩过这个坑）。
        #    vision_grasp.py 内部已经给 grasp_with_verify.py 加了
        #    --min-force 1.2，不需要外部再传。
        #
        # ★ 每次都存调试图（原件/标注/掩码），识别失败时能直接看图定位。
        dbg = os.path.join(_ROOT, "debug")
        rc = self._run([script, "--expect", "grape", "--save-debug", dbg])
        self.runs += 1
        if rc == 0:
            print("  ✅ 抓取成功", flush=True)
        else:
            print(f"  ⚠️ 抓取未成功（退出码 {rc}），继续待命", flush=True)
            print(f"     调试图在 {dbg}/ （last_raw.jpg / last_mask.png）",
                  flush=True)

    def do_stop(self) -> None:
        print("\n▶ 松开夹爪", flush=True)
        script = os.path.join(_HERE, "grasp_demo.py")
        if os.path.exists(script):
            self._run([script, "--open"])
        else:
            print("      （找不到 grasp_demo.py，跳过）", flush=True)

    def do_status(self) -> None:
        print(f"\n▶ 状态：本次已执行 {self.runs} 次抓取", flush=True)

    # ---------------- 主循环 ----------------

    def loop(self) -> int:
        while not self._stop:
            try:
                audio = self.rec.record_audio()
            except KeyboardInterrupt:
                break
            except Exception as exc:  # noqa: BLE001
                print(f"  ⚠️ 录音异常：{exc}（1 秒后重试）", flush=True)
                time.sleep(1.0)
                continue

            empty = (audio is None or
                     (hasattr(audio, "size") and audio.size == 0) or
                     (isinstance(audio, str) and not audio))
            if empty:
                continue

            t0 = time.time()
            try:
                raw = self.asr.generate(audio)
            except Exception as exc:  # noqa: BLE001
                print(f"  ⚠️ 识别异常：{exc}", flush=True)
                continue
            dt = time.time() - t0

            if isinstance(raw, (list, tuple)):
                raw = raw[0] if raw else ""
            raw = str(raw)

            intent, text = parse_intent(raw)
            if self.args.echo or intent != "none":
                print(f"  🗣 听到：{text or '（空）'}"
                      f"   → 意图 {intent}   ({dt:.2f}s)", flush=True)

            if intent == "grasp":
                self.do_grasp()
            elif intent == "stop":
                self.do_stop()
            elif intent == "status":
                self.do_status()

        return 0


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="产品模式（常听，无需键盘）")
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", type=int, default=0, help="录音设备索引")
    ap.add_argument("--rate", type=int, default=44100, help="录音采样率")
    ap.add_argument("--trig-on", type=float, default=0.12,
                    help="VAD 触发阈值。太低会被环境噪声反复触发"
                         "（实测 0.05 时每 6 秒空录一次），"
                         "太高会漏掉句首（官方 demo 用 0.30 太迟钝）")
    ap.add_argument("--trig-off", type=float, default=0.30)
    ap.add_argument("--silence", type=float, default=1.5,
                    help="静音多少秒算一句话结束。越长越不会截断，"
                         "但每轮延迟越大")
    ap.add_argument("--max-time", type=int, default=4,
                    help="单句最长秒数。太长的话误触发要白白等很久")
    ap.add_argument("--dry-run", action="store_true",
                    help="不驱动硬件，只走识别和意图判断")
    ap.add_argument("--echo", action="store_true",
                    help="打印识别到的每一句话（调识别率时用）")
    args = ap.parse_args()

    runner = ProductRunner(args)

    def _sig(_signum, _frame):
        runner._stop = True
        print("\n收到退出信号 ...", flush=True)

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    if not runner.setup():
        return 1
    try:
        return runner.loop()
    finally:
        runner.shutdown()


if __name__ == "__main__":
    sys.exit(main())