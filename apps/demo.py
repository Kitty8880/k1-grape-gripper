#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键演示入口 —— 在板子上直接运行，不需要电脑远程连接。

为什么需要它
------------
开发阶段所有命令都是通过 SSH 从电脑发的，但**提交演示时不能这样**。
板子本身是完整的 Linux 桌面（Bianbu LXQt），插上显示器 + 键盘鼠标
就能独立运行，只需要一个"启动入口"把散落的命令收在一起。

本脚本提供：
    ① 自检——一次性检查所有硬件和配置，快速定位问题
    ② 菜单——选个数字就能跑对应的演示，不用记命令

用法::

    python3 apps/demo.py            # 菜单
    python3 apps/demo.py --check    # 只自检
    python3 apps/demo.py --run 1    # 直接跑第 1 项（无人值守演示用）

对应的 shell 入口：项目根目录的 run_demo.sh
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

# 菜单项：(编号, 标题, 命令, 是否需要 sudo)
MENU = [
    ("1", "完整演示：语音 → 视觉 → 抓取", "voice", True),
    ("2", "只测视觉：拍照 + 识别葡萄", "vision", True),
    ("3", "只测力控：夹取 + 提起验证", "grasp", True),
    ("4", "只测语音：识别一句话的意图", "voice_only", False),
    ("5", "标定：逐步下降找接触点", "step", True),
    ("6", "采样：只用摄像头拍数据集", "capture", True),
    ("7", "重新自检", "check", False),
]


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------

def check_all(verbose: bool = True) -> list:
    """
    检查所有前提条件，返回 [(名称, 是否通过, 说明)]。
    """
    results = []

    def add(name: str, ok: bool, msg: str = "") -> None:
        results.append((name, ok, msg))

    # ---- ① 串口（两只传感器） ----
    ports = sorted(glob.glob("/dev/ttyUSB*"))
    if len(ports) >= 2:
        add("串口", True, f"{len(ports)} 个：{', '.join(ports)}")
    elif len(ports) == 1:
        add("串口", False, f"只有 1 个（{ports[0]}）—— 检查两个 USB 转串口是否都插好")
    else:
        add("串口", False, "一个都没有 —— 两个 USB 转串口都没插好")

    # ---- ② 摄像头 ----
    cam_json = "/usr/share/camera_json/csi1_camera_auto.json"
    if os.path.exists(cam_json):
        add("摄像头配置", True, cam_json)
    else:
        add("摄像头配置", False,
            f"找不到 {cam_json} —— 先跑一次 sudo cam-test csi1_camera_detect.json")

    if os.path.exists("/usr/bin/cam-test"):
        add("cam-test", True, "/usr/bin/cam-test")
    else:
        add("cam-test", False, "找不到 /usr/bin/cam-test")

    # ---- ③ ASR 模型 ----
    asr_model = os.path.expanduser("~/.cache/sensevoice/model_quant_optimized.onnx")
    if os.path.exists(asr_model):
        sz = os.path.getsize(asr_model) / 1e6
        add("ASR 模型", True, f"{asr_model}（{sz:.0f} MB）")
    else:
        add("ASR 模型", False, f"找不到 {asr_model} —— 首次运行会自动下载 229MB")

    # ---- ④ 麦克风 ----
    has_mic = False
    mic_msg = ""
    try:
        import pyaudio
        pa = pyaudio.PyAudio()
        for i in range(pa.get_device_count()):
            d = pa.get_device_info_by_index(i)
            if int(d.get("maxInputChannels", 0)) > 0 and "usb" in \
                    str(d.get("name", "")).lower():
                has_mic = True
                mic_msg = f"index {i}: {d['name'][:40]}"
                break
        if not has_mic:
            for i in range(pa.get_device_count()):
                d = pa.get_device_info_by_index(i)
                if int(d.get("maxInputChannels", 0)) > 0:
                    mic_msg = (mic_msg or
                               f"找到输入设备 index {i}: {d['name'][:40]}"
                               "（但名字里没有 USB）")
        pa.terminate()
    except Exception as exc:  # noqa: BLE001
        mic_msg = f"pyaudio 不可用：{exc}"
    add("麦克风", has_mic, mic_msg or "没找到 USB 麦克风")

    # ---- ⑤ 配置 ----
    cfg_path = os.path.join(_ROOT, "config.yaml")
    if not os.path.exists(cfg_path):
        add("配置文件", False, f"找不到 {cfg_path}")
    else:
        try:
            from grape_gripper.config import Config
            c = Config.load(cfg_path)
            adc = c.get_str("sensor.adc", "?")
            pmin = c.get_float("servo.pulse_min_us", 0)
            sim = c.get_bool("hardware.simulate", False)
            ok = (adc.startswith("ps01") and pmin >= 700 and not sim)
            msg = f"adc={adc} pulse_min={pmin:.0f}us simulate={sim}"
            if sim:
                msg += "  ⚠️ simulate=true 会让你以为硬件没反应"
            if pmin < 700:
                msg += "  ⚠️ pulse_min 太低，会压坏传感器"
            add("配置文件", ok, msg)
        except Exception as exc:  # noqa: BLE001
            add("配置文件", False, f"解析失败：{exc}")

    # ---- ⑥ 数据集（分类模型用的，可选） ----
    ds = os.path.join(_ROOT, "datasets")
    if os.path.isdir(ds):
        classes = sorted(d for d in os.listdir(ds)
                         if os.path.isdir(os.path.join(ds, d)))
        if classes:
            counts = []
            for c in classes:
                n = len([f for f in os.listdir(os.path.join(ds, c))
                         if f.lower().endswith((".jpg", ".jpeg", ".png"))])
                counts.append(f"{c}={n}")
            add("数据集", True, "  ".join(counts))
        else:
            add("数据集", False, "目录存在但没有类别子目录")
    else:
        add("数据集", False, "还没采集（不影响抓取，但分类模型需要）")

    # ---- ⑦ 权限 ----
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        add("权限", False,
            "当前不是 root —— 舵机/串口需要 sudo，或者用 ./run_demo.sh（自动提权）")
    else:
        add("权限", True, "root")

    # ---- 输出 ----
    if verbose:
        print("=" * 74)
        print("系统自检")
        print("=" * 74)
        for name, ok, msg in results:
            mark = "✅" if ok else "❌"
            print(f"  {mark} {name:<12} {msg}")
        print("-" * 74)
        bad = [r for r in results if not r[1]]
        if bad:
            print(f"  ⚠️ {len(bad)} 项未通过：")
            for name, _, msg in bad:
                print(f"     · {name}：{msg}")
        else:
            print("  ✅ 全部通过，可以开始演示")
        print("=" * 74)

    return results


# ---------------------------------------------------------------------------
# 动作
# ---------------------------------------------------------------------------

def _run(cmd: list) -> int:
    print()
    print(f"$ {' '.join(cmd)}")
    print()
    try:
        return subprocess.call(cmd)
    except FileNotFoundError as exc:
        print(f"❌ 命令不存在：{exc}")
        return 127
    except KeyboardInterrupt:
        print("\n中断")
        return 130


def do_action(key: str, extra: list) -> int:
    if key in ("1", "voice"):
        return _run([sys.executable, os.path.join(_HERE, "voice_grasp.py"),
                     "--once", "--object", "grape", "--fixed", "3.0"] + extra)
    if key in ("2", "vision"):
        return _run([sys.executable, os.path.join(_HERE, "vision_grasp.py"),
                     "--expect", "grape", "--no-execute"] + extra)
    if key in ("3", "grasp"):
        return _run([sys.executable,
                     os.path.join(_HERE, "grasp_with_verify.py"),
                     "--object", "grape", "--min-force", "1.0"] + extra)
    if key in ("4", "voice_only"):
        return _run([sys.executable, os.path.join(_HERE, "voice_grasp.py"),
                     "--once", "--object", "grape", "--fixed", "3.0",
                     "--no-vision", "--dry-run"] + extra)
    if key in ("5", "step"):
        return _run([sys.executable,
                     os.path.join(_HERE, "servo_step_test.py")] + extra)
    if key in ("6", "capture"):
        return _run([sys.executable,
                     os.path.join(_HERE, "capture_dataset.py"),
                     "--label", "grape", "--count", "50",
                     "--auto", "2.0"] + extra)
    if key in ("7", "check"):
        check_all(verbose=True)
        return 0
    print(f"❌ 未知选项：{key}")
    return 2


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="一键演示入口")
    ap.add_argument("--check", action="store_true", help="只自检")
    ap.add_argument("--run", default=None,
                    help="直接跑某一项（1~7），适合无人值守演示")
    ap.add_argument("--no-check", action="store_true",
                    help="跳过启动自检")
    args, extra = ap.parse_known_args()

    if args.check:
        check_all(verbose=True)
        return 0

    if args.run:
        return do_action(args.run, extra)

    # ---------- 菜单模式 ----------
    print("=" * 74)
    print("葡萄抓取机器人 —— 演示入口")
    print("=" * 74)
    print()
    if not args.no_check:
        res = check_all(verbose=True)
        if any(not ok for _, ok, _ in res):
            print()
            print("  ⚠️ 有项目未通过，仍然可以继续（部分功能可能不可用）")

    while True:
        print()
        print("=" * 74)
        print("请选择要执行的操作：")
        for key, title, _, _ in MENU:
            print(f"  {key}) {title}")
        print("  0) 退出")
        print("=" * 74)
        try:
            choice = input("输入编号：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if choice in ("0", "q", ""):
            break
        if choice not in [m[0] for m in MENU]:
            print(f"  ❌ 无效选项：{choice}")
            continue

        rc = do_action(choice, [])
        print()
        print(f"（退出码 {rc}）")
        if choice == "7":
            continue

    print("退出。")
    return 0


if __name__ == "__main__":
    sys.exit(main())