#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集采集工具 —— 用 K1 摄像头批量拍照，为分类模型准备训练数据。

为什么需要专门的工具
--------------------
训练图片**必须用部署时的同一个摄像头、同一套光照**拍。
用手机拍会在部署时因为领域差异（domain gap）导致识别率大幅下降 ——
这是小型数据集分类失败的头号原因。

实现方式
--------
直接复用已验证的 ``scripts/k1_camera_capture.py`` 里的 ``K1Camera``。
K1 的 MIPI 摄像头不能直接用 OpenCV 打开，必须由官方 cam-test 建立
ISP 管线；而"交互按键 dump"那种方式实测拿不到帧。
可用方式是**在 JSON 配置里指定 dump_one_frame**，让 cam-test 自动存一帧，
单次约 0.85 秒。

用法::

    # 采集"鸡蛋"类 150 张
    sudo python3 apps/capture_dataset.py --label egg --count 150

    # 自动连拍（每 2 秒一张，适合转着物体拍）
    sudo python3 apps/capture_dataset.py --label egg --count 150 --auto 2.0

    # 采集负样本（手、桌面、夹爪…用于训练 unknown 类）
    sudo python3 apps/capture_dataset.py --label other --count 100 --auto 2.0

拍摄要点
--------
    · 背景要有变化（白桌、木桌、手拿、不同角度）
    · 距离 30~60cm（实际工作距离）
    · 光照与演示时一致（开室内灯）
    · 每张图只放一个物体
    · 用 --auto 转动物体，能快速拿到多角度样本
"""
from __future__ import annotations

import argparse
import os
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


def main() -> int:
    ap = argparse.ArgumentParser(description="数据集采集")
    ap.add_argument("--label", required=True,
                    help="类别名（egg / grape / pingpong / other）")
    ap.add_argument("--count", type=int, default=150, help="要采集多少张")
    ap.add_argument("--out", default=None,
                    help="输出目录，默认 datasets/<label>")
    ap.add_argument("--auto", type=float, default=0.0,
                    help="自动连拍间隔（秒）。0 = 每张都要按回车")
    ap.add_argument("--start-index", type=int, default=None,
                    help="起始编号（续拍时用）。默认自动接着已有文件编号")
    ap.add_argument("--quality", type=int, default=92, help="JPEG 质量")
    args = ap.parse_args()

    out_dir = args.out or os.path.join(_ROOT, "datasets", args.label)
    os.makedirs(out_dir, exist_ok=True)

    try:
        import cv2
    except ImportError:
        print("❌ 需要 opencv： sudo apt install -y python3-opencv")
        return 1

    try:
        from k1_camera_capture import K1Camera
    except ImportError as exc:
        print(f"❌ 无法导入 K1Camera：{exc}")
        print("   确认 scripts/k1_camera_capture.py 存在")
        return 1

    # 自动推算起始编号，避免覆盖已有文件
    existing = sorted(f for f in os.listdir(out_dir) if f.endswith(".jpg"))
    if args.start_index is not None:
        idx = args.start_index
    elif existing:
        try:
            idx = int(existing[-1].rsplit("_", 1)[1].split(".")[0]) + 1
        except (IndexError, ValueError):
            idx = len(existing)
    else:
        idx = 0

    print("=" * 70)
    print(f"数据集采集 —— 类别 {args.label}")
    print("=" * 70)
    print(f"  输出目录  {out_dir}")
    print(f"  已有图片  {len(existing)} 张")
    print(f"  本次目标  {args.count} 张（从 {idx:04d} 开始编号）")
    mode_desc = (f"自动连拍 {args.auto:.1f}s/张" if args.auto > 0
                 else "手动（每张按回车）")
    print(f"  模式      {mode_desc}")
    print()
    print("拍摄要点：背景多变、距离 30~60cm、光照与演示一致、每张只放一个物体")
    print()

    need_sudo = hasattr(os, "geteuid") and os.geteuid() != 0
    if need_sudo:
        print("⚠️ 当前不是 root，K1Camera 会为每次拍摄调用 sudo cam-test")
        print()

    cam = K1Camera(use_sudo=need_sudo)

    # 先试拍一张，确认摄像头可用
    print("试拍一张，确认摄像头可用...")
    try:
        frame = cam.capture()
    except Exception as exc:  # noqa: BLE001
        print(f"❌ 试拍失败：{exc}")
        print()
        print("排查：")
        print("   1. 摄像头排线是否插好")
        print("   2. 先用官方工具确认： sudo cam-test /usr/share/camera_json/csi1_camera_auto.json")
        print("   3. 检查 /usr/share/camera_json/csi1_camera_auto.json 是否存在")
        return 1
    print(f"✅ 试拍成功，图像尺寸 {frame.shape[1]}x{frame.shape[0]}，"
          f"耗时 {cam.last_duration:.2f}s")
    print()

    saved = 0
    try:
        while saved < args.count:
            if args.auto > 0:
                print(f"  [{idx:04d}] {args.auto:.1f}s 后拍摄... ", end="")
                sys.stdout.flush()
                time.sleep(args.auto)
            else:
                try:
                    line = input(f"  [{idx:04d}] 摆好后按回车拍摄"
                                 f"（输入 q 结束）... ")
                except EOFError:
                    break
                if line.strip().lower() == "q":
                    break

            try:
                frame = cam.capture()
            except Exception as exc:  # noqa: BLE001
                print(f"❌ 拍摄失败：{exc}")
                continue

            name = f"{args.label}_{idx:04d}.jpg"
            path = os.path.join(out_dir, name)
            ok = cv2.imwrite(path, frame,
                             [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
            if not ok:
                print("❌ 写文件失败")
                continue

            saved += 1
            idx += 1
            size_kb = os.path.getsize(path) / 1024
            print(f"✅ {name}   {size_kb:.0f}KB   （{saved}/{args.count}，"
                  f"耗时 {cam.last_duration:.2f}s）")

    except KeyboardInterrupt:
        print("\n中断")

    print()
    print("=" * 70)
    total = len([f for f in os.listdir(out_dir) if f.endswith(".jpg")])
    print(f"本次采集 {saved} 张，『{args.label}』类现在共 {total} 张")
    print(f"目录：{out_dir}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())