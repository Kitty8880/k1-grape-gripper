#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 读取 OV5647 摄像头测试。

K1 的摄像头通过 cam-test 配置后，数据从 /dev/video6 输出（NV12 1920x1080）。

用法（在 K1 上）：
    sudo python3 camera_test_opencv.py
    sudo python3 camera_test_opencv.py --device 6 --frames 5
"""
import argparse
import sys
import time

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

try:
    import cv2
    import numpy as np
except ImportError as exc:
    print(f"缺少依赖: {exc}")
    print("安装: sudo apt install -y python3-opencv python3-numpy")
    sys.exit(1)


def main() -> int:
    ap = argparse.ArgumentParser(description="OpenCV 摄像头测试")
    ap.add_argument("--device", type=int, default=6,
                    help="视频设备号，默认 6（/dev/video6）")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--frames", type=int, default=5,
                    help="抓几帧看看")
    ap.add_argument("--out", default="/tmp/opencv_shot.jpg")
    args = ap.parse_args()

    print("=" * 60)
    print(f"OpenCV 摄像头测试  /dev/video{args.device}")
    print("=" * 60)

    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print(f"❌ 打不开 /dev/video{args.device}")
        print("   可能原因：")
        print("   1. 需要先跑 cam-test 初始化 pipeline")
        print("   2. 设备号不对，试试 --device 1 / 2 / 3")
        return 2

    print(f"✅ 已打开 /dev/video{args.device}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"   实际分辨率: {actual_w}x{actual_h}")

    ok_frames = 0
    t0 = time.time()
    saved = None
    for i in range(args.frames):
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"   帧 {i+1}: ❌ 读取失败")
            continue
        ok_frames += 1
        h, w = frame.shape[:2]
        print(f"   帧 {i+1}: ✅ {w}x{h}  shape={frame.shape}")
        if saved is None:
            cv2.imwrite(args.out, frame)
            saved = args.out

    elapsed = time.time() - t0
    cap.release()

    print()
    print("=" * 60)
    if ok_frames:
        fps = ok_frames / max(0.001, elapsed)
        print(f"✅ 成功读取 {ok_frames}/{args.frames} 帧，约 {fps:.1f} fps")
        print(f"   已保存: {saved}")
    else:
        print("❌ 一帧都没读到")
        print("   建议：先跑 cam-test 初始化 pipeline：")
        print("   sudo cam-test /usr/share/camera_json/csi1_camera_auto.json")
    print("=" * 60)
    return 0 if ok_frames else 1


if __name__ == "__main__":
    sys.exit(main())
