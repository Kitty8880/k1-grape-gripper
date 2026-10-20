#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K1 MIPI 摄像头采集模块（基于官方 cam-test 工具）。

原理
----
K1 的 MIPI 摄像头（OV5647）不能像 USB 摄像头那样直接用 OpenCV 打开，
必须由官方 SDK 建立 pipeline（sensor -> CSI -> ISP -> CPP）后才能出图。

本模块通过调用官方的 ``cam-test`` 工具完成一次采集：
    1. 用精简的 JSON 配置（只采 8 帧、保存第 3 帧）启动 cam-test
    2. cam-test 把 NV12 帧写到 /tmp/cpp0_output_<w>x<h>_s<stride>.nv12
    3. 读该文件，转成 OpenCV 的 BGR 格式返回

实测：单次采集耗时约 0.85 秒。

用法
----
命令行::

    sudo python3 k1_camera_capture.py --out /tmp/shot.jpg
    sudo python3 k1_camera_capture.py --out /tmp/shot.jpg --repeat 3

作为模块::

    from k1_camera_capture import K1Camera
    cam = K1Camera()
    frame = cam.capture()      # numpy BGR 数组
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass


DEFAULT_JSON = "/tmp/quick.json"
DEFAULT_OUT = "/tmp/cpp0_output_1920x1080_s1920.nv12"
DEFAULT_SRC_JSON = "/usr/share/camera_json/csi1_camera_auto.json"


def make_quick_json(dst: str = DEFAULT_JSON,
                    src: str = DEFAULT_SRC_JSON,
                    test_frame: int = 8,
                    dump_frame: int = 3) -> str:
    """生成"只采少数几帧"的配置，把单次采集压到 1 秒以内。"""
    with open(src, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg["test_frame"] = test_frame
    cfg["dump_one_frame"] = dump_frame
    cfg["use_v4l"] = 0
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=4)
    return dst


def nv12_to_bgr(buf: bytes, width: int, height: int,
                stride: int | None = None) -> np.ndarray:
    """NV12 -> BGR。stride 为 Y 平面每行字节数，默认等于 width。"""
    stride = stride or width
    y_size = stride * height
    uv_size = stride * (height // 2)

    y_plane = np.frombuffer(buf[:y_size], dtype=np.uint8).reshape(height, stride)
    uv_plane = np.frombuffer(buf[y_size:y_size + uv_size],
                             dtype=np.uint8).reshape(height // 2, stride // 2, 2)

    y = y_plane[:, :width]
    u = uv_plane[:, :width // 2, 0]
    v = uv_plane[:, :width // 2, 1]

    if _HAS_CV2:
        u2 = cv2.resize(u, (width, height), interpolation=cv2.INTER_LINEAR)
        v2 = cv2.resize(v, (width, height), interpolation=cv2.INTER_LINEAR)
        yuv = cv2.merge([y, u2, v2])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    # 没有 cv2 时的纯 numpy 实现（最近邻上采样）
    yf = y.astype(np.float32)
    uf = np.repeat(np.repeat(u.astype(np.float32), 2, axis=0), 2, axis=1)
    vf = np.repeat(np.repeat(v.astype(np.float32), 2, axis=0), 2, axis=1)
    c = yf - 16.0
    d = uf - 128.0
    e = vf - 128.0
    r = 1.164 * c + 1.596 * e
    g = 1.164 * c - 0.392 * d - 0.813 * e
    b = 1.164 * c + 2.017 * d
    bgr = np.stack([b, g, r], axis=-1)
    return np.clip(bgr, 0, 255).astype(np.uint8)


class K1Camera:
    """K1 MIPI 摄像头：每次 capture() 调一次 cam-test。"""

    def __init__(self,
                 json_path: str = DEFAULT_JSON,
                 out_path: str = DEFAULT_OUT,
                 width: int = 1920,
                 height: int = 1080,
                 timeout: float = 15.0,
                 use_sudo: bool = True,
                 verbose: bool = False):
        self.json_path = json_path
        self.out_path = out_path
        self.width = width
        self.height = height
        self.timeout = timeout
        self.use_sudo = use_sudo
        self.verbose = verbose
        self.last_duration = 0.0
        self.capture_count = 0

        if not os.path.isfile(json_path):
            make_quick_json(json_path)

    def _run_cam_test(self) -> None:
        cmd = []
        if self.use_sudo:
            cmd.append("sudo")
        cmd += ["cam-test", self.json_path]

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=self.timeout)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"cam-test 超时（{self.timeout}s）") from exc

        if self.verbose:
            tail = (proc.stdout or "")[-800:]
            print(tail)
        if proc.returncode != 0:
            raise RuntimeError(
                f"cam-test 失败（exit={proc.returncode}）：\n"
                f"{(proc.stderr or proc.stdout or '')[-500:]}"
            )

    def capture(self) -> np.ndarray:
        """采一帧，返回 BGR 的 numpy 数组。"""
        t0 = time.time()

        # 删掉旧文件，避免读到上次的缓存
        try:
            os.remove(self.out_path)
        except OSError:
            pass

        self._run_cam_test()

        if not os.path.exists(self.out_path):
            raise RuntimeError(
                f"没有生成图像文件 {self.out_path}\n"
                "  检查：cam-test 是否正常、JSON 里 dump_one_frame 是否小于 test_frame"
            )

        with open(self.out_path, "rb") as fh:
            buf = fh.read()

        self.last_duration = time.time() - t0
        self.capture_count += 1

        return nv12_to_bgr(buf, self.width, self.height, stride=self.width)

    def capture_and_save(self, path: str) -> np.ndarray:
        frame = self.capture()
        if _HAS_CV2:
            cv2.imwrite(path, frame)
        else:
            raise RuntimeError("保存图片需要 opencv：sudo apt install -y python3-opencv")
        return frame


def main() -> int:
    ap = argparse.ArgumentParser(description="K1 摄像头采集")
    ap.add_argument("--json", default=DEFAULT_JSON)
    ap.add_argument("--out", default="/tmp/shot.jpg")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    print("=" * 60)
    print("K1 MIPI 摄像头采集")
    print("=" * 60)

    cam = K1Camera(json_path=args.json, verbose=args.verbose)

    for i in range(args.repeat):
        try:
            frame = cam.capture()
        except Exception as exc:  # noqa: BLE001
            print(f"[{i+1}] ❌ {exc}")
            return 1

        h, w = frame.shape[:2]
        mean = frame.mean(axis=(0, 1))
        print(f"[{i+1}] ✅ {w}x{h}  耗时 {cam.last_duration:.2f}s  "
              f"平均BGR=({mean[0]:.0f},{mean[1]:.0f},{mean[2]:.0f})")

        if _HAS_CV2:
            out = args.out if args.repeat == 1 else args.out.replace(
                ".jpg", f"_{i+1}.jpg")
            cv2.imwrite(out, frame)
            print(f"     已保存 {out}")

    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())