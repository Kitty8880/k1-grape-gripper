#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K1 MIPI 摄像头 —— 交互模式连续采集（类视频流）。

原理
----
官方 ``cam-test`` 在 ``auto_run: 0`` 时进入交互模式，pipeline 常开：

    s  开始推流（sensor -> CSI -> ISP -> CPP，内部 15.6 fps）
    d  保存当前帧到 /tmp/cpp0_output_<w>x<h>_s<stride>.nv12
    c  停止推流
    q  退出

本模块把 cam-test 作为子进程启动，保持 pipeline 常开，
需要取帧时发送 ``d``，等文件更新后读入并转成 BGR —— 这样就不用每次
重建 pipeline（省掉约 0.85 秒的初始化开销）。

用法
----
命令行（采 30 帧、报告帧率）::

    sudo python3 k1_camera_stream.py --frames 30 --save-every 10

作为模块::

    from k1_camera_stream import K1CameraStream
    with K1CameraStream() as cam:
        while True:
            frame = cam.read_frame()   # numpy BGR
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pty
import select
import subprocess
import sys
import threading
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


DEFAULT_JSON = "/tmp/interactive.json"
DEFAULT_OUT = "/tmp/cpp0_output_1920x1080_s1920.nv12"
DEFAULT_SRC_JSON = "/usr/share/camera_json/csi1_camera_auto.json"


def make_interactive_json(dst: str = DEFAULT_JSON,
                          src: str = DEFAULT_SRC_JSON,
                          test_frame: int = 2000000) -> str:
    """生成交互模式配置：pipeline 常开，帧由按键触发保存。"""
    with open(src, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg["auto_run"] = 0
    cfg["test_frame"] = test_frame
    cfg["dump_one_frame"] = test_frame - 1   # 不自动保存，避免干扰
    cfg["use_v4l"] = 0
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=4)
    return dst


def nv12_to_bgr(buf: bytes, width: int, height: int,
                stride: int | None = None) -> np.ndarray:
    """NV12 -> BGR。"""
    stride = stride or width
    y_size = stride * height
    uv_size = stride * (height // 2)

    y = np.frombuffer(buf[:y_size], dtype=np.uint8).reshape(height, stride)[:, :width]
    uv = np.frombuffer(buf[y_size:y_size + uv_size],
                       dtype=np.uint8).reshape(height // 2, stride // 2, 2)
    u = uv[:, :width // 2, 0]
    v = uv[:, :width // 2, 1]

    if _HAS_CV2:
        u2 = cv2.resize(u, (width, height), interpolation=cv2.INTER_LINEAR)
        v2 = cv2.resize(v, (width, height), interpolation=cv2.INTER_LINEAR)
        return cv2.cvtColor(cv2.merge([y, u2, v2]), cv2.COLOR_YUV2BGR)

    yf = y.astype(np.float32)
    uf = np.repeat(np.repeat(u.astype(np.float32), 2, axis=0), 2, axis=1)
    vf = np.repeat(np.repeat(v.astype(np.float32), 2, axis=0), 2, axis=1)
    c, d, e = yf - 16.0, uf - 128.0, vf - 128.0
    bgr = np.stack([1.164 * c + 2.017 * d,
                    1.164 * c - 0.392 * d - 0.813 * e,
                    1.164 * c + 1.596 * e], axis=-1)
    return np.clip(bgr, 0, 255).astype(np.uint8)


class K1CameraStream:
    """把 cam-test 交互模式包装成可按需取帧的"摄像头"。"""

    def __init__(self,
                 json_path: str = DEFAULT_JSON,
                 out_path: str = DEFAULT_OUT,
                 width: int = 1920,
                 height: int = 1080,
                 use_sudo: bool = True,
                 verbose: bool = False,
                 save_log: str | None = None):
        self.json_path = json_path
        self.out_path = out_path
        self.width = width
        self.height = height
        self.use_sudo = use_sudo
        self.verbose = verbose
        self.save_log = save_log

        self.expected_size = int(width * height * 1.5)
        self.frame_count = 0
        self.last_dump_ms = 0.0

        self._proc: subprocess.Popen | None = None
        self._master: int | None = None
        self._log_lines: collections.deque[str] = collections.deque(maxlen=800)
        self._reader: threading.Thread | None = None
        self._lock = threading.Lock()

        if not os.path.isfile(json_path):
            make_interactive_json(json_path)

    # ---------------- 子进程管理 ----------------

    def _drain_stdout(self) -> None:
        """
        从 pty 主设备读输出。

        为什么用 pty：cam-test 是 C 程序，printf 输出到**管道**时是全缓冲
        （4KB 一块），日志会卡在缓冲区里读不到；输出到**终端**时才是行缓冲。
        给一个伪终端，它就会按行输出。
        """
        while True:
            if self._master is None:
                break
            with self._lock:
                alive = self._proc is not None and self._proc.poll() is None
            try:
                ready, _, _ = select.select([self._master], [], [], 0.2)
            except (OSError, ValueError):
                break
            if not ready:
                if not alive:
                    break
                continue
            try:
                data = os.read(self._master, 4096)
            except OSError:
                break
            if not data:
                break
            text = data.decode("utf-8", "replace")
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            for line in text.split("\n"):
                s = line.strip()
                if not s:
                    continue
                with self._lock:
                    self._log_lines.append(s)
                if self.verbose:
                    print(f"    [cam-test] {s}")

    def _send(self, ch: str) -> None:
        if self._master is None:
            raise RuntimeError("cam-test 未启动")
        os.write(self._master, (ch + "\n").encode())

    def _wait_for(self, pattern: str, timeout: float = 15.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._lock:
                if any(pattern in ln for ln in self._log_lines):
                    return True
            time.sleep(0.02)
        return False

    def tail_log(self, n: int = 15) -> str:
        with self._lock:
            return "\n".join(list(self._log_lines)[-n:])

    def open(self) -> "K1CameraStream":
        # 如果脚本本身已经是 root，就不需要再套一层 sudo
        need_sudo = self.use_sudo and os.geteuid() != 0
        cmd = (["sudo"] if need_sudo else []) + ["cam-test", self.json_path]

        self._master, slave = pty.openpty()
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=slave,
                stdout=slave,
                stderr=slave,
                close_fds=True,
            )
        finally:
            os.close(slave)

        self._reader = threading.Thread(target=self._drain_stdout, daemon=True)
        self._reader.start()

        if not self._wait_for("Input a character", timeout=20):
            raise RuntimeError(
                "cam-test 没有进入交互模式。\n"
                f"  最近输出：\n{self.tail_log(20)}"
            )

        self._send("s")
        if not self._wait_for("sensor stream on", timeout=25):
            raise RuntimeError(
                "推流失败（等不到 'sensor stream on'）。\n"
                f"  最近输出：\n{self.tail_log(20)}"
            )

        time.sleep(0.4)
        return self

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            self._send("c")
            time.sleep(0.3)
            self._send("q")
        except (OSError, ValueError):
            pass
        try:
            self._proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        if self._master is not None:
            try:
                os.close(self._master)
            except OSError:
                pass
            self._master = None
        if self.save_log:
            with open(self.save_log, "w", encoding="utf-8") as fh:
                with self._lock:
                    fh.write("\n".join(self._log_lines))
        self._proc = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    # ---------------- 取帧 ----------------

    def _stat(self):
        try:
            st = os.stat(self.out_path)
            return (st.st_mtime_ns, st.st_size)
        except OSError:
            return None

    def read_frame(self, timeout: float = 3.0) -> np.ndarray:
        """发一次 ``d``，等新帧落盘，返回 BGR 数组。"""
        if self._proc is None:
            raise RuntimeError("请先 open()")

        before = self._stat()
        t_send = time.time()
        self._send("d")

        # 等文件更新 + 大小正确 + 连续两次相同（说明写完）
        last = None
        t0 = time.time()
        while time.time() - t0 < timeout:
            st = self._stat()
            if st is not None and st[1] == self.expected_size:
                if before is None or st != before:
                    if last == st:
                        break
                    last = st
            time.sleep(0.005)
        else:
            raise TimeoutError(
                f"等待新帧超时（{timeout}s）。"
                f"期望 {self.expected_size} 字节，实际 {self._stat()}"
            )

        self.last_dump_ms = (time.time() - t_send) * 1000.0
        with open(self.out_path, "rb") as fh:
            buf = fh.read()

        self.frame_count += 1
        return nv12_to_bgr(buf, self.width, self.height, stride=self.width)


def main() -> int:
    ap = argparse.ArgumentParser(description="K1 摄像头连续采集")
    ap.add_argument("--json", default=DEFAULT_JSON)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=0,
                    help="每 N 帧存一张 JPEG（0=不存）")
    ap.add_argument("--outdir", default="/tmp")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    print("=" * 64)
    print("K1 摄像头连续采集（交互模式）")
    print("=" * 64)

    if not _HAS_CV2:
        print("⚠️  没有 opencv，无法保存 JPEG")

    try:
        cam = K1CameraStream(json_path=args.json, verbose=args.verbose)
        cam.open()
    except Exception as exc:  # noqa: BLE001
        print(f"❌ 启动失败：{exc}")
        return 1

    print(f"✅ pipeline 已启动，开始采 {args.frames} 帧\n")

    durations = []
    try:
        for i in range(args.frames):
            t0 = time.time()
            frame = cam.read_frame()
            dt = (time.time() - t0) * 1000.0
            durations.append(dt)

            h, w = frame.shape[:2]
            mean = frame.mean(axis=(0, 1))
            print(f"[{i+1:>3}] {w}x{h}  {dt:6.1f} ms  "
                  f"平均BGR=({mean[0]:5.0f},{mean[1]:5.0f},{mean[2]:5.0f})")

            if args.save_every and (i + 1) % args.save_every == 0 and _HAS_CV2:
                p = os.path.join(args.outdir, f"frame_{i+1:04d}.jpg")
                cv2.imwrite(p, frame)
                print(f"       已保存 {p}")
    except KeyboardInterrupt:
        print("\n中断")
    finally:
        cam.close()

    if durations:
        avg = sum(durations) / len(durations)
        print()
        print("=" * 64)
        print(f"共 {len(durations)} 帧，平均 {avg:.1f} ms/帧 "
              f"→ 约 {1000.0/avg:.1f} fps")
        print("=" * 64)

    return 0


if __name__ == "__main__":
    sys.exit(main())