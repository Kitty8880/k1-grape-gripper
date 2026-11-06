#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分辨率探针：找出"葡萄边缘最锐利"的采集分辨率。

为什么需要这个
--------------
2026-11-05 决定走"边缘判断葡萄状态"的新方案，但实测发现图像太糊：

    历史 100% 识别率的那张图    Laplacian 26.9
    今天拍的 31 张里最好的      Laplacian 38.7
    边缘分析需要的（我猜的）    ≥ 80

**那个 80 也是我凭经验拍的，和之前的 100 一样不可信。**
所以在改算法之前，先用实测把两件事定下来：

    ① 哪个采集分辨率能给出最锐利的边缘
    ② 到底需要多锐利，flatness/aspect 才稳定

摄像头配置（/usr/share/camera_json/csi1_camera_auto.json）里
分辨率和 fps 都是可配的：

    isp0.out_width/out_height    1920 x 1080
    cpp0.size_width/size_height  1920 x 1080
    isp0.fps                     30

OV5647 常见模式有 640x480 / 1280x720 / 1920x1080。
**降低分辨率不一定变糊** —— 1080p 若经过 ISP 下采样，
反而可能比原生的 720p/480p 更软。

用法::

    # 逐个分辨率测，每个拍 3 帧就够
    sudo python3 scripts/capture_probe.py -w 640  -h 480  -n 3
    sudo python3 scripts/capture_probe.py -w 1280 -h 720  -n 3
    sudo python3 scripts/capture_probe.py -w 1920 -h 1080 -n 3

    # 三种一次跑完（推荐）
    sudo python3 scripts/capture_probe.py --sweep

输出::

    debug/probe/ 下的图 + 控制台统计表

⚠️ 这是**短时任务**（每档 3 帧，几秒就结束），不会像预览那样长时间占 CPU。
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _HERE)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

import cv2
import numpy as np

SRC_JSON = "/usr/share/camera_json/csi1_camera_auto.json"
GRAPE_HSV = ((120, 15, 15), (180, 255, 130))


def make_json(w: int, h: int, fps: int = 30, dst: str | None = None) -> str:
    """
    按指定分辨率生成一份 cam-test 配置。

    同时改三处，缺一处就会不一致：
        cpp0.size_width/height    → 输出的帧尺寸
        isp0.out_width/out_height → ISP 输出尺寸
        isp0.in_width/in_height   → ISP 输入尺寸（跟随传感器模式）
    """
    dst = dst or f"/tmp/probe_{w}x{h}.json"
    with open(SRC_JSON, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)

    for node in cfg.get("cpp_node", []):
        if node.get("name") == "cpp0":
            node["size_width"] = w
            node["size_height"] = h

    for node in cfg.get("isp_node", []):
        if node.get("name") == "isp0":
            node["out_width"] = w
            node["out_height"] = h
            node["in_width"] = w
            node["in_height"] = h
            node["fps"] = fps

    cfg["auto_run"] = 0
    cfg["test_frame"] = 2000000
    cfg["dump_one_frame"] = 1999999
    cfg["use_v4l"] = 0

    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=4)
    return dst


def analyze(bgr):
    """返回 (全图清晰度, 葡萄区域清晰度, 葡萄面积占比, 葡萄中心)。"""
    h, w = bgr.shape[:2]
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    focus = float(cv2.Laplacian(g, cv2.CV_64F).var())

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array(GRAPE_HSV[0], np.uint8),
                    np.array(GRAPE_HSV[1], np.uint8))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)

    n, labels, stats, cents = cv2.connectedComponentsWithStats(m, 8)
    best = None
    for idx in range(1, n):
        a = int(stats[idx, cv2.CC_STAT_AREA])
        if a < (w * h * 0.001):          # 太小忽略
            continue
        bw = int(stats[idx, cv2.CC_STAT_WIDTH])
        bh = int(stats[idx, cv2.CC_STAT_HEIGHT])
        if bw <= 0 or bh <= 0 or min(bw, bh) / max(bw, bh) < 0.55:
            continue
        if best is None or a > best[1]:
            best = (idx, a)

    if best is None:
        return focus, None, None, None

    idx, area = best
    cx, cy = cents[idx]
    frac = area / (w * h)

    # 葡萄区域内的清晰度（这才是"葡萄拍清楚了吗"）
    blob = (labels == idx).astype(np.uint8)
    ys, xs = np.nonzero(blob)
    if xs.size < 50:
        return focus, None, frac, (cx, cy)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop = g[max(0, y0):y1, max(0, x0):x1]
    fg = float(cv2.Laplacian(crop, cv2.CV_64F).var()) if crop.size else None
    return focus, fg, frac, (cx, cy)


def probe(w: int, h: int, frames: int, save_dir: str, outdir: str) -> dict:
    """测一个分辨率，返回统计。"""
    from k1_camera_stream import K1CameraStream

    print(f"\n── {w}x{h} " + "─" * 46)
    try:
        js = make_json(w, h)
        print(f"   配置：{js}")
    except Exception as exc:  # noqa: BLE001
        print(f"   ❌ 生成配置失败：{exc}")
        return {"res": f"{w}x{h}", "ok": False}

    out_path = f"/tmp/cpp0_output_{w}x{h}_s{w}.nv12"
    cam = None
    try:
        t0 = time.time()
        cam = K1CameraStream(json_path=js, out_path=out_path,
                             width=w, height=h, use_sudo=False,
                             verbose=False)
        cam.open()
        print(f"   打开耗时 {time.time()-t0:.1f}s")
    except Exception as exc:  # noqa: BLE001
        print(f"   ❌ 打开失败：{exc}")
        return {"res": f"{w}x{h}", "ok": False}

    foc, fgs, fracs, times = [], [], [], []
    try:
        for i in range(frames):
            t = time.time()
            bgr = cam.read_frame(timeout=6.0)
            dt = (time.time() - t) * 1000
            if bgr is None:
                print(f"   帧{i+1}: 超时")
                continue
            times.append(dt)
            focus, fg, frac, cen = analyze(bgr)
            foc.append(focus)
            if fg is not None:
                fgs.append(fg)
            if frac is not None:
                fracs.append(frac)

            os.makedirs(save_dir, exist_ok=True)
            name = f"{w}x{h}_{i+1}_f{focus:.1f}.jpg"
            p = os.path.join(save_dir, name)
            okc, buf = cv2.imencode(".jpg", bgr,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            if okc:
                buf.tofile(p)

            print(f"   帧{i+1}: {bgr.shape[1]}x{bgr.shape[0]}  "
                  f"全图清晰 {focus:6.1f}  "
                  f"葡萄 {f'{fg:6.1f}' if fg else '  --  '}  "
                  f"占比 "
                  f"{f'{frac*100:5.2f}%' if frac else '  --  '}  "
                  f"{dt:.0f}ms")
    finally:
        try:
            cam.close()
        except Exception:  # noqa: BLE001
            pass

    if not foc:
        return {"res": f"{w}x{h}", "ok": False}

    return {
        "res": f"{w}x{h}",
        "ok": True,
        "focus_med": statistics.median(foc),
        "focus_max": max(foc),
        "grape_med": statistics.median(fgs) if fgs else None,
        "frac_med": statistics.median(fracs) if fracs else None,
        "ms_med": statistics.median(times) if times else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="采集分辨率探针")
    ap.add_argument("-w", "--width", type=int, default=1280)
    ap.add_argument("-H", "--height", type=int, default=720)
    ap.add_argument("-n", "--frames", type=int, default=3)
    ap.add_argument("--sweep", action="store_true",
                    help="依次试 640x480 / 1280x720 / 1920x1080")
    ap.add_argument("--outdir", default="debug/probe")
    args = ap.parse_args()

    save_dir = os.path.join(_ROOT, args.outdir)
    print("=" * 72)
    print("采集分辨率探针")
    print("=" * 72)
    print(f"  输出目录：{save_dir}")
    print(f"  这是短时任务，每档 {args.frames} 帧")

    combos = ([(640, 480), (1280, 720), (1920, 1080)] if args.sweep
              else [(args.width, args.height)])

    results = []
    for w, h in combos:
        results.append(probe(w, h, args.frames, save_dir, args.outdir))

    print()
    print("=" * 72)
    print("汇总")
    print("=" * 72)
    print(f"  {'分辨率':<12}{'全图清晰':>10}{'葡萄清晰':>10}"
          f"{'占比%':>8}{'耗时ms':>8}")
    best = None
    for r in results:
        if not r["ok"]:
            print("  %-12s %10s" % (r["res"], "失败"))
            continue
        # ⚠️ 这里不用嵌套 f-string：Python 3.9 不允许 f-string 里
        #    再用同一种引号，会报 "f-string: unmatched '['"。
        #    先算好字符串再格式化。
        fg = r["grape_med"]
        fr = r["frac_med"]
        ms = r["ms_med"]
        s_fg = "%.1f" % fg if fg else "--"
        s_fr = "%.2f" % (fr * 100) if fr else "--"
        s_ms = "%.0f" % ms if ms else "--"
        print("  %-12s %10.1f %10s %8s %8s"
              % (r["res"], r["focus_med"], s_fg, s_fr, s_ms))
        score = fg if fg else r["focus_med"]
        if best is None or score > best[1]:
            best = (r["res"], score)

    print()
    if best:
        print(f"  ★ 最清晰的分辨率：**{best[0]}**（清晰度 {best[1]:.1f}）")
        print()
        print("  下一步：")
        print(f"    1. 用 {best[0]} 拍一张，看 debug/probe/ 里的图是否清楚")
        print("    2. 如果还不够清楚，说明是**对焦距离**问题，不是分辨率")
        print("       做法：固定用这个分辨率，把葡萄放在 20/25/30/35cm 各拍一张")
    return 0


if __name__ == "__main__":
    sys.exit(main())