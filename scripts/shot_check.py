#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单张照片质量检查 —— 拍 31 张之前先拍 1 张验证。

为什么需要这个
--------------
2026-11-05 第一次拍摄的 31 张图**无法用于边缘分析**，原因全在成像：

    ① 严重失焦 —— 边缘糊成一片，flatness 全在 0.2 附近，无法区分状态
    ② 视角不对 —— 相机顺着夹爪方向看，夹爪/传感器把葡萄挡住，
       set_b 里葡萄只剩 3,698 px（另外两组是 39,000 px，差 10 倍）
    ③ 构图偏 —— 葡萄贴在画面最上沿，夹爪在左上角，2/3 画面是空桌面

这些都是**在按快门之前就能检查出来**的。所以做这个工具：
拍一张 → 跑一下 → 五项全过 → 再拍 30 张。

五项检查
--------
    focus     Laplacian 方差 ≥ 100        图像是否清晰
    size      葡萄面积占画面 ≥ 1.5%        是否拍得太远
    center    质心离画面中心 ≤ 35% 半对角   构图是否偏
    contrast  葡萄内外亮度差 ≥ 25          边缘是否够强
    whole     葡萄轮廓闭合度 ≥ 0.85        是否被遮挡/贴边

用法::

    python scripts/shot_check.py --img debug/set_a/xxx.jpg
    python scripts/shot_check.py --dir debug/set_b
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2
import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

GRAPE_HSV = ((120, 15, 15), (180, 255, 130))

#: 五项阈值（先按经验给，实测后再调）
#: 五项阈值。
#:
#: ⚠️ focus 的来历（2026-11-05 修正过）：
#:   第一版凭经验写了 100（一般相机锐利图通常 >500）。
#:   但实测本机 OV5647 的成像**最好也只有 ~50**：
#:       hist_ok.jpg   Laplacian 26.9   ← 这张让视觉识别率达到 100%
#:       hist_ok2.jpg  Laplacian 48.9
#:   所以 100 这个阈值让**所有图**都判"失焦"，是错的。
#:
#:   但要区分两件事：
#:     · HSV 颜色检测对模糊**不敏感** —— 所以旧的 100% 识别率成立
#:     · 边缘/轮廓分析对模糊**很敏感** —— 边缘被抹宽几像素，
#:       flatness / aspect 全部失真
#:   因此 focus 阈值按用途分两档：
#:       ≥ FOCUS_MIN_COLOR(30)  够做颜色检测
#:       ≥ FOCUS_MIN_EDGE(80)   够做边缘分析（本机目前达不到）
TH = {
    "focus_color": 30.0,   # 够做 HSV 颜色检测
    "focus_edge": 80.0,    # 够做边缘分析
    "size": 0.015,         # 葡萄面积 / 画面面积
    "center": 0.35,        # 质心偏移 / 半对角
    "contrast": 25.0,      # 葡萄内外平均灰度差
    "whole": 0.85,         # 实心度（轮廓面积 / 凸包面积）
}


def imread_unicode(path):
    """读图，兼容中文路径（cv2.imread 在 Windows 上不吃非 ASCII 路径）。"""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:  # noqa: BLE001
        return None


def check(path):
    bgr = imread_unicode(path)
    if bgr is None:
        return None
    H, W = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    res = {"file": os.path.basename(path), "W": W, "H": H,
           "pass": True, "msgs": []}

    # ---- ① 清晰度 ----
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    res["focus"] = float(lap)
    if lap < TH["focus_color"]:
        res["pass"] = False
        res["msgs"].append(
            f"太糊，连颜色检测都不可靠（Laplacian {lap:.0f} < "
            f"{TH['focus_color']:.0f}）→ 检查距离/对焦")
    elif lap < TH["focus_edge"]:
        res["msgs"].append(
            f"⚠️ 只够颜色检测，不够边缘分析"
            f"（Laplacian {lap:.0f} < {TH['focus_edge']:.0f}）"
            f"→ 边缘会被抹宽，flatness/aspect 不可靠")

    # ---- 找葡萄 ----
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array(GRAPE_HSV[0], np.uint8),
                    np.array(GRAPE_HSV[1], np.uint8))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)

    n, labels, stats, cents = cv2.connectedComponentsWithStats(m, 8)
    best = None
    for idx in range(1, n):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < 800:
            continue
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        if w <= 0 or h <= 0 or min(w, h) / max(w, h) < 0.55:
            continue
        if best is None or area > best[1]:
            best = (idx, area)

    if best is None:
        res["pass"] = False
        res["msgs"].append("没检出葡萄（可能太远、被遮挡，或不在 HSV 范围）")
        return res

    idx, area = best
    blob = (labels == idx).astype(np.uint8) * 255
    cx, cy = cents[idx]
    res["area"] = area
    res["frac"] = area / (W * H)
    res["cx"], res["cy"] = float(cx), float(cy)

    # ---- ② 大小 ----
    if res["frac"] < TH["size"]:
        res["pass"] = False
        res["msgs"].append(
            f"葡萄太小（占画面 {res['frac']*100:.2f}% < "
            f"{TH['size']*100:.1f}%）→ 把相机靠近些")

    # ---- ③ 构图 ----
    d = np.hypot(cx - W / 2, cy - H / 2)
    half_diag = np.hypot(W / 2, H / 2)
    res["off"] = float(d / half_diag)
    if res["off"] > TH["center"]:
        where = []
        if cy < H * 0.2:
            where.append("太靠上")
        if cy > H * 0.8:
            where.append("太靠下")
        if cx < W * 0.2:
            where.append("太靠左")
        if cx > W * 0.8:
            where.append("太靠右")
        res["pass"] = False
        res["msgs"].append(
            f"构图偏（偏离中心 {res['off']*100:.0f}% "
            f"{'、'.join(where)}）→ 把葡萄放到画面中央")

    # ---- ④ 对比度 ----
    inner = cv2.erode(blob, np.ones((15, 15), np.uint8))
    ring = cv2.dilate(blob, np.ones((41, 41), np.uint8))
    ring = cv2.bitwise_and(ring, cv2.bitwise_not(blob))
    gi = float(gray[inner > 0].mean()) if (inner > 0).any() else 0.0
    go = float(gray[ring > 0].mean()) if (ring > 0).any() else 0.0
    res["contrast"] = abs(go - gi)
    if res["contrast"] < TH["contrast"]:
        res["pass"] = False
        res["msgs"].append(
            f"背景对比不足（差 {res['contrast']:.0f} < "
            f"{TH['contrast']:.0f}）→ 铺白纸当背景")

    # ---- ⑤ 轮廓完整度（是否被遮挡/贴边）----
    cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    ca = cv2.contourArea(c)
    hull = cv2.convexHull(c)
    ha = cv2.contourArea(hull)
    solidity = ca / ha if ha > 0 else 0.0
    res["solidity"] = float(solidity)
    if solidity < TH["whole"]:
        res["pass"] = False
        res["msgs"].append(
            f"轮廓不完整（实心度 {solidity:.2f} < {TH['whole']:.2f}）"
            f"→ 葡萄被夹爪/传感器遮挡，或贴到画面边缘")

    return res


def report(r):
    if r is None:
        print("  ❌ 读不出来")
        return
    flag = "✅ 通过" if r["pass"] else "❌ 不合格"
    print(f"  {r['file']}   {flag}")
    print(f"    清晰度 {r['focus']:7.1f}   "
          f"占比 {r.get('frac', 0)*100:5.2f}%   "
          f"偏心 {r.get('off', 0)*100:4.0f}%   "
          f"对比 {r.get('contrast', 0):5.1f}   "
          f"实心度 {r.get('solidity', 0):.2f}"
          + (f"   位置({r['cx']:.0f},{r['cy']:.0f})" if "cx" in r else ""))
    for m in r["msgs"]:
        print(f"      ⚠️ {m}")


def main() -> int:
    ap = argparse.ArgumentParser(description="单张照片质量检查")
    ap.add_argument("--img", default=None)
    ap.add_argument("--dir", default=None)
    args = ap.parse_args()

    print("=" * 74)
    print("拍摄质量检查")
    print("=" * 74)
    print(f"  阈值：颜色用清晰度 ≥{TH['focus_color']:.0f} / "
          f"边缘用 ≥{TH['focus_edge']:.0f}  占比 ≥{TH['size']*100:.1f}%  "
          f"偏心 ≤{TH['center']*100:.0f}%  对比 ≥{TH['contrast']:.0f}  "
          f"实心度 ≥{TH['whole']:.2f}")
    print()

    files = []
    if args.img:
        files = [args.img]
    elif args.dir:
        files = sorted(glob.glob(os.path.join(args.dir, "*.jpg")))
        files = [f for f in files if "_gap" not in f]
    else:
        print("❌ 要给 --img 或 --dir")
        return 1

    npass = 0
    for f in files:
        r = check(f)
        report(r)
        if r and r["pass"]:
            npass += 1
        print()

    print("=" * 74)
    print(f"  合计 {len(files)} 张，通过 {npass} 张")
    if npass < len(files):
        print("  ★ 先把不合格的原因解决掉，再拍完整数据集。")
        print("    最常见的两条：")
        print("      1. 相机离太近/太远 → 失焦（OV5647 固定焦点约 30cm）")
        print("      2. 视角顺夹爪方向 → 葡萄被挡住（应从上方俯视）")
    return 0


if __name__ == "__main__":
    sys.exit(main())