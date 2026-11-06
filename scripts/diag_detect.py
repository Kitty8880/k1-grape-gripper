#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断：绿葡萄为什么检不出 + 模板匹配可行性验证。

背景
----
2026-11-06 实测：绿葡萄在"夹爪远离"时能检出（占比 2.13%），
但夹爪一靠近就变成 0.00%（"没检出"）。

画面里葡萄清清楚楚（人眼一看就在中间），所以问题一定在**某一条筛选条件**
或**色域边界**上。这个脚本把每一层的通过/淘汰情况打印出来，
指出到底是哪一步排掉的。

同时验证用户提出的思路
----------------------
「用自由状态拍到的葡萄当模板，去匹配后续的葡萄」
—— 这就是 cv2.matchTemplate。它**不依赖颜色**，
所以同色物体（黑葡萄+黑夹爪）也能工作。

做法：
    ① 从 set_a（自由状态）取一张，用宽松掩码找到葡萄 → 裁出模板
    ② 在 set_b / set_c 的图上做模板匹配，报匹配分数和位置
    ③ 如果分数高（>0.6），说明模板匹配可用

用法::

    sudo python3 scripts/diag_detect.py --free debug/set_a --grip debug/set_b
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "apps"))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

import cv2
import numpy as np


def imread_u(path):
    buf = np.fromfile(path, dtype=np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def imwrite_u(path, img):
    ok, b = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if ok:
        b.tofile(path)


def stat(p, data):
    """打印一条掩码统计。"""
    n = int((data > 0).sum())
    pct = 100.0 * n / data.size
    print("    %-38s %8d px  (%5.2f%%)" % (p, n, pct))
    return n


def diag_mask(bgr):
    """
    逐层打印掩码，找出绿葡萄落在哪。
    """
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0].astype(np.int16)
    S = hsv[:, :, 1].astype(np.int16)
    V = hsv[:, :, 2].astype(np.int16)

    print("  ── 全图 HSV 分布（高饱和像素的色相直方图）──")
    hi_s = S > 60
    if hi_s.any():
        hh = H[hi_s]
        hist, _ = np.histogram(hh, bins=18, range=(0, 180))
        for i, c in enumerate(hist):
            if c > hh.size * 0.03:            # 只打印占比 >3% 的桶
                print("    H %3d~%3d : %6d px (%5.1f%%) %s"
                      % (i * 10, (i + 1) * 10 - 1, c, 100.0 * c / hh.size,
                         "#" * int(min(40, c / (hh.size / 1200)))))
    else:
        print("    （没有高饱和像素，画面很灰）")

    print("  ── 逐层掩码 ──")
    m_black = ((V < 110) & (S < 110)).astype(np.uint8) * 255
    m_purple = ((H >= 110) & (H <= 180) & (S >= 60) & (V >= 20)
                & (V <= 170)).astype(np.uint8) * 255
    m_green = ((H >= 20) & (H <= 55) & (S >= 55) & (V >= 25)
               & (V <= 250)).astype(np.uint8) * 255
    stat("黑/深灰 (V<110 & S<110)", m_black)
    stat("紫 (H110-180 & S>=60)", m_purple)
    stat("绿 (H20-55 & S>=55 & V25-250)", m_green)

    m = cv2.bitwise_or(cv2.bitwise_or(m_black, m_purple), m_green)
    stat("三者合并", m)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m2 = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    m2 = cv2.morphologyEx(m2, cv2.MORPH_OPEN, k, iterations=1)
    stat("形态学处理后", m2)

    print("  ── 连通域与几何筛选（这一步决定「被谁排除」）──")
    n, labels, stats, cents = cv2.connectedComponentsWithStats(m2, 8)
    cands = []
    for idx in range(1, n):
        a = int(stats[idx, cv2.CC_STAT_AREA])
        if a < w * h * 0.002:
            continue
        bw = int(stats[idx, cv2.CC_STAT_WIDTH])
        bh = int(stats[idx, cv2.CC_STAT_HEIGHT])
        if bw <= 0 or bh <= 0:
            continue
        aspect = min(bw, bh) / max(bw, bh)
        blob = (labels == idx).astype(np.uint8)
        cnts, _ = cv2.findContours(blob * 255, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        perim = cv2.arcLength(c, True) or 1.0
        circ = 4.0 * np.pi * a / (perim * perim)
        why = []
        if a > w * h * 0.35:
            why.append("面积过大(%.1f%%)" % (100.0 * a / (w * h)))
        if aspect < 0.62:
            why.append("长宽比%.2f<0.62" % aspect)
        if circ < 0.60:
            why.append("圆度%.2f<0.60" % circ)
        cands.append((a, aspect, circ, why, cents[idx],
                      stats[idx, cv2.CC_STAT_WIDTH],
                      stats[idx, cv2.CC_STAT_HEIGHT]))

    cands.sort(key=lambda t: -t[0])
    if not cands:
        print("    ❌ 没有面积 >0.2% 的连通域 —— 说明颜色层面就没抓到葡萄")
        return None

    for i, (a, aspect, circ, why, cen, bw, bh) in enumerate(cands[:6], 1):
        tag = "✅ 通过" if not why else "❌ " + "、".join(why)
        print("    #%d 面积%7d(%.2f%%) 尺寸%dx%d 长宽比%.2f 圆度%.2f  质心(%.0f,%.0f)  %s"
              % (i, a, 100.0 * a / (w * h), bw, bh, aspect, circ,
                 cen[0], cen[1], tag))
    return cands


def make_template(bgr, cen, radius, pad=1.15):
    """按检测到的中心和半径裁出模板。"""
    h, w = bgr.shape[:2]
    r = int(radius * pad)
    x0 = max(0, int(cen[0]) - r)
    x1 = min(w, int(cen[0]) + r)
    y0 = max(0, int(cen[1]) - r)
    y1 = min(h, int(cen[1]) + r)
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None, None
    return bgr[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)


def match(bgr, tmpl, scales=(0.8, 0.9, 1.0, 1.1, 1.25)):
    """多尺度模板匹配，返回 (最高分, 位置, 尺度)。"""
    best = (0.0, None, None)
    for s in scales:
        t = cv2.resize(tmpl, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        th, tw = t.shape[:2]
        if th >= bgr.shape[0] or tw >= bgr.shape[1]:
            continue
        res = cv2.matchTemplate(bgr, t, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(res)
        if mx > best[0]:
            best = (mx, loc, (s, tw, th))
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description="绿葡萄检测诊断 + 模板匹配验证")
    ap.add_argument("--free", default="debug/set_a", help="自由状态目录")
    ap.add_argument("--grip", action="append", default=None,
                    help="要测试的目录，可多次传")
    ap.add_argument("--n", type=int, default=3, help="每组看几张")
    args = ap.parse_args()

    def _abs(d):
        return d if os.path.isabs(d) else os.path.join(_ROOT, d)

    print("=" * 78)
    print("绿葡萄检测诊断 + 模板匹配验证")
    print("=" * 78)

    free_files = sorted(glob.glob(os.path.join(_abs(args.free), "*.jpg")),
                        key=os.path.getmtime)
    if not free_files:
        print("❌ 自由状态目录没有图片")
        return 1
    # 取最新的一张（用户刚拍的绿葡萄）
    ref_path = free_files[-1]
    print(f"\n参考图（最新一张自由状态）：{os.path.basename(ref_path)}")
    ref = imread_u(ref_path)
    if ref is None:
        print("❌ 读不出来")
        return 1

    cands = diag_mask(ref)

    tmpl = None
    if cands:
        # 选通过筛选的、面积最大的那个
        ok = [c for c in cands if not c[3]]
        pick = ok[0] if ok else cands[0]
        a, aspect, circ, why, cen, bw, bh = pick
        r = max(bw, bh) / 2.0
        tmpl, box = make_template(ref, cen, r)
        if tmpl is not None:
            p = os.path.join(_ROOT, "debug", "tmpl.jpg")
            os.makedirs(os.path.dirname(p), exist_ok=True)
            imwrite_u(p, tmpl)
            print(f"\n  ✅ 模板已裁出：{tmpl.shape[1]}x{tmpl.shape[0]}  → debug/tmpl.jpg")
            print(f"     来源：面积{a} 长宽比{aspect:.2f} 圆度{circ:.2f} 质心({cen[0]:.0f},{cen[1]:.0f})")
        else:
            print("\n  ⚠️ 模板裁剪失败（边界太小）")

    if tmpl is None:
        print("\n❌ 自由状态也没检测到葡萄，无法建模板")
        return 1

    # ---- 在各组上做模板匹配 ----
    dirs = [args.free] + (args.grip or ["debug/set_b", "debug/set_c"])
    print()
    print("=" * 78)
    print("模板匹配结果（分数 >0.6 算匹配成功）")
    print("=" * 78)
    for d in dirs:
        files = sorted(glob.glob(os.path.join(_abs(d), "*.jpg")),
                       key=os.path.getmtime)[-args.n:]
        if not files:
            print(f"\n{d}: 没有图片")
            continue
        print(f"\n{d}（最新 {len(files)} 张）：")
        for f in files:
            bgr = imread_u(f)
            if bgr is None:
                continue
            score, loc, info = match(bgr, tmpl)
            s = "%.3f" % score
            flag = "✅" if score > 0.6 else ("🔶" if score > 0.45 else "❌")
            pos = f"({loc[0]},{loc[1]})" if loc else "--"
            sc = f"x{info[0]}" if info else "--"
            print("    %-42s 分数 %s  %s  位置 %-12s 尺度 %s"
                  % (os.path.basename(f)[:42], s, flag, pos, sc))
    return 0


if __name__ == "__main__":
    sys.exit(main())