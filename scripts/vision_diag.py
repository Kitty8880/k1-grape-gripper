#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视觉诊断 —— 拍一张照，直接告诉你画面里有哪些色相（H 值），
以及应该用什么样的 HSV 区间才能框住葡萄。

为什么需要它
------------
HSV 检测失败时，"紫色区间没框到"可能有无数种原因：
光照偏色、葡萄品种不是紫的、自动白平衡把色相拉偏了……
靠猜参数效率极低。

本脚本把画面的 **H 通道直方图** 打出来，并给出"能覆盖 X% 像素的
HSV 区间"，照着填进 grape_detect.py 的 COLOR_RANGES 就行。

用法::

    sudo python3 scripts/vision_diag.py
    sudo python3 scripts/vision_diag.py --out debug

输出
----
    · 控制台：亮度统计、H 直方图、推荐的 HSV 区间
    · 文件：hue_map.png（色相可视化，不同色相用不同颜色标出）
          raw.jpg（原图，用于人工确认）
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _HERE)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

# OpenCV 的 H 是 0~180（不是 0~360）
_HUE_NAMES = [
    (0, 8, "红"), (8, 20, "橙"), (20, 32, "黄"), (32, 50, "黄绿"),
    (50, 70, "绿"), (70, 85, "青"), (85, 100, "天蓝"), (100, 115, "蓝"),
    (115, 130, "蓝紫"), (130, 145, "紫"), (145, 160, "品红"),
    (160, 180, "深红/粉"),
]


def hue_name(h: float) -> str:
    for lo, hi, name in _HUE_NAMES:
        if lo <= h < hi:
            return f"{name}({lo}-{hi})"
    return f"未知({h:.0f})"


def main() -> int:
    ap = argparse.ArgumentParser(description="视觉诊断：色相分布")
    ap.add_argument("--out", default=None, help="保存图片的目录")
    ap.add_argument("--sat", type=int, default=50,
                    help="统计时要求的最低饱和度（默认 50，过滤掉灰白色）")
    ap.add_argument("--val", type=int, default=40,
                    help="统计时要求的最低亮度（默认 40，过滤掉黑色）")
    args = ap.parse_args()

    try:
        import cv2
        import numpy as np
    except ImportError:
        print("❌ 需要 opencv： sudo apt install -y python3-opencv")
        return 1

    from k1_camera_capture import K1Camera

    print("=" * 72)
    print("视觉诊断 —— 色相分布")
    print("=" * 72)
    print()
    print("拍照中 ...")
    try:
        cam = K1Camera(use_sudo=(hasattr(os, "geteuid")
                                 and os.geteuid() != 0))
        bgr = cam.capture()
    except Exception as exc:  # noqa: BLE001
        print(f"❌ 拍照失败：{exc}")
        return 1

    h, w = bgr.shape[:2]
    print(f"  ✅ {w}x{h}  亮度均值 {bgr.mean():.1f}  "
          f"最大值 {bgr.max()}")

    # ---- 色相统计 ----
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask = (S >= args.sat) & (V >= args.val)
    total = H.size
    sel = int(mask.sum())
    print()
    print(f"  有效像素（S≥{args.sat} 且 V≥{args.val}）："
          f"{sel:,} / {total:,} = {sel/total*100:.1f}%")
    if sel == 0:
        print("  ⚠️ 几乎没有彩色像素 —— 画面是灰白/过曝/全黑？")
        print("     检查：葡萄是否在画面内、有无光照、镜头有无遮住")
        return 1

    hist = np.bincount(H[mask].ravel(), minlength=180).astype(np.float64)
    hist_pct = hist / hist.sum() * 100

    print()
    print("  H 值分布（按颜色名归并，只列占比 > 1% 的）")
    print("  " + "-" * 62)
    for lo, hi, name in _HUE_NAMES:
        pct = float(hist_pct[lo:hi].sum())
        if pct < 1.0:
            continue
        bar = "█" * int(min(40, pct / 2))
        print(f"   {name:<10} H {lo:>3}-{hi:<3}  {pct:>5.1f}%  {bar}")

    # ---- 单峰占比最高的 H 值 ----
    print()
    print("  明细（占比 > 2% 的单个 H 值）")
    print("  " + "-" * 62)
    top = np.argsort(hist_pct)[::-1][:12]
    for hv in sorted(int(x) for x in top if hist_pct[x] > 2.0):
        bar = "█" * int(min(40, hist_pct[hv] * 1.5))
        print(f"   H={hv:>3}  {hist_pct[hv]:>5.1f}%  "
              f"{hue_name(hv):<12} {bar}")

    # ---- 推荐区间 ----
    print()
    print("=" * 72)
    print("推荐的 HSV 区间")
    print("=" * 72)
    # 找占主导的连续色相带（用平滑后的直方图）
    kern = np.ones(7) / 7.0
    smooth = np.convolve(np.concatenate([hist_pct, hist_pct[:10]]),
                         kern, mode="same")[:180]
    peak = int(np.argmax(smooth))
    lo = peak
    while lo > 0 and smooth[lo - 1] > smooth[peak] * 0.4:
        lo -= 1
    hi = peak
    while hi < 179 and smooth[hi + 1] > smooth[peak] * 0.4:
        hi += 1
    band_pct = float(hist_pct[lo:hi + 1].sum())
    print(f"  画面里最主要的颜色带：H = {lo} ~ {hi}"
          f"   （{hue_name((lo+hi)/2)}）")
    print(f"  它覆盖了有效像素的 {band_pct:.1f}%")
    print()
    print("  如果葡萄就落在这个带里，把它填进")
    print("  scripts/grape_detect.py 的 COLOR_RANGES：")
    print()
    print(f'      "grape": [(({lo}, {args.sat}, {args.val}), '
          f'({hi}, 255, 255))],')
    print()
    print("  ⚠️ 如果葡萄不在这里（比如画面里主要是背景），")
    print("     把葡萄举到镜头前 20cm 再跑一次，让葡萄占满画面中央。")

    # ---- 存图 ----
    if args.out:
        try:
            os.makedirs(args.out, exist_ok=True)
            cv2.imwrite(os.path.join(args.out, "raw.jpg"), bgr)
            # 色相可视化：只保留有效像素，按 H 上色
            vis = np.zeros_like(bgr)
            hue_u8 = (H * 255 // 180).astype(np.uint8)
            vis = cv2.cvtColor(
                np.stack([hue_u8, S, V], axis=-1), cv2.COLOR_HSV2BGR)
            vis[~mask] = (30, 30, 30)
            cv2.imwrite(os.path.join(args.out, "hue_map.png"), vis)
            # 把推荐区间单独画成掩码，直观确认
            m = cv2.inRange(hsv, (lo, args.sat, args.val),
                            (hi, 255, 255))
            cv2.imwrite(os.path.join(args.out, "suggested_mask.png"), m)
            print()
            print(f"  已保存到 {args.out}/：")
            print("    raw.jpg            原图")
            print("    hue_map.png        色相可视化（灰色=低饱和度）")
            print("    suggested_mask.png 推荐区间的掩码")
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠️ 保存图片失败：{exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())