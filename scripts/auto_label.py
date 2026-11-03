#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用现有的 HSV 检测器**自动生成 YOLO 标注**。

为什么这样做
------------
采集的 280 张葡萄图原来是给**分类**模型用的（整张图就是一个类别），
没有位置标注。YOLO 是**检测**模型，需要每张图的边界框。

手工标 280 张图约 45 分钟，而且容易标错。

但现有的 HSV 检测器在这 280 张上已经能稳定找到葡萄（实测置信度
0.93~0.98、半径波动 ±4%），所以**让它当标注工**：

        原图 → HSV 检测 → 圆 (x, y, r) → 边界框 → YOLO 标签

这是工程上常用的"弱监督 / 伪标签"做法：
用已经调好的传统算法生成标注，再用这些标注训练神经网络，
最后让神经网络替代传统算法（速度、鲁棒性都更好）。

⚠️ 伪标签的局限
----------------
· HSV 漏检的图 → 没有标签 → 该图被当作"纯背景"训练，
  这会**教坏模型**（把有葡萄的图当负样本）。
  所以漏检的图必须**单独列出来并排除**，不能当负样本。
· HSV 框得不准的图 → 标签偏 → 影响精度。
  所以本脚本会按置信度过滤，低置信度的单独统计。

用法::

    # 生成标注（在板子上跑，因为有 cv2 和检测器）
    python3 scripts/auto_label.py

    # 指定目录和阈值
    python3 scripts/auto_label.py --src datasets --out labels --min-conf 0.5

    # 只看统计，不写文件
    python3 scripts/auto_label.py --dry-run

输出::

    labels/grape/grape_0000.txt     ← 每行 "0 cx cy w h"（归一化 0~1）
    labels/other/other_0000.txt     ← 空文件（负样本，无目标）

    并在控制台打印：检出率、置信度分布、被排除的文件清单
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

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description="用 HSV 检测器自动生成 YOLO 标注")
    ap.add_argument("--src", default=os.path.join(_ROOT, "datasets"),
                    help="数据集根目录（下面有 grape/ other/ 等子目录）")
    ap.add_argument("--out", default=os.path.join(_ROOT, "labels"),
                    help="标注输出目录")
    ap.add_argument("--color", default="grape", help="HSV 颜色档")
    ap.add_argument("--roi", default="0.0,0.0,1.0,1.0",
                    help="ROI 比例。自动标注时**不要用**默认的中心 ROI —— "
                         "采集时葡萄不一定在中央，用全图才标得全")
    ap.add_argument("--min-conf", type=float, default=0.45,
                    help="低于这个置信度的检测不生成标注，"
                         "该图列入「需人工检查」清单")
    ap.add_argument("--pad", type=float, default=0.06,
                    help="边界框外扩比例（默认 6%），"
                         "给检测留一点余量，避免框太紧")
    ap.add_argument("--dry-run", action="store_true", help="只统计，不写文件")
    args = ap.parse_args()

    try:
        parts = [float(v) for v in args.roi.split(",")]
        if len(parts) != 4:
            raise ValueError
        roi = tuple(parts)
    except ValueError:
        print(f"❌ --roi 格式错误：{args.roi}")
        return 1

    try:
        import cv2
        from grape_detect import GrapeDetector
    except ImportError as exc:
        print(f"❌ 导入失败：{exc}")
        print("   这个脚本需要在板子上跑（板子有 cv2 和 grape_detect）")
        return 1

    if not os.path.isdir(args.src):
        print(f"❌ 找不到数据集目录：{args.src}")
        return 1

    det = GrapeDetector(color=args.color)
    # 自动标注时阈值放宽：宁可多标一点，也不要漏标（漏标会变成假负样本）
    circ_th = min(det.min_circularity, 0.35)
    sol_th = min(det.min_solidity, 0.30)
    print("=" * 68)
    print("自动生成 YOLO 标注")
    print("=" * 68)
    print(f"  数据集　：{args.src}")
    print(f"  输出　　：{args.out}")
    print(f"  颜色档　：{args.color}（圆度 ≥{circ_th:.2f} 实心度 ≥{sol_th:.2f}）")
    print(f"  ROI　　：{'全图' if roi == (0.0, 0.0, 1.0, 1.0) else roi}")
    print(f"  置信度　：≥{args.min_conf} 才生成标注")
    print()

    classes = sorted(d for d in os.listdir(args.src)
                     if os.path.isdir(os.path.join(args.src, d)))
    total_ok = total_skip = 0
    summary = []

    for cls in classes:
        cdir = os.path.join(args.src, cls)
        files = sorted(glob.glob(os.path.join(cdir, "*.jpg"))
                       + glob.glob(os.path.join(cdir, "*.jpeg"))
                       + glob.glob(os.path.join(cdir, "*.png")))
        if not files:
            continue
        is_target = cls.lower() in ("grape", "葡萄", "target")

        outdir = os.path.join(args.out, cls)
        if not args.dry_run:
            os.makedirs(outdir, exist_ok=True)

        ok = skip = neg = 0
        skipped = []
        confs = []
        print(f"── {cls}/　共 {len(files)} 张"
              f"{'（目标类，需要标注）' if is_target else '（负样本）'}")

        for f in files:
            stem = os.path.splitext(os.path.basename(f))[0]
            lab = os.path.join(outdir, stem + ".txt")

            if not is_target:
                # 负样本：写空文件（YOLO 约定：无目标就是空 txt）
                if not args.dry_run:
                    with open(lab, "w", encoding="utf-8") as fh:
                        pass
                neg += 1
                continue

            bgr = cv2.imread(f)
            if bgr is None:
                skipped.append((stem, "读不出来"))
                skip += 1
                continue
            h, w = bgr.shape[:2]

            found = det.detect(bgr)
            # ROI 过滤（默认全图，等于不过滤）
            x0, y0, x1, y1 = roi
            inside = [d for d in found
                      if x0 * w <= d.x <= x1 * w and y0 * h <= d.y <= y1 * h]
            if not inside:
                skipped.append((stem, f"没检出（候选 {len(found)} 个）"))
                skip += 1
                continue

            best = max(inside, key=lambda d: d.area)
            conf = min(1.0, max(0.0, (best.circularity - 0.5) / 0.5))
            if conf < args.min_conf:
                skipped.append((stem, f"置信度低 {conf:.2f}"))
                skip += 1
                continue

            # 圆 → 边界框
            #
            # ⚠️ **不能用 best.radius 换算框**！
            #
            #   2026-11-03 实测：用 radius 生成的 256 个标注里，
            #   高度中位数 0.503、长宽比 0.56（葡萄是圆的，应该约 1.0），
            #   也就是**系统性偏大且偏高**。
            #
            #   原因：`radius` 是凸包等效半径（sqrt(hull_area/π)），
            #   它把葡萄旁边的阴影、反光、粘连的背景一起算进了凸包 ——
            #   实测某个检测 r=341px，而葡萄本身只有 r≈150px，差一倍多。
            #
            #   正确做法：取**凸包的外接矩形**。它贴合真实轮廓，
            #   形状接近正方形（与圆形的葡萄一致），也自带正确的长宽比。
            hull = cv2.convexHull(best.contour)
            bx, by, bw_px, bh_px = cv2.boundingRect(hull)
            # 外扩一点余量，避免框太紧导致训练时目标被切边
            pad_x = bw_px * args.pad
            pad_y = bh_px * args.pad
            bx0 = max(0.0, bx - pad_x)
            by0 = max(0.0, by - pad_y)
            bx1 = min(float(w), bx + bw_px + pad_x)
            by1 = min(float(h), by + bh_px + pad_y)
            cx = (bx0 + bx1) / 2 / w
            cy = (by0 + by1) / 2 / h
            bw = (bx1 - bx0) / w
            bh = (by1 - by0) / h
            # 太小的框没有训练价值
            if bw * w < 20 or bh * h < 20:
                skipped.append((stem, f"框太小 {bw*w:.0f}x{bh*h:.0f}px"))
                skip += 1
                continue

            if not args.dry_run:
                with open(lab, "w", encoding="utf-8") as fh:
                    fh.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            ok += 1
            confs.append(conf)

        total_ok += ok
        total_skip += skip
        avgc = sum(confs) / len(confs) if confs else 0.0
        print(f"     生成标注 {ok}　"
              f"{'负样本 ' + str(neg) if neg else ''}"
              f"{'　跳过 ' + str(skip) if skip else ''}"
              f"{f'　平均置信度 {avgc:.2f}' if confs else ''}")
        if skipped:
            print(f"     ⚠️ 跳过 {len(skipped)} 张（这些图**不能当负样本**，"
                  f"建议人工检查或重新采集）：")
            for s_, why in skipped[:8]:
                print(f"         {s_}  ← {why}")
            if len(skipped) > 8:
                print(f"         ... 还有 {len(skipped)-8} 张")
        summary.append((cls, len(files), ok, skip, neg))

    print()
    print("=" * 68)
    print("汇总")
    print("=" * 68)
    print(f"  {'类别':<12}{'总数':>6}{'标注':>8}{'跳过':>8}{'负样本':>8}")
    for cls, n, ok, sk, ng in summary:
        print(f"  {cls:<12}{n:>6}{ok:>8}{sk:>8}{ng:>8}")
    print()
    if args.dry_run:
        print("  （--dry-run：没有写文件）")
    else:
        print(f"  标注已写入：{args.out}/")
        print()
        print("  下一步：把数据集和标注一起拷到训练机：")
        print("    scp -r ~/grape-gripper/datasets ~/grape-gripper/labels <PC>:...")
    return 0


if __name__ == "__main__":
    sys.exit(main())