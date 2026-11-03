#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把「分类数据集 + 自动标注」整理成 YOLO 训练目录。

为什么需要这一步
----------------
YOLO 要求固定结构：

    yolo_data/
      images/train/*.jpg
      images/val/*.jpg
      labels/train/*.txt      ← 与图片同名
      labels/val/*.txt
      data.yaml

而我们的数据是：

    yolo_data/datasets/grape/*.jpg    280 张
    yolo_data/datasets/other/*.jpg     10 张
    yolo_data/labels/grape/*.txt      256 个（另有 24 张没标出来）

**关键：没有标注的 24 张必须排除。**
YOLO 的约定是"图片存在但对应 txt 不存在/为空 = 该图没有目标"，
如果把这 24 张放进去，等于告诉模型"这些图里没有葡萄" ——
但它们其实有葡萄（只是 HSV 没检出），会严重污染训练。

用法::

    python tools/prep_yolo_dataset.py --root yolo_data --val-ratio 0.15
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import shutil
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

IMG_EXT = (".jpg", ".jpeg", ".png")


def main() -> int:
    ap = argparse.ArgumentParser(description="整理 YOLO 训练目录")
    ap.add_argument("--root", default="yolo_data", help="数据根目录")
    ap.add_argument("--val-ratio", type=float, default=0.15,
                    help="验证集比例")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None,
                    help="输出目录（默认 <root>/yolo）")
    args = ap.parse_args()

    root = args.root
    out = args.out or os.path.join(root, "yolo")
    ds = os.path.join(root, "datasets")
    lb = os.path.join(root, "labels")

    if not os.path.isdir(ds):
        print(f"❌ 找不到 {ds}")
        return 1

    # ---- 收集「有标注」的图片 ----
    pairs = []          # (img_path, label_path, is_negative)
    excluded = []
    for cls in sorted(os.listdir(ds)):
        cdir = os.path.join(ds, cls)
        if not os.path.isdir(cdir):
            continue
        for img in sorted(glob.glob(os.path.join(cdir, "*"))):
            if os.path.splitext(img)[1].lower() not in IMG_EXT:
                continue
            stem = os.path.splitext(os.path.basename(img))[0]
            lab = os.path.join(lb, cls, stem + ".txt")
            if not os.path.exists(lab):
                excluded.append(os.path.basename(img))
                continue
            txt = open(lab, encoding="utf-8").read().strip()
            pairs.append((img, lab, len(txt) == 0))

    n_neg = sum(1 for p in pairs if p[2])
    print("=" * 66)
    print("整理 YOLO 训练目录")
    print("=" * 66)
    print(f"  数据集　：{ds}")
    print(f"  标注　　：{lb}")
    print(f"  可用图片：{len(pairs)}（其中负样本 {n_neg} 张无目标）")
    print(f"  ⚠️ 排除　 ：{len(excluded)} 张（没有标注 → 不能当负样本）")
    if excluded:
        for e in excluded[:6]:
            print(f"       {e}")
        if len(excluded) > 6:
            print(f"       ... 还有 {len(excluded)-6} 张")
    print()

    if not pairs:
        print("❌ 没有可用的图片+标注对")
        return 1

    # ---- 划分 ----
    random.seed(args.seed)
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * args.val_ratio))
    val, train = pairs[:n_val], pairs[n_val:]

    # ---- 写文件 ----
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            d = os.path.join(out, sub, split)
            os.makedirs(d, exist_ok=True)

    for split, items in (("train", train), ("val", val)):
        for img, lab, _neg in items:
            dst_img = os.path.join(out, "images", split, os.path.basename(img))
            dst_lab = os.path.join(out, "labels", split,
                                   os.path.basename(lab))
            shutil.copy2(img, dst_img)
            shutil.copy2(lab, dst_lab)

    # ---- data.yaml ----
    yaml_path = os.path.join(out, "data.yaml")
    abs_out = os.path.abspath(out).replace("\\", "/")
    with open(yaml_path, "w", encoding="utf-8") as fh:
        fh.write(f"path: {abs_out}\n")
        fh.write("train: images/train\n")
        fh.write("val: images/val\n")
        fh.write("nc: 1\n")
        fh.write("names:\n")
        fh.write("  0: grape\n")

    print(f"  训练集：{len(train)} 张")
    print(f"  验证集：{len(val)} 张")
    print(f"  配置　：{yaml_path}")
    print()
    print("  下一步训练：")
    print(f"    D:\\yolo-env\\Scripts\\yolo.exe detect train "
          f"model=yolov8n.pt data={yaml_path} epochs=60 imgsz=640 batch=8")
    return 0


if __name__ == "__main__":
    sys.exit(main())