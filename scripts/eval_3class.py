#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三档分类准确率评估：用几何特征区分 自由 / 贴住 / 夹紧。

背景
----
2026-11-06 受控实验（同一会话、只改脉宽）的 30 张图，逐张分析发现：

    占比%    2.09~2.34 / 2.18~2.29 / 2.16~2.20   几乎一样，无区分力
    圆度     0.786~0.852 / 0.671~0.804 / 0.748~0.782  重叠
    实心度   0.976~0.985 / 0.940~0.971 / 0.956~0.965  重叠
    扁平度   中位 0.132 / 0.156 / 0.200    ★ 单调，+52%

所以这个脚本做两件事：

    ① 把扁平度（以及几个备选特征）在三组上的分布画成表格
    ② 用一维阈值搜索，找出最优的两条分界，报出准确率和混淆矩阵

"一维阈值搜索"的意思是：三档只需要两个切点 t1 < t2，
在候选值里穷举 (t1, t2)，取准确率最高的组合。
这比凭感觉定阈值靠谱。

用法::

    sudo python3 scripts/eval_3class.py /tmp/ga2 /tmp/gb2 /tmp/gc2
"""
from __future__ import annotations

import glob
import os
import statistics
import sys

sys.path.insert(0, "/home/daiweikai/grape-gripper/src")
sys.path.insert(0, "/home/daiweikai/grape-gripper/scripts")
sys.path.insert(0, "/home/daiweikai/grape-gripper/apps")

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

import cv2
import numpy as np
from shot_web import analyze

LABELS = ["自由", "贴住", "夹紧"]


def features(bgr, blob):
    """算一组候选特征，看哪个最能区分状态。"""
    f = {}
    cnts, _ = cv2.findContours(blob * 255, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return f
    c = max(cnts, key=cv2.contourArea)
    perim = cv2.arcLength(c, True)
    area = cv2.contourArea(c)

    # ① 扁平度：最长直边 / 周长（多个 eps 试）
    for eps in (0.01, 0.015, 0.02, 0.03):
        ap = cv2.approxPolyDP(c, eps * perim, True).reshape(-1, 2)
        if len(ap) >= 3:
            segs = []
            n = len(ap)
            for i in range(n):
                p1 = ap[i].astype(np.float32)
                p2 = ap[(i + 1) % n].astype(np.float32)
                dd = p2 - p1
                segs.append(float(np.hypot(dd[0], dd[1])))
            f["flat_%.3f" % eps] = max(segs) / perim
            # 前两长边之和（被压时两侧都变直）
            s2 = sorted(segs, reverse=True)[:2]
            f["flat2_%.3f" % eps] = sum(s2) / perim
        else:
            f["flat_%.3f" % eps] = float("nan")
            f["flat2_%.3f" % eps] = float("nan")

    # ② 拟合椭圆短轴/长轴
    if len(c) >= 5:
        (_, _), (ma, mi), _ = cv2.fitEllipse(c)
        f["ellipse_ar"] = (min(ma, mi) / max(ma, mi)) if ma > 0 else float("nan")
    else:
        f["ellipse_ar"] = float("nan")

    # ③ 外接框宽高比
    _, _, bw, bh = cv2.boundingRect(c)
    f["bbox_ar"] = (min(bw, bh) / max(bw, bh)) if max(bw, bh) > 0 else float("nan")

    # ④ 圆度
    f["circ"] = 4.0 * np.pi * area / (perim * perim) if perim > 0 else float("nan")
    return f


def collect(dirs):
    data = []          # [(label, {feat: val}, filename)]
    for label, d in zip(LABELS, dirs):
        for p in sorted(glob.glob(os.path.join(d, "*.jpg"))):
            buf = np.fromfile(p, dtype=np.uint8)
            bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            info, blob = analyze(bgr)
            if blob is None:
                print("  ⚠️ %-30s 未检出，跳过" % os.path.basename(p)[:30])
                continue
            data.append((label, features(bgr, blob),
                         os.path.basename(p)))
    return data


def best_2thresholds(values, labels):
    """
    一维两阈值穷举：找出准确率最高的 t1 < t2。
    三档按 values 升序对应 自由 / 贴住 / 夹紧。
    """
    vals = sorted(set(v for v in values if not np.isnan(v)))
    if len(vals) < 3:
        return None, None, 0.0
    # 候选切点取相邻值的中点
    cands = [(vals[i] + vals[i + 1]) / 2 for i in range(len(vals) - 1)]
    n = len(values)
    best = (None, None, -1.0)
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            t1, t2 = cands[i], cands[j]
            correct = 0
            for v, lb in zip(values, labels):
                if np.isnan(v):
                    continue
                pred = 0 if v < t1 else (1 if v < t2 else 2)
                if LABELS[pred] == lb:
                    correct += 1
            acc = correct / n
            if acc > best[2]:
                best = (t1, t2, acc)
    return best


def main() -> int:
    dirs = sys.argv[1:] or ["/tmp/ga2", "/tmp/gb2", "/tmp/gc2"]
    if len(dirs) != 3:
        print("❌ 需要正好 3 个目录（自由/贴住/夹紧）")
        return 1

    print("=" * 78)
    print("三档分类准确率评估")
    print("=" * 78)
    data = collect(dirs)
    print(f"  有效样本 {len(data)} 张\n")
    if not data:
        return 1

    # 收集所有特征名
    feats = sorted({k for _, f, _ in data for k in f})

    print("  %-16s %-24s %-24s %-24s %s"
          % ("特征", "自由 中位(范围)", "贴住 中位(范围)", "夹紧 中位(范围)", "单特征最优准确率"))
    print("  " + "-" * 100)

    results = []
    for name in feats:
        rows = {}
        for lb in LABELS:
            vs = [f.get(name) for l, f, _ in data
                  if l == lb and f.get(name) is not None
                  and not np.isnan(f.get(name))]
            rows[lb] = vs
        if any(not v for v in rows.values()):
            continue
        allv = [f.get(name) for _, f, _ in data]
        labs = [l for l, _, _ in data]
        t1, t2, acc = best_2thresholds(allv, labs)
        results.append((acc, name, t1, t2))

        def rng(vs):
            return "%7.3f (%6.3f~%6.3f)" % (statistics.median(vs),
                                            min(vs), max(vs))
        print("  %-16s %-24s %-24s %-24s %.1f%%"
              % (name, rng(rows["自由"]), rng(rows["贴住"]),
                 rng(rows["夹紧"]), acc * 100))

    results.sort(reverse=True)
    print()
    print("=" * 78)
    print("最佳特征与阈值")
    print("=" * 78)
    for acc, name, t1, t2 in results[:3]:
        print("  %-16s 准确率 %.1f%%   阈值: <%.3f=%s  %.3f~%.3f=%s  >%.3f=%s"
              % (name, acc * 100, t1, LABELS[0], t1, t2, LABELS[1],
                 t2, LABELS[2]))

    if results:
        acc, name, t1, t2 = results[0]
        print()
        print(f"  ── 用最优特征 {name} 的混淆矩阵 ──")
        print("     %-8s %-8s %-8s %-8s %s" % ("真实\\预测", *LABELS, "正确率"))
        for lb in LABELS:
            row = {k: 0 for k in LABELS}
            tot = 0
            for l, f, _ in data:
                v = f.get(name)
                if v is None or np.isnan(v):
                    continue
                if l != lb:
                    continue
                pred = LABELS[0] if v < t1 else (LABELS[1] if v < t2 else LABELS[2])
                row[pred] += 1
                tot += 1
            cells = "  ".join("%6d" % row[k] for k in LABELS)
            print("     %-8s %s   %5.1f%%"
                  % (lb, cells, 100.0 * row[lb] / tot if tot else 0))
    return 0


if __name__ == "__main__":
    sys.exit(main())