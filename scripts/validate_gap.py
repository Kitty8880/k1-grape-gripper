#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P1 验证：葡萄边缘的「极坐标断口」能否区分 自由 / 贴住 / 夹紧 三种状态。

这个脚本只做一件事，但它是整个新方案的**生死判断**
------------------------------------------------
新方案的核心假设是：

    葡萄被夹住时，它的边缘会被夹爪"截断"，
    而截断的大小（角度覆盖率）随夹紧程度单调变化。

如果能用「边缘点的角度覆盖」区分三种状态，方案成立；
如果三组数值重叠，方案必须换思路（例如换照明、换视角）。

算法
----
1. HSV 找葡萄 → 掩码 + 质心
2. Canny 找边缘 → 只保留落在葡萄掩码内的边缘点
3. 把边缘点转成以质心为极点的角度 `θ`
4. 排序后求相邻角度差，取**最大间隙** `max_gap`
5. 统计大间隙个数（>20° 才算一个"断口"）
6. 对未被断口打断的主弧拟合椭圆 → `aspect`（形变度）

判据（尚未标定，先用这三组数据看分布）
------------------------------------
    自由    max_gap 应 < 30°，断口 0 个
    贴住    max_gap 30~60°，断口 1~2 个
    夹紧    max_gap > 60°，aspect < 0.95

用法::

    python scripts/validate_gap.py --root debug
    python scripts/validate_gap.py --root debug --save-overlay
"""
from __future__ import annotations

import argparse
import glob
import os
import statistics
import sys

import cv2
import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

#: 葡萄的 HSV 范围（与 scripts/grape_detect.py 里的 darkpurple 一致）
GRAPE_HSV = ((120, 15, 15), (180, 255, 130))

#: 夹具：只保留葡萄掩码内的边缘点，并去掉离质心太远/太近的
MIN_POINTS = 30


def imread_unicode(path):
    """
    读取图片，兼容中文路径。

    ⚠️ cv2.imread 在 Windows 上**不支持非 ASCII 路径** ——
       本项目目录是 D:\文档\... ，直接 imread 会静默返回 None，
       表现为"所有图片都读不出来"，很容易误判成图像有问题。
       用 np.fromfile + cv2.imdecode 绕过。
    """
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:  # noqa: BLE001
        return None


def grape_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array(GRAPE_HSV[0], np.uint8),
                    np.array(GRAPE_HSV[1], np.uint8))
    # 形态学闭运算，把反光造成的小洞补上（★ 这是抗反光的第一步）
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    return m


def largest_blob(mask, min_circ=0.45, min_aspect=0.55, min_area=1500):
    """
    在掩码里挑**最像葡萄**的连通域。

    ⚠️ 不能只按面积挑最大 —— 实测踩过：
       set_a 里面积最大的是一根 564×103 的扁长横条（桌面阴影/缝隙），
       真正的葡萄是第 2 大（219×242，圆形）。
       只按面积挑会把横条当成葡萄，后面所有指标全废。

    所以加三个几何约束（和原 GrapeDetector 的思路一致）：
        圆度 circularity = 4πA/P²   ≥ min_circ
        外接框长宽比                 ≥ min_aspect
        面积                        ≥ min_area

    返回 (blob, 质心, 面积, 圆度)。
    """
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return None, None, 0, 0.0

    best = None
    for idx in range(1, n):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        if w <= 0 or h <= 0:
            continue
        aspect = min(w, h) / max(w, h)
        if aspect < min_aspect:
            continue

        blob = (labels == idx).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        perim = cv2.arcLength(c, True)
        if perim <= 0:
            continue
        circ = 4.0 * np.pi * area / (perim * perim)
        if circ < min_circ:
            continue

        # 满足约束的里挑**面积最大**的（更近的那个）
        if best is None or area > best[2]:
            best = (blob, (float(cents[idx][0]), float(cents[idx][1])),
                    area, float(circ))

    if best is None:
        return None, None, 0, 0.0
    return best


def edge_points(bgr, blob, cx, cy):
    """
    在葡萄区域内做 Canny，返回边缘点的极角数组和半径数组。

    ⚠️ 只在掩码内取边，避免把桌面/夹爪边缘误算进来。
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # 稍微模糊再 Canny，抑制反光造成的碎边
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.bitwise_and(edges, blob)

    ys, xs = np.nonzero(edges)
    if xs.size < MIN_POINTS:
        return None, None

    dx = xs.astype(np.float32) - cx
    dy = ys.astype(np.float32) - cy
    ang = np.arctan2(dy, dx) % (2 * np.pi)
    rad = np.hypot(dx, dy)
    return ang, rad


def flat_metrics(blob, eps_ratio=0.02, min_seg_ratio=0.06):
    """
    分析轮廓上的**直边** —— 这才是判断"被夹紧"的正确边缘判据。

    为什么放弃"极坐标角度断口"
    --------------------------
    最初设想：葡萄被夹住时边缘会被截断，所以角度覆盖会小于 360°。

    **但这是错的。** 任何闭合轮廓，它自己的边界点必然覆盖 360°：
    葡萄被夹住时，可见的紫色区域变小，可它的边界**仍然是一圈闭合曲线**。
    实测印证了这个错误：set_a（自由状态）算出 max_gap 中位 112°、
    最大 175° —— 自由葡萄不可能有 175° 的空缺，
    说明量本身没有物理意义。

    正确的物理事实
    --------------
    球体被平板压住时，接触处**变成一段直线**：

        自由                 夹紧
        ╭────╮             ╭────╮
       │      │            ┃      ┃   ← 上下变成直边
        ╰────╯             ╰────╯      且互相平行

    所以可测的量是：
        flatness   最长直边长度 / 周长        （自由 ~0.05，夹紧 >0.15）
        n_flat     长度 > min_seg_ratio×周长 的直边数
        parallel   是否存在两条大致平行且相对的直边

    返回 (flatness, n_flat, parallel, max_seg_px, perim_px)
    """
    cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 20:
        return None

    perim = cv2.arcLength(c, True)
    if perim < 50:
        return None

    # 多边形简化：eps 越大，留下的顶点越少、直边越明显
    approx = cv2.approxPolyDP(c, eps_ratio * perim, True).reshape(-1, 2)
    if len(approx) < 3:
        return None

    # 逐条边的长度和方向
    segs = []
    n = len(approx)
    for i in range(n):
        p1 = approx[i].astype(np.float32)
        p2 = approx[(i + 1) % n].astype(np.float32)
        d = p2 - p1
        L = float(np.hypot(d[0], d[1]))
        ang = float(np.degrees(np.arctan2(d[1], d[0]))) % 180.0
        segs.append((L, ang))

    max_seg = max(L for L, _ in segs)
    flatness = max_seg / perim
    n_flat = sum(1 for L, _ in segs if L > min_seg_ratio * perim)

    # 找两条"够长且平行"的边（方向差 <20°，长度都超过阈值）
    long_segs = [(L, a) for L, a in segs if L > min_seg_ratio * perim]
    parallel = False
    for i in range(len(long_segs)):
        for j in range(i + 1, len(long_segs)):
            da = abs(long_segs[i][1] - long_segs[j][1])
            da = min(da, 180.0 - da)
            if da < 20.0:
                parallel = True
                break
        if parallel:
            break

    return flatness, n_flat, parallel, max_seg, perim


def aspect_of(blob):
    """对葡萄掩码的最大轮廓拟合椭圆，返回短轴/长轴。"""
    cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 5:
        return None
    (_, _), (ma, mi), _ = cv2.fitEllipse(c)
    if ma <= 0:
        return None
    return float(min(ma, mi) / max(ma, mi))


def analyze(path, save_overlay=False):
    bgr = imread_unicode(path)
    if bgr is None:
        return {"file": os.path.basename(path), "ok": False,
                "why": "读不出来"}
    m = grape_mask(bgr)
    blob, c, area, circ = largest_blob(m)
    if blob is None:
        return {"file": os.path.basename(path), "ok": False,
                "why": "没检出葡萄"}

    fm = flat_metrics(blob)
    if fm is None:
        return {"file": os.path.basename(path), "ok": False,
                "why": "轮廓点太少", "area": area}
    flatness, n_flat, parallel, max_seg, perim = fm
    asp = aspect_of(blob)

    if save_overlay:
        cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
        c0 = max(cnts, key=cv2.contourArea)
        perim0 = cv2.arcLength(c0, True)
        ap = cv2.approxPolyDP(c0, 0.02 * perim0, True)
        vis = bgr.copy()
        cv2.drawContours(vis, [c0], -1, (0, 255, 0), 2)
        cv2.polylines(vis, [ap], True, (0, 0, 255), 2)
        for pt in ap.reshape(-1, 2):
            cv2.circle(vis, tuple(int(v) for v in pt), 5, (255, 0, 0), -1)
        cv2.circle(vis, (int(c[0]), int(c[1])), 4, (0, 255, 255), -1)
        out = os.path.splitext(path)[0] + "_gap.jpg"
        okc, buf = cv2.imencode(".jpg", vis)
        if okc:
            buf.tofile(out)

    return {"file": os.path.basename(path), "ok": True,
            "cx": c[0], "cy": c[1], "area": area, "circ": circ,
            "flatness": flatness, "n_flat": n_flat,
            "parallel": parallel, "max_seg": max_seg,
            "perim": perim, "aspect": asp}


def main() -> int:
    ap = argparse.ArgumentParser(description="P1 验证：极坐标断口能否区分状态")
    ap.add_argument("--root", default="debug")
    ap.add_argument("--save-overlay", action="store_true",
                    help="输出 *_gap.jpg，绿点是边缘、红点是质心")
    args = ap.parse_args()

    groups = []
    for d in sorted(glob.glob(os.path.join(args.root, "set_*"))):
        if os.path.isdir(d):
            imgs = sorted(glob.glob(os.path.join(d, "*.jpg")))
            imgs = [p for p in imgs if "_gap" not in p]
            if imgs:
                groups.append((os.path.basename(d), imgs))

    if not groups:
        print(f"❌ {args.root} 下没找到 set_* 目录")
        return 1

    print("=" * 78)
    print("P1 验证：葡萄边缘极坐标断口")
    print("=" * 78)
    print(f"  HSV 范围：{GRAPE_HSV}")
    print(f"  Canny：40 / 120（需按实拍光照标定）")
    print()

    all_stats = {}
    for name, imgs in groups:
        rows = []
        for p in imgs:
            r = analyze(p, args.save_overlay)
            if r:
                rows.append(r)
        oks = [r for r in rows if r["ok"]]
        fails = [r for r in rows if not r["ok"]]
        all_stats[name] = oks

        print(f"── {name}　共 {len(rows)} 张，成功 {len(oks)}，失败 {len(fails)}")
        if fails:
            reasons = {}
            for f in fails:
                reasons[f.get("why", "?")] = reasons.get(f.get("why", "?"), 0) + 1
            for w, n in reasons.items():
                print(f"     ⚠️ {w}：{n} 张")

        if oks:
            fl = [r["flatness"] for r in oks]
            nf = [r["n_flat"] for r in oks]
            pp = sum(1 for r in oks if r["parallel"])
            asp = [r["aspect"] for r in oks if r["aspect"]]
            ar = [r["area"] for r in oks]
            ci = [r.get("circ", 0) for r in oks]
            print(f"     面积     中位 {statistics.median(ar):7.0f} px  "
                  f"圆度 {statistics.median(ci):.2f}")
            print(f"     flatness 中位 {statistics.median(fl):.3f}  "
                  f"范围 {min(fl):.3f}~{max(fl):.3f}")
            print(f"     直边数   中位 {statistics.median(nf):.0f}")
            print(f"     有平行对 {pp}/{len(oks)} 张")
            if asp:
                print(f"     aspect   中位 {statistics.median(asp):.3f}  "
                      f"范围 {min(asp):.3f}~{max(asp):.3f}")
            print(f"     逐张 flatness：" + " ".join(f"{v:.2f}" for v in fl))
        print()

    # ---------- 可分离性判断 ----------
    print("=" * 78)
    print("可分离性判断")
    print("=" * 78)
    names = [n for n, _ in groups if all_stats.get(n)]
    if len(names) < 2:
        print("  ❌ 有效组不足 2 个，无法比较")
        return 0

    med = {n: statistics.median([r["flatness"] for r in all_stats[n]])
           for n in names}
    lo = {n: min(r["flatness"] for r in all_stats[n]) for n in names}
    hi = {n: max(r["flatness"] for r in all_stats[n]) for n in names}

    print(f"  {'组':<10}{'flatness中位':>14}{'最小':>10}{'最大':>10}")
    for n in names:
        print(f"  {n:<10}{med[n]:>14.3f}{lo[n]:>10.3f}{hi[n]:>10.3f}")
    print()

    order = sorted(names, key=lambda n: med[n])
    ok = True
    for i in range(len(order) - 1):
        a, b = order[i], order[i + 1]
        overlap = max(0.0, hi[a] - lo[b])
        sep = lo[b] - hi[a]
        if sep > 0:
            print(f"  ✅ {a} < {b} 完全分开（间隔 {sep:.3f}）")
        elif overlap < 10:
            print(f"  🔶 {a} < {b} 轻微重叠 {overlap:.3f}（可接受）")
        else:
            print(f"  ❌ {a} < {b} 重叠 {overlap:.3f} —— 分不开")
            ok = False

    print()
    if ok:
        print("  ★ 结论：max_gap 能区分状态，**方案成立**")
        print(f"  ★ 建议阈值（取相邻两组的中间）：")
        for i in range(len(order) - 1):
            a, b = order[i], order[i + 1]
            print(f"       {a} / {b} 之间：{(hi[a] + lo[b]) / 2:.3f}")
    else:
        print("  ★ 结论：**当前图像不可分**。先别改算法，按顺序检查：")
        print("       1. 图像是否失焦（边缘会糊成一片）")
        print("       2. 葡萄是否被夹爪或传感器遮挡（遮挡 = 假断口）")
        print("       3. 背景是否够亮（深色背景 -> 边缘弱）")
        print("       4. Canny 阈值是否合适（--save-overlay 看绿点）")
    return 0


if __name__ == "__main__":
    sys.exit(main())