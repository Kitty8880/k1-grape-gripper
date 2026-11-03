#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视觉识别率测试 —— 连拍 N 次，自动判对错，出报告。

为什么需要固化的测试流程
------------------------
"识别率 90%"这个数字要经得起追问：场景怎么摆的、分母是多少、
失败怎么算。手记 20 次必然出错（数错、场景变了没注意、算错分母）。
本脚本把流程固定下来：

    · 场景**保持不变**，连拍 N 次（这是测"稳定性"，不是测"泛化"）
    · 每次自动判：有没有识别到、位置对不对、尺寸合不合理
    · 自动算成功率并出表

判"识别正确"的条件，以及**失败要分两类**
--------------------------------------------
工程上必须把这两种失败分开，否则报告里的「识别率」会被低估：

    A. 识别失败（算法问题）—— 应该改进算法
       ① miss      : ROI 内没框到任何东西（颜色/光照/距离）
       ② lowconf   : 框到了但置信度不足

    B. 取景问题（摆放问题）—— 应该调整摆放，不是改算法
       ③ offcenter : 框到了、置信度够，但目标明显偏离画面中心

    → 报告里输出两个数：
       「识别成功率」= (总数 − miss − lowconf) / 总数   ← 反映算法能力
       「可用率」    = 全部三项都通过 / 总数            ← 反映实际能用的比例

为什么必须这样分：2026-11-03 实测时葡萄在桌面上**缓慢滚动**，
20 帧里 y 坐标从 320 漂到 575 px，前 3 帧甚至滚出了 ROI。
如果把这些算成"识别失败"，会得出 80% 的结论；
但葡萄只要在 ROI 内，16 帧全部以 **0.97 的置信度**稳定命中 ——
算法其实是 100%。**混在一起统计会自我抹黑。**

用法::

    # 场景摆好后，连拍 20 次（最常用）
    sudo python3 scripts/vision_bench.py --times 20

    # 同时统计误报（把葡萄拿开，场景保持不变）
    sudo python3 scripts/vision_bench.py --times 10 --expect-none

    # 换颜色档
    sudo python3 scripts/vision_bench.py --times 20 --color darkpurple

    # 结果存文件，贴进测试报告
    sudo python3 scripts/vision_bench.py --times 20 --csv vision_result.csv
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _HERE)

_HOME = os.path.dirname(_ROOT)
if os.path.isdir(os.path.join(_HOME, "spacemit-demo")):
    os.environ["HOME"] = _HOME

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description="视觉识别率测试")
    ap.add_argument("--times", type=int, default=20, help="连拍次数（默认 20）")
    ap.add_argument("--color", default="grape",
                    help="颜色档（grape / darkpurple / purple / green / any）")
    ap.add_argument("--min-conf", type=float, default=0.35,
                    help="置信度阈值，低于它算识别失败")
    ap.add_argument("--roi", default="0.25,0.15,0.80,0.85",
                    help="ROI 比例，与 vision_grasp.py 保持一致")
    ap.add_argument("--pos-tol", type=float, default=0.40,
                    help="位置容差：中心偏离画面中心超过该比例记为"
                         "「取景问题」（不算识别失败）。默认 0.40 很宽松，"
                         "因为实测发现葡萄会在桌面上缓慢滚动，"
                         "用严格容差会把摆放问题算成算法问题")
    ap.add_argument("--expect-none", action="store_true",
                    help="场景中**没有**目标，用于统计误报率（fp = 错误地识别到了）")
    ap.add_argument("--interval", type=float, default=0.8,
                    help="两次拍照之间的间隔秒数")
    ap.add_argument("--save-dir", default=None,
                    help="保存每次的原图（可选，便于事后复盘）")
    ap.add_argument("--csv", default=None, help="结果另存为 CSV")
    args = ap.parse_args()

    try:
        parts = [float(v) for v in args.roi.split(",")]
        if len(parts) != 4:
            raise ValueError
        args.roi = tuple(parts)
    except ValueError:
        print(f"❌ --roi 格式错误：{args.roi}")
        return 1

    try:
        from grape_detect import GrapeDetector
        from k1_camera_capture import K1Camera
    except ImportError as exc:
        print(f"❌ 导入失败：{exc}")
        print("   需要在板子上运行，且 scripts/ 目录存在")
        return 1

    det = GrapeDetector(color=args.color)
    cam = K1Camera(use_sudo=(hasattr(os, "geteuid") and os.geteuid() != 0))

    print("=" * 70)
    print("视觉识别率测试")
    print("=" * 70)
    print(f"  颜色档　：{args.color}")
    print(f"  判定阈值：置信度 ≥ {args.min_conf}，"
          f"圆度 ≥ {det.min_circularity:.2f}，"
          f"实心度 ≥ {det.min_solidity:.2f}")
    print(f"  ROI　　 ：x {args.roi[0]:.0%}~{args.roi[2]:.0%}  "
          f"y {args.roi[1]:.0%}~{args.roi[3]:.0%}")
    print(f"  次数　　：{args.times}")
    if args.expect_none:
        print(f"  模式　　：**负样本**（场景中不应有目标，统计误报）")
    else:
        print(f"  模式　　：正样本（场景中应有目标，统计漏检）")
    print()
    print("  ⚠️ 测试期间**不要移动物体、不要改变光照** ——")
    print("     这里测的是稳定性，不是泛化能力")
    print("=" * 70)
    print()

    if args.save_dir:
        try:
            os.makedirs(args.save_dir, exist_ok=True)
        except OSError as exc:
            print(f"  ⚠️ 无法创建 {args.save_dir}：{exc}")

    rows = []
    for i in range(1, args.times + 1):
        t0 = time.time()
        try:
            bgr = cam.capture()
        except Exception as exc:  # noqa: BLE001
            print(f"[{i:2}/{args.times}] 拍照失败：{exc}")
            rows.append({"i": i, "ok": 0, "conf": 0.0, "ms": 0,
                         "reason": "拍照失败", "x": 0, "y": 0, "r": 0,
                         "n": 0})
            time.sleep(args.interval)
            continue

        t1 = time.time()
        found = det.detect(bgr)
        dt = (time.time() - t1) * 1000

        h, w = bgr.shape[:2]
        x0, y0, x1, y1 = args.roi
        rx0, ry0, rx1, ry1 = x0 * w, y0 * h, x1 * w, y1 * h

        inside = [d for d in found
                  if rx0 <= d.x <= rx1 and ry0 <= d.y <= ry1]
        best = max(inside, key=lambda d: d.area) if inside else None
        conf = (min(1.0, max(0.0, (best.circularity - 0.5) / 0.5))
                if best else 0.0)

        # ---- 判定 ----
        # category: ok / miss / lowconf / offcenter / fp
        reason = ""
        cat = "ok"
        if args.expect_none:
            # 负样本：识别到（且在 ROI 内且置信度够）就是误报
            if best is not None and conf >= args.min_conf:
                cat, reason = "fp", "误报（本不该有目标）"
        else:
            # 正样本
            if best is None:
                cat = "miss"
                if found:
                    reason = f"ROI 外有个候选（共 {len(found)} 个）"
                else:
                    reason = "没框到任何东西（颜色/光照/距离）"
            elif conf < args.min_conf:
                cat = "lowconf"
                reason = f"置信度不足（{conf:.2f} < {args.min_conf:.2f}）"
            else:
                # 位置偏差 —— 归为「取景问题」，不算识别失败
                dx = (best.x - w / 2) / w
                dy = (best.y - h / 2) / h
                if abs(dx) > args.pos_tol or abs(dy) > args.pos_tol:
                    cat = "offcenter"
                    reason = (f"取景偏（偏离中心 {dx:+.0%}, {dy:+.0%}，"
                              f"容差 ±{args.pos_tol:.0%}）")
        ok = 1 if cat == "ok" else 0

        mark = "✅" if ok else ("◻️" if cat == "offcenter" else "❌")
        print(f"[{i:2}/{args.times}] {mark}  "
              f"候选{len(found)}个  选中={'(%d,%d) r=%d' % (best.x, best.y, best.radius) if best else '无'}"
              f"  conf={conf:.2f}  检测{dt:.0f}ms"
              + (f"  ← {reason}" if reason else ""))

        if args.save_dir:
            try:
                import cv2
                cv2.imwrite(os.path.join(args.save_dir,
                                         f"{i:02d}_{cat}.jpg"), bgr)
            except Exception:  # noqa: BLE001
                pass

        rows.append({"i": i, "ok": ok, "cat": cat, "conf": conf, "ms": dt,
                     "reason": reason,
                     "x": best.x if best else 0,
                     "y": best.y if best else 0,
                     "r": best.radius if best else 0,
                     "n": len(found)})
        time.sleep(args.interval)

    # ---------------- 报告 ----------------
    n = len(rows)
    okn = sum(r["ok"] for r in rows)
    rate = okn / n if n else 0.0
    times = [r["ms"] for r in rows if r["ms"] > 0]
    confs = [r["conf"] for r in rows if r["ok"]]

    print()
    print()
    print("=" * 70)
    print("测试结果")
    print("=" * 70)
    if args.expect_none:
        print(f"  模式　　　：负样本（统计误报）")
        print(f"  正确率　　：{rate*100:5.1f}%   （{okn}/{n} 次正确未误报）")
        print(f"  误报次数　：{n-okn} / {n}")
    else:
        # ★ 识别失败与取景问题分开统计
        miss = [r for r in rows if r.get("cat") in ("miss", "lowconf")]
        off = [r for r in rows if r.get("cat") == "offcenter"]
        rec_rate = (n - len(miss)) / n if n else 0.0
        print(f"  模式　　　：正样本")
        print()
        print(f"  ① 识别成功率　：{rec_rate*100:5.1f}%   "
              f"（{n-len(miss)}/{n}）  ← 反映算法能力")
        print(f"  ② 可用率　　　：{rate*100:5.1f}%   "
              f"（{okn}/{n}）  ← 反映实际能用的比例")
        print()
        print(f"     其中 取景偏（摆放问题，非算法问题）：{len(off)} 次")
        if confs:
            print(f"     成功时置信度：平均 {statistics.mean(confs):.2f}　"
                  f"最低 {min(confs):.2f}")
    if times:
        print()
        print(f"  检测耗时：平均 {statistics.mean(times):.0f}ms　"
              f"中位 {statistics.median(times):.0f}ms　"
              f"最小 {min(times):.0f}ms　最大 {max(times):.0f}ms")

    # 失败原因归类
    fails = [r for r in rows if not r["ok"]]
    if fails:
        print()
        print(f"  未通过明细（{len(fails)} 次）")
        print("  " + "-" * 60)
        for r in fails:
            tag = "取景" if r.get("cat") == "offcenter" else "识别"
            print(f"    #{r['i']:2}  [{tag}]  {r['reason']}")
        from collections import Counter
        cat = Counter(r["reason"].split("（")[0] for r in fails)
        print()
        print("  原因归类")
        print("  " + "-" * 60)
        for k, v in cat.most_common():
            print(f"    {k:<28} {v} 次")

    # 评分档
    print()
    print("  " + "-" * 60)
    # 评分用的是「识别准确率」，所以按识别成功率（不含取景问题）判档
    grade_val = rate if args.expect_none else rec_rate
    if grade_val >= 0.90:
        g = "≥90% → 「优秀」(9 分)"
    elif grade_val >= 0.80:
        g = "≥80% → 「良好」(6 分)"
    elif grade_val >= 0.70:
        g = "≥70% → 「一般」(3 分)"
    else:
        g = "<70% → 0 分"
    print(f"  比赛评分参考：{g}")
    if not args.expect_none and len(off) > 0:
        print(f"  注：{len(off)} 次取景偏不计入识别率 —— "
              f"葡萄在桌面上滚动导致，属摆放问题")

    if args.csv:
        try:
            with open(args.csv, "w", encoding="utf-8-sig") as fh:
                fh.write("序号,通过,分类,置信度,检测ms,选中x,选中y,"
                         "半径,候选数,说明\n")
                for r in rows:
                    fh.write(f"{r['i']},{r['ok']},{r.get('cat','')},"
                             f"{r['conf']:.3f},{r['ms']:.0f},{r['x']},"
                             f"{r['y']},{r['r']},{r['n']},{r['reason']}\n")
            print(f"\n  明细已存：{args.csv}")
        except OSError as exc:
            print(f"\n  ⚠️ 写 CSV 失败：{exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())