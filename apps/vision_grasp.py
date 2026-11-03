#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视觉抓取 —— 拍照 → 识别 → 交叉校验 → 抓取。

在多模态链路里的位置
--------------------
    语音（ASR）  →  说出"要夹什么"      ← 意图
    视觉（CV）   →  看到"实际是什么"    ← 事实
                      ↓
                  【交叉校验】
                      ↓
    力控         →  用对应的力去夹

**为什么两个模态都要有**

    只靠语音：用户说葡萄，系统就按葡萄的力去夹 —— 哪怕面前是鸡蛋 → 夹碎
    只靠视觉：说"夹那个东西"就没法确定意图
    两者结合：语音给意图，视觉给事实，不一致就拒绝执行

这也是比赛"多模态协同"评分项的落地方式。

当前识别的实现
--------------
用的是传统 CV（HSV 颜色 + 凸包圆度/实心度），**不是深度学习分类器**。
原因：分类模型还在训练中（数据集刚采完）。
本脚本的接口已经按"分类器"的形状设计好了 ——
训练完只需把 `classify()` 换成 ONNX 推理，其他逻辑不用动。

用法::

    # 只识别，不抓取（先把相机和检测调通）
    sudo python3 apps/vision_grasp.py --no-execute

    # 识别 + 抓取
    sudo python3 apps/vision_grasp.py

    # 带语音给的"期望物体"，做交叉校验
    sudo python3 apps/vision_grasp.py --expect grape

    # 看检测过程图（输出到 /tmp）
    sudo python3 apps/vision_grasp.py --no-execute --save-debug /tmp/vision

输出
----
    ✅ 看到葡萄 → （校验通过）→ 抓取
    ⚠️ 没看到葡萄 → 拒绝执行
    ⚠️ 语音说葡萄但看到别的东西 → 拒绝执行（安全）
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

from grape_gripper.config import Config, default_config_path   # noqa: E402

# 期望物体 → 力策略里的类别名
_ALIAS = {
    "grape": "grape", "葡萄": "grape", "提子": "grape",
    "egg": "egg", "鸡蛋": "egg", "蛋": "egg",
    "pingpong": "pingpong", "乒乓球": "pingpong", "球": "pingpong",
    "": "unknown", "unknown": "unknown", "未知": "unknown",
}

_CN = {"grape": "葡萄", "egg": "鸡蛋", "pingpong": "乒乓球",
       "unknown": "未知物体"}


# ---------------------------------------------------------------------------
# 拍照
# ---------------------------------------------------------------------------

def grab_frame(args):
    """
    取一帧 BGR 图像。

    ★ `--frame <路径>` 时直接读文件，不再调摄像头。
      这是为了支持**流水线**：上层（product_mode）可以在做 ASR 的同时
      就把照片拍好存到 /tmp，等 ASR 出来再调用本脚本并传入这张图。
      拍照 1.0 s 就从关键路径上消失了（实测响应延迟 3.8 s → 2.8 s）。
    """
    import cv2

    pre = getattr(args, "frame", None)
    if pre:
        if not os.path.exists(pre):
            raise FileNotFoundError(f"--frame 指定的文件不存在：{pre}")
        bgr = cv2.imread(pre)
        if bgr is None:
            raise ValueError(f"读不出图片：{pre}")
        print(f"  ✅ 使用预拍摄的帧：{pre}")
        return bgr

    from k1_camera_capture import K1Camera

    need_sudo = hasattr(os, "geteuid") and os.geteuid() != 0
    cam = K1Camera(use_sudo=need_sudo)
    return cam.capture()


# ---------------------------------------------------------------------------
# 识别
# ---------------------------------------------------------------------------

def explain_candidates(bgr, args):
    """
    把阈值全部放开，看看到底有多少候选轮廓、各自什么指标。

    为什么需要：检测"没看到葡萄"时，真正的原因可能是
        ① HSV 颜色没框到（mask 全黑）
        ② 框到了但圆度不够（葡萄被遮挡/反光/变形）
        ③ 框到了但实心度不够（边缘不连续）
        ④ 面积超出区间（太远/太近）
        ⑤ **位置在 ROI 之外**（摄像头没对准）
    不打印这些指标就只能瞎猜。
    """
    import cv2
    from grape_detect import GrapeDetector

    loose = GrapeDetector(color=args.color, min_area=0, max_area=10 ** 9,
                          min_circularity=0.0, min_solidity=0.0)
    # ⚠️ 判定阈值要用"实际生效值" —— args 里可能是 None（表示按颜色档取默认），
    #    直接和 None 比较会抛 TypeError（踩过）。
    ref = GrapeDetector(color=args.color)
    circ_th = ref.min_circularity
    sol_th = ref.min_solidity

    found = loose.detect(bgr)
    print(f"     诊断：放开全部阈值后有 {len(found)} 个候选轮廓")
    if not found:
        print("       → 一个都没有，说明 **HSV 颜色区间没框到任何东西**")
        print(f"         当前颜色档：{args.color}")
        print("         处理：换 --color 或调光照")
        return

    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = args.roi
    rx0, ry0, rx1, ry1 = x0 * w, y0 * h, x1 * w, y1 * h
    print(f"     生效阈值：圆度 ≥ {circ_th:.2f}  实心度 ≥ {sol_th:.2f}")
    print(f"     ROI：x {rx0:.0f}~{rx1:.0f}  y {ry0:.0f}~{ry1:.0f}")
    for i, d in enumerate(found[:8]):
        hull = cv2.convexHull(d.contour)
        harea = float(cv2.contourArea(hull)) + 1e-6
        sol = d.area / harea
        bad = []
        if not (rx0 <= d.x <= rx1 and ry0 <= d.y <= ry1):
            bad.append("在ROI外")
        if d.area < 1500:
            bad.append(f"面积小({d.area:.0f})")
        if d.area > 400000:
            bad.append(f"面积大({d.area:.0f})")
        if d.circularity < circ_th:
            bad.append(f"圆度低({d.circularity:.2f})")
        if sol < sol_th:
            bad.append(f"实心度低({sol:.2f})")
        print(f"       #{i+1} 中心({d.x:.0f},{d.y:.0f}) 半径{d.radius:.0f} "
              f"面积{d.area:.0f} 圆度{d.circularity:.2f} 实心度{sol:.2f}"
              + ("   ❌ " + "、".join(bad) if bad else "   ✅ 应通过"))


def draw_roi(bgr, args):
    """在图上画出 ROI 框，方便肉眼确认该把葡萄放哪。"""
    import cv2

    out = bgr.copy()
    h, w = out.shape[:2]
    x0, y0, x1, y1 = args.roi
    p0 = (int(x0 * w), int(y0 * h))
    p1 = (int(x1 * w), int(y1 * h))
    cv2.rectangle(out, p0, p1, (0, 255, 0), 3)
    cv2.putText(out, "ROI: put the grape in here",
                (p0[0] + 8, p0[1] + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.line(out, (w // 2, h // 2 - 20), (w // 2, h // 2 + 20),
             (0, 255, 0), 2)
    cv2.line(out, (w // 2 - 20, h // 2), (w // 2 + 20, h // 2),
             (0, 255, 0), 2)
    return out


def save_debug(bgr, args, dets, tag: str):
    """保存原件 + 标注图 + 颜色掩码，便于人工检查。"""
    import cv2
    from grape_detect import GrapeDetector

    d = args.save_debug
    if not d:
        return None
    try:
        os.makedirs(d, exist_ok=True)
    except OSError as exc:
        print(f"      ⚠️ 无法创建调试目录 {d}：{exc}")
        return None

    stamp = time.strftime("%m%d_%H%M%S")
    base = os.path.join(d, f"{tag}_{stamp}")
    try:
        cv2.imwrite(base + "_raw.jpg", bgr)
        det = GrapeDetector(color=args.color)
        if dets:
            ann = det.annotate(bgr, dets)
        else:
            ann = bgr.copy()
        # ★ 把 ROI 框画上去 —— 让人一眼看出该把葡萄放哪
        cv2.imwrite(base + "_det.jpg", draw_roi(ann, args))
        mask = det.mask_of(bgr)
        cv2.imwrite(base + "_mask.png", mask)
        # last_* 固定名字，方便反复查看最新一张
        cv2.imwrite(os.path.join(d, "last_raw.jpg"), bgr)
        cv2.imwrite(os.path.join(d, "last_det.jpg"),
                    draw_roi(ann, args))
        cv2.imwrite(os.path.join(d, "last_mask.png"), mask)
        return base
    except Exception as exc:  # noqa: BLE001
        print(f"      ⚠️ 保存调试图失败：{exc}")
        return None


def classify(bgr, args):
    """
    返回 (类别名, 置信度, 详情列表, 耗时)。

    ⚠️ 2026-11-02 实测踩坑后的两条修正：

      ① **选优标准从"圆度最高"改成"面积最大（在有 ROI 约束的前提下）"**
         原因：手指甲又小又圆，圆度 0.90；葡萄又大又不规则，圆度 0.48。
         按圆度选会把手指甲当葡萄（实测真的发生了，报告"置信度 0.81"）。

      ② **加中心 ROI 约束**
         原因：桌面场景里深色/紫色物体太多（胶带卷、键盘、手机、手），
         不加位置约束会选中画面边角的杂物。
         根据使用方式（把葡萄放在两指之间/画面中央），
         只接受中心区域内的候选是合理且有效的约束。
    """
    import cv2
    from grape_detect import GrapeDetector

    det = GrapeDetector(
        color=args.color,
        min_circularity=args.min_circularity,
        min_solidity=args.min_solidity,
    )
    t0 = time.time()
    found = det.detect(bgr)
    dt = time.time() - t0

    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = args.roi
    rx0, ry0 = x0 * w, y0 * h
    rx1, ry1 = x1 * w, y1 * h

    details = []
    for d in found:
        inside = (rx0 <= d.x <= rx1) and (ry0 <= d.y <= ry1)
        conf = min(1.0, max(0.0, (d.circularity - 0.5) / 0.5))
        details.append({"x": d.x, "y": d.y, "radius": d.radius,
                        "area": d.area, "circularity": d.circularity,
                        "conf": conf, "in_roi": inside})

    inside = [d for d in details if d["in_roi"]]
    if inside:
        # ★ 在 ROI 内选**面积最大**的，不选圆度最高的
        best = max(inside, key=lambda d: d["area"])
        return "grape", best["conf"], details, dt

    return "unknown", 0.0, details, dt


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="视觉抓取（拍照→识别→校验→抓取）")
    ap.add_argument("--config", default=None)
    ap.add_argument("--expect", default=None,
                    help="语音给的期望物体（grape/egg/pingpong/葡萄…）。"
                         "给了就做交叉校验，不一致则拒绝执行")
    ap.add_argument("--color", default="grape",
                    choices=["grape", "darkpurple", "purple", "red",
                             "green", "any"],
                    help="HSV 检测用的颜色档。grape = 紫+绿通吃"
                         "（2026-11-02 实测标定）；"
                         "只想检紫葡萄用 darkpurple，想调试单个区间用别的")
    ap.add_argument("--min-circularity", type=float, default=None,
                    help="圆度阈值。不填则按颜色档取默认值"
                         "（darkpurple=0.45，purple=0.60）")
    ap.add_argument("--min-solidity", type=float, default=None,
                    help="实心度阈值。不填则按颜色档取默认值"
                         "（darkpurple=0.40，purple=0.72）")
    ap.add_argument("--roi", default="0.25,0.15,0.80,0.85",
                    help="只在这个中心区域内找目标，格式 x0,y0,x1,y1，"
                         "取值 0~1 是相对画面尺寸的比例。"
                         "默认只保留画面中央 55%%×70%% 的范围 —— "
                         "因为实际使用时葡萄是放在两指之间/画面中央的，"
                         "而桌面上的杂物（胶带、键盘、手）都在边角。"
                         "填 0,0,1,1 关闭这个约束")
    ap.add_argument("--min-conf", type=float, default=0.35,
                    help="置信度低于它判为 unknown（深色葡萄圆度天然低，"
                         "不要设太高）")
    ap.add_argument("--frame", default=None,
                    help="用这张已拍好的图代替实时拍照。"
                         "用于流水线：上层在跑 ASR 时先把照片拍好，"
                         "可以把 1.0 s 的拍照时间从关键路径上省掉")
    ap.add_argument("--no-execute", action="store_true",
                    help="只识别，不抓取")
    ap.add_argument("--save-debug", default=None,
                    help="把每次拍照的原件/标注图/掩码存到这个目录"
                         "（推荐 debug/，方便逐张检查）")
    ap.add_argument("--grasp-arg", action="append", default=[],
                    help="透传给 grasp_with_verify.py 的参数，可重复")
    args = ap.parse_args()

    # 解析 ROI（x0,y0,x1,y1 比例）
    try:
        parts = [float(v) for v in args.roi.split(",")]
        if len(parts) != 4:
            raise ValueError
        args.roi = tuple(parts)
    except ValueError:
        print(f"❌ --roi 格式错误：{args.roi}（应为 x0,y0,x1,y1，例如 0.25,0.15,0.8,0.85）")
        return 1

    cfg = Config.load(args.config or default_config_path())

    print("=" * 74)
    print("视觉抓取 —— 拍照 → 识别 → 交叉校验 → 抓取")
    print("=" * 74)

    # ---------- ① 拍照 ----------
    print("\n[1/4] 拍照 ...")
    try:
        frame = grab_frame(args)
    except Exception as exc:  # noqa: BLE001
        print(f"  ❌ 拍照失败：{exc}")
        print("     排查：sudo cam-test /usr/share/camera_json/csi1_camera_auto.json")
        return 1
    print(f"  ✅ {frame.shape[1]}x{frame.shape[0]}")

    # ---------- ② 识别 ----------
    print("\n[2/4] 识别 ...")
    try:
        obj, conf, details, dt = classify(frame, args)
    except Exception as exc:  # noqa: BLE001
        print(f"  ❌ 识别失败：{exc}")
        return 1

    if obj != "unknown" and conf < args.min_conf:
        print(f"  检测到疑似目标但置信度 {conf:.2f} < {args.min_conf:.2f}"
              f" → 判为未知")
        obj, conf = "unknown", conf

    if obj == "grape":
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = args.roi
        print(f"  ✅ 看到{_CN[obj]}   置信度 {conf:.2f}   "
              f"耗时 {dt*1000:.0f}ms")
        print(f"     ROI：x {x0*w:.0f}~{x1*w:.0f}  y {y0*h:.0f}~{y1*h:.0f} 内的候选")
        for i, d in enumerate(details[:8]):
            flag = "✅在ROI" if d["in_roi"] else "✗在ROI外"
            print(f"       #{i+1} 中心({d['x']:.0f},{d['y']:.0f}) "
                  f"半径 {d['radius']:.0f}px 圆度 {d['circularity']:.2f} "
                  f"面积 {d['area']:.0f}  {flag}")
    else:
        print(f"  ⚠️ 没看到葡萄（耗时 {dt*1000:.0f}ms）")
        print("     判定条件（生效值）："
              f"圆度 ≥ {args.min_circularity if args.min_circularity is not None else '按颜色档默认'}，"
              f"实心度 ≥ {args.min_solidity if args.min_solidity is not None else '按颜色档默认'}，"
              f"颜色 {args.color}")
        # ★ 自动诊断：告诉用户到底卡在哪一条，而不是让人瞎猜
        try:
            explain_candidates(frame, args)
        except Exception as exc:  # noqa: BLE001
            print(f"     （诊断失败：{exc}）")

    # ★ 每次拍照都存图 —— 不管成功失败，方便事后逐张检查
    saved = None
    if args.save_debug:
        saved = save_debug(frame, args, None if obj == "unknown" else
                           __import__("grape_detect").GrapeDetector(
                               color=args.color).detect(frame),
                           "ok" if obj != "unknown" else "miss")
        if saved:
            print(f"     调试图已存：{saved}_raw.jpg / _det.jpg / _mask.png")

    # ---------- ③ 交叉校验 ----------
    print("\n[3/4] 交叉校验 ...")
    expect = _ALIAS.get((args.expect or "").strip().lower(),
                        _ALIAS.get((args.expect or "").strip(), None))
    ok = True
    if expect is None:
        print("  （没给 --expect，跳过校验）")
    elif expect == "unknown":
        print("  语音没指定物体 → 按视觉结果处理")
    elif obj == "unknown":
        print("  ❌ 语音说要夹"
              f"{_CN.get(expect, expect)}，但视觉没看到目标")
        print("     → 拒绝执行（避免盲夹）")
        ok = False
    elif obj != expect:
        print("╔" + "═" * 66 + "╗")
        print("║  ⚠️  目标不一致，已停止执行" + " " * 37 + "║")
        print("╠" + "═" * 66 + "╣")
        print(f"║  语音说要夹：  {_CN.get(expect, expect):<20}"
              f"{' ' * (66 - 14 - len(_CN.get(expect, expect)) * 2)}║")
        print(f"║  视觉看到的是：{_CN.get(obj, obj):<20}"
              f"（置信度 {conf:.2f}）{' ' * 30}║")
        print("╠" + "═" * 66 + "╣")
        print("║  为安全起见不执行。请确认："
              + " " * 38 + "║")
        print(f"║    · 说「夹{_CN.get(obj, obj)}」改用对应的力策略"
              + " " * 27 + "║")
        print("║    · 或把目标换成语音里说的物体"
              + " " * 31 + "║")
        print("╚" + "═" * 66 + "╝")
        ok = False
    else:
        print(f"  ✅ 一致（都是{_CN.get(obj, obj)}）")

    if not ok:
        print("\n视觉流程结束（未执行抓取）")
        return 1

    if args.no_execute:
        print("\n（--no-execute：不执行抓取）")
        return 0

    # ---------- ④ 抓取 ----------
    print("\n[4/4] 执行抓取 ...")
    import subprocess
    obj_for_policy = expect if expect not in (None, "unknown") else obj
    script = os.path.join(_HERE, "grasp_with_verify.py")
    cmd = [sys.executable, script, "--object", obj_for_policy,
           "--min-force", "1.2"]
    cmd.extend(args.grasp_arg)
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        cmd = ["sudo"] + cmd
    print(f"  $ {' '.join(cmd)}")
    print()
    try:
        rc = subprocess.call(cmd)
    except Exception as exc:  # noqa: BLE001
        print(f"  ❌ 启动抓取失败：{exc}")
        return 1

    print()
    print("=" * 74)
    if rc == 0:
        print("  ✅ 视觉抓取流程：成功")
    else:
        print(f"  ❌ 视觉抓取流程：失败（退出码 {rc}）")
    print("=" * 74)
    return rc


if __name__ == "__main__":
    sys.exit(main())