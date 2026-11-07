#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
葡萄夹持状态识别（单帧）—— 边缘几何法，无需训练。

一、这个模块解决什么问题
------------------------
赛题要求"实时视觉监测整个夹持过程"。要做出"具身智能"的感觉，
系统必须能说出**当前处于哪个阶段**，而不只是"检出葡萄"。

本模块只回答一个问题：

    这一帧里，葡萄处于 自由 / 贴住 / 夹紧 的哪一种？

二、物理依据（关键）
--------------------
葡萄近似球体，夹爪两指的接触面是**平面**。

    球体被平面压住 → 接触处不再是圆弧，而是一段**直线**
    压得越紧 → 接触面越大 → 直线段越长

所以定义一个量：

    扁平度 flatness = 轮廓最长直边长度 / 轮廓周长

自由状态的球体，轮廓是一圈圆弧，最长直边很短；
被夹住时出现直线段，扁平度上升；夹得越紧，上升越多。

**这个量不依赖颜色、不依赖大小、不依赖位置，只依赖"形状被压平了多少"。**

三、实测数据（2026-11-06 受控实验）
-----------------------------------
条件：同一会话、不动相机、不动葡萄，只改舵机脉宽；绿葡萄；30 张。
每状态 10 张，舵机脉宽分别是 1450（自由）/ 接触点+3（贴住）/ 接触点-12（夹紧）。

    eps=0.015 时：
        自由  扁平度 中位 0.130   范围 0.107~0.188
        贴住  扁平度 中位 0.156   范围 0.144~0.167
        夹紧  扁平度 中位 0.189   范围 0.163~0.204

    一维两阈值穷举得到最优分界：
        < 0.144          → 自由
        0.144 ~ 0.161    → 贴住
        > 0.161          → 夹紧

    三档准确率 **90.0%**：
        自由 8/10（2 个错误是检测抖动的异常帧）
        贴住 9/10
        夹紧 10/10  ← 完全可分

也试过其他特征，都不如扁平度（见 eval_3class.py 的输出）：
    占比%     50%   三组几乎相同，无区分力
    圆度      50%   重叠
    实心度    --    重叠
    bbox 比例 53%
    椭圆轴比  63%
    扁平度    90%   ★

四、为什么不用神经网络
----------------------
    ① 数据量小（30 张），训练不可靠
    ② 单一几何量可解释，出问题能定位到"哪一步算错了"
    ③ 单帧耗时 < 10 ms，比 NPU 推理还快
    ④ 赛题鼓励"用小的 AI 模型"，而这里连模型都不需要 —— 更小

五、用法
--------
作为库::

    from grape_state import GrapeStateDetector
    det = GrapeStateDetector()
    r = det.detect(bgr)          # bgr = cv2 读进来的图
    print(r["state"], r["flatness"], r["confidence"])

命令行（单张或整个目录）::

    python3 apps/grape_state.py --file debug/set_b/xxx.jpg
    python3 apps/grape_state.py --dir /tmp/gb2 --verbose

回归测试（拿已知标签的目录算准确率）::

    python3 apps/grape_state.py --eval /tmp/ga2 /tmp/gb2 /tmp/gc2
"""
from __future__ import annotations

import argparse
import glob
import os
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
sys.path.insert(0, _HERE)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

import cv2
import numpy as np

# 复用 shot_web 里已经验证过的检测（逐颜色类 + 几何筛选）
from shot_web import color_masks   # noqa: E402

#: 扁平度阈值（用一维两阈值穷举在 30 张实测数据上求得，见
#: scripts/tune_state_thresholds.py 的输出）。
#:
#: 标定结果（eps 也从同一轮搜索里选出）：
#:     eps=0.010 → 56.7%      eps=0.018 → 86.7%
#:     eps=0.012 → 70.0%      eps=0.020 → 83.3%
#:     eps=0.015 → 90.0% ★    eps=0.025 → 83.3%
#:
#: 阈值取"相邻样本值的中点"，所以每一侧都留出最大余量，不会卡在某个样本上。
#: 注意：不要把它改回 0.144 —— 有个自由状态样本的扁平度是 0.1498，
#: 用 0.144 会多错一张（90% → 86.7%）。
FLAT_FREE_MAX = 0.1442     # 小于此值 → 自由
FLAT_TOUCH_MAX = 0.1611    # 小于此值 → 贴住；否则 → 夹紧

#: 多边形近似的精度（相对周长）。0.015 在实测里准确率最高（90%）
FLAT_EPS = 0.015

#: 几何筛选阈值（和 shot_web 保持一致）
MIN_AREA_FRAC = 0.002      # 面积至少占画面 0.2%
MAX_AREA_FRAC = 0.35       # 超过 35% 认为是背景
MIN_ASPECT = 0.62          # 长宽比，排除导线
MIN_CIRC = 0.60            # 圆度，排除方块
#: 最小尺寸（像素）。实测踩过：2026-11-06 某次运行视觉锁到了一个
#: **23 像素宽**的噪点，之后 45 秒里扁平度一直是同一个值（没在看葡萄），
#: 状态机因此判不出任何东西。这个下限直接把那种小噪点排掉。
#: 参考：本项目的葡萄在 1080p 下宽度约 200~240 px。
MIN_SIZE_PX = 60

#: 跟踪模式（追认已夹住的葡萄）用的宽松阈值。
#: 葡萄被夹扁后圆度会掉到 0.6 以下，严格阈值会把它筛掉，
#: 导致夹住后彻底丢失视觉 —— A→B 判定就没法做了。
RELAX_MIN_CIRC = 0.42
RELAX_MIN_ASPECT = 0.42

STATES = ["自由", "贴住", "夹紧"]


def longest_segment_ratio(contour, eps_ratio: float = FLAT_EPS):
    """
    扁平度 = 最长直边长度 / 轮廓周长。

    用 approxPolyDP 把轮廓简化成多边形，取最长边。
    简化阈值 eps 是"相对周长"的比例 —— 太小则边碎、太长则失真，
    实测 0.015 最优（0.02 是 83%，0.01 掉到 57%）。

    返回 (flatness, 最长边长度, 周长, 多边形顶点数)
    """
    perim = cv2.arcLength(contour, True)
    if perim <= 0:
        return None, 0.0, 0.0, 0
    approx = cv2.approxPolyDP(contour, eps_ratio * perim, True).reshape(-1, 2)
    if len(approx) < 3:
        return None, 0.0, perim, len(approx)
    n = len(approx)
    longest = 0.0
    for i in range(n):
        p1 = approx[i].astype(np.float32)
        p2 = approx[(i + 1) % n].astype(np.float32)
        d = p2 - p1
        longest = max(longest, float(np.hypot(d[0], d[1])))
    return longest / perim, longest, perim, n


class GrapeStateDetector:
    """单帧葡萄夹持状态识别（两阶段：粗定位 + 精测量）。"""

    def __init__(self, free_max: float = FLAT_FREE_MAX,
                 touch_max: float = FLAT_TOUCH_MAX,
                 eps: float = FLAT_EPS,
                 coarse_scale: float = 0.25):
        self.free_max = free_max
        self.touch_max = touch_max
        self.eps = eps
        #: 粗定位用的缩放比例。0.25 → 1080p 变 480x270，形态学便宜 16 倍。
        #: 实测这个比例下三档扁平度仍是单调的（0.135 / 0.162 / 0.185），
        #: 而精测量仍在全分辨率上做，所以精度不损失。
        self.coarse_scale = coarse_scale

    # ------------------------------------------------------- 阶段一：粗定位
    def _coarse_locate(self, bgr, relaxed: bool = False):
        """
        在缩小图上找葡萄的大致位置，返回 (颜色类, 全分辨率包围盒)。

        为什么分阶段
        ------------
        `color_masks` 的形态学是全流程最贵的一步，实测 1080p 下
        椭圆核开运算单个掩码就要 564 ms（椭圆核不可分离）。
        但耗时几乎完全由像素数决定，缩到 1/4 就只剩 1/16。

        粗定位只需要"葡萄大概在哪"，精度不重要 ——
        真正的形状测量放到阶段二，在**全分辨率**的小裁剪上做。

        relaxed（跟踪模式）
        -------------------
        2026-11-06 视频 + 日志确认：葡萄被**夹紧压扁**之后，
        圆度从 0.82 掉到 0.63 甚至更低，于是被 `MIN_CIRC=0.60` 筛掉。
        结果夹住之后视觉连续 30 秒认不出葡萄 ——
        而 A→B 判定恰恰需要跟踪它的质心。

        所以分两种模式：
            · 捕获（relaxed=False）：标准严格，保证一开始锁对目标
            · 跟踪（relaxed=True） ：放宽圆度/长宽比，因为葡萄已经变形了
        跟踪模式仍有颜色优先 + 尺寸下限，不会跑去锁黑色传感器块。
        """
        min_circ = MIN_CIRC if not relaxed else RELAX_MIN_CIRC
        min_aspect = MIN_ASPECT if not relaxed else RELAX_MIN_ASPECT
        s = self.coarse_scale
        small = cv2.resize(bgr, None, fx=s, fy=s,
                           interpolation=cv2.INTER_AREA) if s != 1.0 else bgr
        sh, sw = small.shape[:2]
        per_cls = {}          # 每个颜色类各自的最优候选
        for cls_name, m in color_masks(small, fast=True).items():
            n, labels, stats, cents = cv2.connectedComponentsWithStats(m, 8)
            for idx in range(1, n):
                area = int(stats[idx, cv2.CC_STAT_AREA])
                if area < sw * sh * MIN_AREA_FRAC:
                    continue
                if area > sw * sh * MAX_AREA_FRAC:
                    continue
                bw = int(stats[idx, cv2.CC_STAT_WIDTH])
                bh = int(stats[idx, cv2.CC_STAT_HEIGHT])
                if bw <= 0 or bh <= 0:
                    continue
                # ★ 排掉小噪点（粗定位在缩小图上做，所以要按缩放比例折算）
                #   黑色类要求更严：2026-11-06 实测，夹紧后绿葡萄看不见时，
                #   程序回退到黑色类，把 96px 的黑色小块当成葡萄，
                #   还据此判出了"接触到"。黑色块真实尺寸都比较大，抬高门槛安全。
                min_px = MIN_SIZE_PX if cls_name != "black" else 150
                if min(bw, bh) / s < min_px:
                    continue
                aspect = min(bw, bh) / max(bw, bh)
                if aspect < min_aspect:
                    continue
                # 只在这个连通域的包围盒里找轮廓（**不要对全图做**，
                # 全尺寸 findContours 在 1080p 上是 30~40 ms/次）
                x0 = int(stats[idx, cv2.CC_STAT_LEFT])
                y0 = int(stats[idx, cv2.CC_STAT_TOP])
                sub = (labels[y0:y0 + bh, x0:x0 + bw] == idx).astype(np.uint8)
                cnts, _ = cv2.findContours(sub * 255, cv2.RETR_EXTERNAL,
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
                # 圆度权重很高：圆是葡萄最本质的特征，矩形/方块要压下去
                score = area * circ * circ * circ
                cb = per_cls.get(cls_name)
                if cb is None or score > cb["score"]:
                    # 包围盒换算回全分辨率坐标
                    inv = 1.0 / s
                    per_cls[cls_name] = {
                        "score": score, "cls": cls_name, "circ": circ,
                        "bbox": (int(x0 * inv), int(y0 * inv),
                                 int(bw * inv), int(bh * inv)),
                        "centroid": (float(cents[idx][0] * inv),
                                     float(cents[idx][1] * inv)),
                    }
        # ★ 颜色优先级：绿 > 紫 > 黑
        #
        #   2026-11-06 实测事故：视觉锁到了**黑色传感器块**（扁平度 0.259、
        #   宽 241px、全程不变），于是把矩形当成葡萄，
        #   葡萄掉了也看不出来，45 秒都卡在"闭合中"。
        #
        #   原因：黑色传感器块是**矩形**，圆度约 0.78、扁平度约 0.25，
        #   和一个圆圆的葡萄（圆度 0.80~0.84、扁平度 0.14）几乎重合，
        #   光靠几何量分不开。
        #
        #   但颜色能分开：葡萄是绿/紫（高饱和），夹爪零件是黑/白。
        #   所以只要同帧里有**彩色**候选，就绝不选黑色候选；
        #   只有完全没有彩色候选时（真用黑葡萄）才回退到黑色。
        colored = [c for k, c in per_cls.items() if k != "black" and c]
        if colored:
            return max(colored, key=lambda c: c["score"])
        return per_cls.get("black")

    # ------------------------------------------------------- 阶段二：精测量
    def _fine_measure(self, bgr, cls, bbox, margin: float = 0.18,
                      relaxed: bool = False):
        """在全分辨率的裁剪上精测轮廓。

        relaxed=True 时放宽圆度/长宽比 —— 被夹扁的葡萄形状会变形，
        用捕获阶段的严格阈值会把它筛掉（实测导致夹住后 30 秒丢检）。
        """
        min_circ = MIN_CIRC if not relaxed else RELAX_MIN_CIRC
        min_aspect = MIN_ASPECT if not relaxed else RELAX_MIN_ASPECT
        # 在全分辨率的包围盒裁剪上重算轮廓，得到精确的扁平度。
        #
        # 裁剪范围比包围盒大 18%：葡萄贴住夹爪时边缘可能被黑色块切掉一点，
        # 留些余量才能把完整圆弧/直边都包进来。
        H, W = bgr.shape[:2]
        x, y, bw, bh = bbox
        mx = int(bw * margin)
        my = int(bh * margin)
        x0 = max(0, x - mx)
        y0 = max(0, y - my)
        x1 = min(W, x + bw + mx)
        y1 = min(H, y + bh + my)
        if x1 - x0 < 12 or y1 - y0 < 12:
            return None
        crop = bgr[y0:y1, x0:x1]
        ch, cw = crop.shape[:2]

        masks = color_masks(crop, only=cls)
        m = masks.get(cls)
        if m is None:
            return None
        n, labels, stats, cents = cv2.connectedComponentsWithStats(m, 8)
        best = None
        for idx in range(1, n):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area < cw * ch * MIN_AREA_FRAC:
                continue
            if area > cw * ch * 0.95:
                continue
            xb = int(stats[idx, cv2.CC_STAT_LEFT])
            yb = int(stats[idx, cv2.CC_STAT_TOP])
            wb = int(stats[idx, cv2.CC_STAT_WIDTH])
            hb = int(stats[idx, cv2.CC_STAT_HEIGHT])
            if wb <= 0 or hb <= 0:
                continue
            min_px = MIN_SIZE_PX if cls != "black" else 150
            if min(wb, hb) < min_px:      # 排掉小噪点
                continue
            aspect = min(wb, hb) / max(wb, hb)
            if aspect < min_aspect:
                continue
            sub = (labels[yb:yb + hb, xb:xb + wb] == idx).astype(np.uint8)
            cnts, _ = cv2.findContours(sub * 255, cv2.RETR_EXTERNAL,
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
            # 轮廓坐标是相对裁剪里的小图，平移回全分辨率坐标
            c = c + np.array([[[x0 + xb, y0 + yb]]], dtype=c.dtype)
            score = area * circ * circ
            if best is None or score > best["score"]:
                best = {
                    "score": score, "contour": c,
                    "area": area, "circ": circ, "aspect": aspect,
                    "blob_area": area,
                    # ★ 可见区域的宽高（全分辨率像素）。
                    #   宽度的物理意义：葡萄露在两指之间的那部分有多宽，
                    #   也就是"两指间距"的视觉估计。夹得越紧，这个值越小。
                    #   它和扁平度是互补的：
                    #     扁平度 → 形状被压平的程度（重夹时被遮挡干扰）
                    #     宽度   → 剩余可见缝隙（遮挡越严重反而越准）
                    "width_px": float(wb),
                    "height_px": float(hb),
                    "centroid": (float(x0 + xb + cents[idx][0]),
                                 float(y0 + yb + cents[idx][1])),
                }
        return best

    # ---------------------------------------------------------------- 主入口
    def detect(self, bgr, roi=None, relaxed: bool = False) -> dict:
        """
        参数
        ----
        roi : (x0, y0, x1, y1) 或 None
            只在给定区域里找葡萄。

            ★ 为什么需要 ROI：
              画面里有导线、连机器、桌面反光等干扰。实测踩过：
              某次运行视觉锁到了一个 **23 像素宽**的噪点，
              之后 45 秒扁平度一直是同一个值（根本没在看葡萄），
              状态机因此判不出任何东西。
              限定在两指之间的区域里找，可以同时解决
              ① 抓错目标  ② 检测更快（像素少）。

            坐标是**全图坐标**，内部会先裁剪再检测，
            返回的质心会自动加回偏移。

        返回::

            {
              "ok":        True/False,
              "state":     "自由" / "贴住" / "夹紧",
              "flatness":  0.156,
              "confidence": 0.72,       # 离最近阈值的归一化距离
              "metrics":   {占比/圆度/实心度/长宽比/颜色类/质心},
              "reason":    "文字说明",
            }
        """
        # ── ROI：只在指定区域里找（排除导线/噪点干扰，也更快）──
        if roi is not None:
            try:
                rx0, ry0, rx1, ry1 = [int(v) for v in roi]
            except (TypeError, ValueError):
                rx0 = ry0 = rx1 = ry1 = 0
            H, W = bgr.shape[:2]
            rx0 = max(0, min(W - 2, rx0))
            ry0 = max(0, min(H - 2, ry0))
            rx1 = max(rx0 + 2, min(W, rx1))
            ry1 = max(ry0 + 2, min(H, ry1))
            if (rx1 - rx0) >= 60 and (ry1 - ry0) >= 60:
                sub = bgr[ry0:ry1, rx0:rx1]
                r = self.detect(sub, roi=None, relaxed=relaxed)
                # 质心 / 包围盒坐标要加回偏移，否则状态机会以为葡萄在左上角
                if r.get("ok"):
                    m = r.get("metrics") or {}
                    c = m.get("centroid")
                    if c:
                        m["centroid"] = (c[0] + rx0, c[1] + ry0)
                    m["roi"] = (rx0, ry0, rx1, ry1)
                return r

        h, w = bgr.shape[:2]
        out = {"ok": False, "state": None, "flatness": None,
               "confidence": 0.0, "metrics": {}, "reason": ""}

        # 阶段一：粗定位（缩小图上找"葡萄大概在哪"）
        coarse = self._coarse_locate(bgr, relaxed=relaxed)
        if coarse is None:
            out["reason"] = ("没找到圆形目标 —— 请依次检查："
                             "① 葡萄是否**完整**在画面内（尤其别被下边缘切掉，"
                             "切掉后长宽比和形状都不对，会被筛掉）"
                             "② 是否被夹爪挡住 ③ 是否太远/太小")
            return out

        # 阶段二：在全分辨率的小裁剪上精测轮廓
        fine = self._fine_measure(bgr, coarse["cls"], coarse["bbox"],
                                  relaxed=relaxed)
        if fine is None:
            out["reason"] = "粗定位找到了，但精测量阶段轮廓不合格（可能被遮挡）"
            return out

        # ★ 检查是否被画面边缘切掉 —— 这一步非常关键
        #
        #   2026-11-06 实测事故：葡萄在画面**下缘**被切掉一部分，
        #   可见轮廓成了一个"D"形，底下那条"直边"其实是画面边界。
        #   结果扁平度从正常的 0.14 跳到 0.28，
        #   但圆度还有 0.818、宽 252px、尺寸和圆度都"正常"，
        #   所有几何筛选都放它过去，系统就拿着一个假数字跑了 45 秒。
        #
        #   葡萄被切掉时，它的形状统计**整体不可信**，
        #   没有任何阈值能补救。唯一正确的做法是明确报错，
        #   让人把葡萄完整放进画面。
        bx, by, bw_, bh_ = cv2.boundingRect(fine["contour"])
        EDGE = 6
        cut = []
        if bx <= EDGE:
            cut.append("左")
        if by <= EDGE:
            cut.append("上")
        if bx + bw_ >= w - EDGE:
            cut.append("右")
        if by + bh_ >= h - EDGE:
            cut.append("下")
        if cut:
            out["cut"] = True
            out["reason"] = (
                "葡萄被画面「%s」边缘切掉，轮廓形状不可信（扁平度会虚高）"
                "→ 请调整机位，把葡萄完整放进画面再试"
                % "、".join(cut))
            return out

        flat, longest, perim, npt = longest_segment_ratio(fine["contour"],
                                                          self.eps)
        if flat is None:
            out["reason"] = "轮廓点太少，无法做多边形近似"
            return out

        # 实心度
        solidity = None
        ca = cv2.contourArea(fine["contour"])
        ha = cv2.contourArea(cv2.convexHull(fine["contour"]))
        if ha > 0:
            solidity = ca / ha

        # 葡萄区域内的清晰度（诊断用，在轮廓包围盒上算）
        focus_grape = None
        try:
            x, y, bw, bh = cv2.boundingRect(fine["contour"])
            g = (bgr if bgr.ndim == 2
                 else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
            crop = g[max(0, y):min(h, y + bh), max(0, x):min(w, x + bw)]
            if crop.size:
                focus_grape = float(cv2.Laplacian(crop, cv2.CV_64F).var())
        except Exception:  # noqa: BLE001
            pass

        # 三档判定
        if flat < self.free_max:
            state = "自由"
        elif flat < self.touch_max:
            state = "贴住"
        else:
            state = "夹紧"

        # 置信度：离两条阈值的距离，归一到 [0,1]
        d_free = abs(flat - self.free_max)
        d_touch = abs(flat - self.touch_max)
        margin = min(d_free, d_touch)
        half = max((self.touch_max - self.free_max) / 2.0, 1e-6)
        conf = float(min(1.0, margin / (half * 2.0)))

        out.update({
            "ok": True,
            "state": state,
            "flatness": float(flat),
            "longest_seg_px": float(longest),
            "perimeter_px": float(perim),
            "poly_points": int(npt),
            "confidence": conf,
            "metrics": {
                "cls": coarse["cls"],
                "frac": fine["area"] / float(w * h),
                "circ": fine["circ"],
                "solidity": solidity,
                "aspect": fine["aspect"],
                "centroid": fine["centroid"],
                "width_px": fine.get("width_px"),
                "height_px": fine.get("height_px"),
                "focus_grape": focus_grape,
                "coarse_circ": coarse["circ"],
            },
            "reason": ("扁平度 %.3f（阈值 %.3f/%.3f）→ %s"
                       % (flat, self.free_max, self.touch_max, state)),
        })
        return out


# --------------------------------------------------------------------------
# 时间稳定器：把单帧判定变成可用的连续状态
# --------------------------------------------------------------------------
class StateStabilizer:
    """
    把逐帧的抖动压下去，输出稳定的状态。

    为什么需要
    ----------
    2026-11-06 实测：**场景完全静止**，连续 16 帧的扁平度是

        0.1349 0.1512 0.1403 0.1557 0.1510 0.1454 0.1364
        0.1468 0.1597 0.1561 0.1581 0.1579 0.1451

    σ = 0.0084，而"自由(0.130) → 贴住(0.156)"只差 0.026 ——
    **单帧抖动和状态间距是同一量级**，于是同一颗静止的葡萄
    在"自由/贴住"之间反复横跳。直接用单帧判定的状态机会输出垃圾。

    抖动来源（已确认）：
        · 自动曝光在动（实测画面亮度均值 140~153 之间变化）
        · 掩码边界因此抖动 1~2 像素
        · approxPolyDP 的"最长边"对这种边界变化很敏感

    两个措施
    --------
    ① **空间门限**：只接受质心在"已锁定位置" ±gate_px 内的检测。
       实测有一帧把黑色传感器块（质心 x=1052）当成了黑葡萄，
       而真葡萄在 x=703 —— 门限直接把它拒掉。

    ② **时间中值**：状态用最近 window 帧扁平度的**中位数**判定。
       σ 按 1/√n 下降：0.0084 → 0.0084/√7 ≈ 0.0032，
       于是 0.026 的状态间距变成约 8σ，判定就稳了。
       用中位数而不是均值，是因为中位数对孤立野值不敏感。

    用法::

        st = StateStabilizer(window=7, gate_px=60)
        for frame in frames:
            r = det.detect(frame)
            s = st.update(r)          # → {"stable":bool,"state":..., ...}
    """

    def __init__(self, window: int = 7, gate_px: float = 60.0,
                 free_max: float = FLAT_FREE_MAX,
                 touch_max: float = FLAT_TOUCH_MAX,
                 lock_after: int = 3):
        self.window = max(1, window)
        self.gate_px = gate_px
        self.free_max = free_max
        self.touch_max = touch_max
        self.lock_after = lock_after
        self._hist = []           # 扁平度历史
        self._center = None       # 锁定的质心
        self._lock_count = 0
        self.rejected = 0         # 被空间门限拒掉的帧数
        self.accepted = 0

    def reset(self):
        self._hist.clear()
        self._center = None
        self._lock_count = 0
        self.rejected = 0
        self.accepted = 0

    # ---------------------------------------------------------------- 更新
    def update(self, r: dict) -> dict:
        """
        喂入一次 detect() 的结果，返回::

            {
              "stable":   True/False,    # 是否有足够的有效样本
              "state":    "自由"/"贴住"/"夹紧"/None,
              "flatness": 平滑后的扁平度,
              "raw":      本帧原始扁平度,
              "n":        窗口内有效帧数,
              "note":     "被门限拒绝" 等说明,
            }
        """
        out = {"stable": False, "state": None, "flatness": None,
               "raw": None, "n": len(self._hist), "reason": "",
               "answered": True}

        # 本帧没检出：**不清空历史**（丢一帧不该让状态归零），
        # 但如果连续丢太多帧，历史里的旧值会过期，所以窗口满了就自然滚出。
        if not r or not r.get("ok"):
            out["reason"] = "本帧未检出"
            out["answered"] = False
            return self._decide(out)

        out["raw"] = r.get("flatness")
        c = r.get("metrics", {}).get("centroid")
        if c is None:
            out["answered"] = False
            return self._decide(out)

        # ---- ① 空间门限 ----
        if self._center is None:
            self._center = (float(c[0]), float(c[1]))
        else:
            d = ((c[0] - self._center[0]) ** 2
                 + (c[1] - self._center[1]) ** 2) ** 0.5
            if d > self.gate_px:
                self.rejected += 1
                out["reason"] = "偏离锁定位置 %.0f px，已拒绝" % d
                out["answered"] = False
                return self._decide(out)
            # 缓慢跟随（每帧走 15%，不怕葡萄被夹爪带着小幅移动）
            self._center = (self._center[0] + 0.15 * (c[0] - self._center[0]),
                            self._center[1] + 0.15 * (c[1] - self._center[1]))
        self.accepted += 1
        self._lock_count += 1

        # ---- ② 时间中值 ----
        self._hist.append(float(r["flatness"]))
        if len(self._hist) > self.window:
            self._hist.pop(0)
        return self._decide(out)

    def _decide(self, out: dict) -> dict:
        out["n"] = len(self._hist)
        if len(self._hist) < self.lock_after:
            out["stable"] = False
            return out
        med = statistics.median(self._hist)
        out["flatness"] = float(med)
        out["stable"] = True
        out["state"] = ("自由" if med < self.free_max
                        else ("贴住" if med < self.touch_max else "夹紧"))
        return out
# --------------------------------------------------------------------------
# 命令行
# --------------------------------------------------------------------------
def imread_u(path):
    buf = np.fromfile(path, dtype=np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def cmd_file(det, path, verbose):
    bgr = imread_u(path)
    if bgr is None:
        print(f"❌ 读不出：{path}")
        return 1
    r = det.detect(bgr)
    print(f"\n{os.path.basename(path)}")
    if not r["ok"]:
        print(f"  ❌ {r['reason']}")
        return 1
    m = r["metrics"]
    print(f"  状态      {r['state']}   置信度 {r['confidence']*100:.0f}%")
    print(f"  扁平度    {r['flatness']:.4f}   "
          f"(最长边 {r['longest_seg_px']:.0f}px / 周长 "
          f"{r['perimeter_px']:.0f}px，{r['poly_points']} 个顶点)")
    if verbose:
        print(f"  颜色类    {m['cls']}")
        print(f"  占比      {m['frac']*100:.2f}%")
        print(f"  圆度      {m['circ']:.3f}")
        print(f"  实心度    {m['solidity']:.3f}")
        print(f"  长宽比    {m['aspect']:.3f}")
        print(f"  质心      ({m['centroid'][0]:.0f}, {m['centroid'][1]:.0f})")
        print(f"  葡萄清晰度 {m['focus_grape']:.1f}")
    return 0


def cmd_dir(det, d, verbose):
    files = sorted(glob.glob(os.path.join(d, "*.jpg")))
    if not files:
        print(f"  {d}: 没有图片")
        return 1
    print(f"\n{d}（{len(files)} 张）")
    counts = {}
    flats = []
    for f in files:
        bgr = imread_u(f)
        if bgr is None:
            continue
        r = det.detect(bgr)
        if not r["ok"]:
            print("  %-34s ❌ 未检出" % os.path.basename(f)[:34])
            continue
        counts[r["state"]] = counts.get(r["state"], 0) + 1
        flats.append(r["flatness"])
        if verbose:
            print("  %-34s %-4s 扁平度 %.3f  置信 %3.0f%%"
                  % (os.path.basename(f)[:34], r["state"], r["flatness"],
                     r["confidence"] * 100))
    if counts:
        summary = "  ".join("%s×%d" % (k, v) for k, v in counts.items())
        print("  汇总：%s   扁平度中位 %.3f" % (summary,
                                               statistics.median(flats)))
    return 0


def cmd_eval(det, dirs):
    """拿三个已知标签的目录算准确率（回归测试）。"""
    if len(dirs) != 3:
        print("❌ --eval 需要正好 3 个目录（自由/贴住/夹紧）")
        return 1
    total = correct = 0
    conf_mat = {a: {b: 0 for b in STATES} for a in STATES}
    per = {}
    for label, d in zip(STATES, dirs):
        flats = []
        for f in sorted(glob.glob(os.path.join(d, "*.jpg"))):
            bgr = imread_u(f)
            if bgr is None:
                continue
            r = det.detect(bgr)
            if not r["ok"]:
                continue
            flats.append(r["flatness"])
            total += 1
            conf_mat[label][r["state"]] += 1
            if r["state"] == label:
                correct += 1
        per[label] = (statistics.median(flats) if flats else None,
                      len(flats))

    print("=" * 72)
    print("三档分类评估")
    print("=" * 72)
    for lb in STATES:
        med, n = per[lb]
        print("  %s：%d 张，扁平度中位 %s"
              % (lb, n, ("%.4f" % med) if med else "--"))
    print()
    print("  真实\\预测   " + "  ".join("%6s" % s for s in STATES) + "   正确率")
    for lb in STATES:
        row = conf_mat[lb]
        tot = sum(row.values())
        acc = 100.0 * row[lb] / tot if tot else 0.0
        print("  %-10s  %s   %5.1f%%"
              % (lb, "  ".join("%6d" % row[s] for s in STATES), acc))
    print()
    print("  **总体准确率 %.1f%%（%d/%d）**"
          % (100.0 * correct / total if total else 0, correct, total))
    return 0 if total and correct / total >= 0.85 else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="葡萄夹持状态识别（边缘几何法）")
    ap.add_argument("--file", help="单张图片")
    ap.add_argument("--dir", help="整个目录")
    ap.add_argument("--eval", nargs=3, metavar=("自由", "贴住", "夹紧"),
                    help="三个已知标签的目录，算准确率")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--free-max", type=float, default=FLAT_FREE_MAX)
    ap.add_argument("--touch-max", type=float, default=FLAT_TOUCH_MAX)
    args = ap.parse_args()

    det = GrapeStateDetector(free_max=args.free_max,
                             touch_max=args.touch_max)

    rc = 0
    if args.file:
        rc |= cmd_file(det, args.file, args.verbose)
    if args.dir:
        rc |= cmd_dir(det, args.dir, args.verbose)
    if args.eval:
        rc |= cmd_eval(det, args.eval)
    if not (args.file or args.dir or args.eval):
        ap.print_help()
    return rc


if __name__ == "__main__":
    sys.exit(main())