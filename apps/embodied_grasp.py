#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
具身抓取闭环 —— 视觉持续观测 + 力传感 + 双模判定。

与旧流程的区别（这是重点）
--------------------------
旧的 ``grasp_with_verify.py`` 是**单帧快照**：

    拍照一次 → 算出葡萄位置 → 闭合力控 → 手势
              ↑
        整个抓取过程只有这一个瞬间用视觉

本模块是**连续观测 + 时序判断**：

    视觉线程：约 4 Hz 持续判"自由/贴住/夹紧"（单帧 133 ms）
    力线程  ：约 20 Hz 采样，取中位数
    融合    ：每 100 ms 汇总一次，驱动状态机
                       ↓
              输出一条状态序列

三、双模判定（这是导师要求的核心）
----------------------------------
**"夹到了"必须同时满足力上升和视觉看到接触：**

    接触  = 力 > CONTACT_N          且  视觉 ∈ {贴住, 夹紧}
    夹住  = 力 > GRASP_N            且  视觉 = 夹紧  （连续 3 次确认）
    夹紧  = 力 ≥ 目标力              且  视觉 = 夹紧
    滑移  = 已夹住 但 扁平度持续下降或质心相对位移增大
    脱落  = 已夹住 但 力掉到 < RELEASE_N，或视觉连续判"自由"

**这样做的实际价值**：旧流程里如果夹爪闭到目标力却夹了个空
（葡萄被推走了），力控会认为"夹住了"，要等提起后测力才发现。
现在视觉会立刻指出"没夹到"，**当场就能重试**。

四、A→B 判定
------------
夹住后记录葡萄质心为 A。人手把夹爪移到 B 的过程中持续观测：
质心位移超过 ``A2B_PX`` 且力仍保持在 ``GRASP_N`` 以上 → 判定"已移到 B"。

用法::

    sudo python3 apps/embodied_grasp.py --object grape

    # 不驱动舵机，只跑视觉+力的观测循环（安全调试）
    sudo python3 apps/embodied_grasp.py --dry-run

    # 打印每一拍的明细
    sudo python3 apps/embodied_grasp.py -v
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import threading
import time

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

from grape_gripper.config import Config, default_config_path   # noqa: E402
from grape_state import GrapeStateDetector, StateStabilizer    # noqa: E402

try:
    import cv2
    _CV_OK = True
except ImportError:          # pragma: no cover
    _CV_OK = False

# ---------------------------------------------------------------- 判定参数
#
# ★ 分工（按导师原话："传感器和视觉边缘检测相结合"）
#
#     判断"夹到了" = 传感器：力上升
#                    AND
#                    视觉边缘：扁平度比基线上升（轮廓被平板压出直边）
#                    → **两者缺一不判**
#
#   没有主辅之分，是**结合**。视觉负责"边缘变成什么样了"，
#   力负责"确实受力了"。
#
#   ⚠️ 刻意不判"夹得多紧"：
#      ① 原话没要求
#      ② 实测力 >1.0N 后葡萄被夹爪遮挡，扁平度反而下降
#         （自由 0.138 → 接触 0.159 → 重夹 0.150），测不准
#      ③ 而"夹到了"发生在**接触瞬间**，那时葡萄还完整可见，信号最清晰
CONTACT_N = 0.05        # 传感器：判定"有力了"的阈值（N）
FLAT_RISE = 0.015       # 视觉边缘：扁平度相对基线至少要上升这么多
                        # （实测未接触 0.138 → 接触后 0.159，涨 0.021）
HOLD_N = 0.50           # "已夹住"：力保持在这个值以上（不判"多紧"）
                        # ★ 2026-11-06 从 0.30 提到 0.50：
                        #   实测在 0.86N 就判"已夹住"、然后停止闭合，
                        #   力再也没到目标 1.0N，导致"夹不牢、提不起来"。
                        #   抬高这个门槛可以更早暴露"其实没夹紧"。
GRASP_N = 0.15          # 判滑移/A→B 时的力下限
RELEASE_N = 0.03        # 低于此力且曾夹住 → 判"脱落"
BASELINE_SECONDS = 2.0  # 张开状态下采扁平度基线的时长
BASELINE_MIN_SAMPLES = 4   # 至少攒够几个有效样本才算基线
BASELINE_FAIL_TICKS = 60   # 60 拍（约 6 秒）还采不到 → 判失败并报原因
AIR_CONFIRM = 4         # 力到目标但扁平度一直没上升 → 判"夹空"
DROP_CONFIRM = 5        # 已夹住后视觉连续几拍丢失 → 判"脱落"
SOLID_OK = 0.85         # 实心度低于此值 → 轮廓被切碎，视觉不可信
SLIP_WIDTH_PX = 25.0    # 可见宽度比夹住时变大超过此值 → 判"滑移"
#: A→B 判定：质心位移超过此像素数。
#: ★ 2026-11-06 修正：原来是 60px，换算下来只有约 6 mm
#:   （葡萄直径 22mm → 画面 220px → 10px/mm），
#:   手稍微一动就判"已移到B"，误判率很高。
#:   200px ≈ 2cm，既能避开手抖，又远小于实际搬运距离（10cm ≈ 1000px）。
A2B_PX = 200.0

# ---- B 点：预学目标位置（2026-11-06 新增）----
#
# 为什么需要「预学 B 点」
# ---------------------
# 原来的 A→B 判据只是"质心移动 ≥200px"，也就是"**动得够远**"，
# 而不是"**到了 B**"。用户实测反馈："我一移动它就判定为 A 到 B，
# 那我怎么确定 B 的位置呢？"
#
# 确实没有 B 的概念。补法：让用户**先教一次 B 在哪**。
#
#     sudo python3 apps/embodied_grasp.py --learn-b
#     把葡萄放到希望它最终到达的位置，按回车
#     → 记住那一刻的质心（存成占画面的比例，换分辨率也不失效）
#
# 之后每次运行：
#
#     已移到B = 质心距 B 点 ≤ 容差
#            AND 连续几拍几乎不动（**真的停稳了**，不是路过）
#            AND 力仍在线、扁平度仍高于基线（还夹着）
#
# 没学过 B 点就自动退回"位移 ≥ A2B_PX"，不报错。
B_TARGET_FILE = "b_target.json"
B_TOL_PX = 50.0         # 到点容差（葡萄直径约 250px，50px 约半个半径）
STILL_PX = 8.0          # 相邻两拍质心移动小于此值 → 视为"没在动"
STILL_CONFIRM = 3       # 连续几拍不动才算"停稳"
                        # ★ 2026-11-06 从 5 降到 3：3 fps 下等 5 拍要 1.7 秒，
                        #   加上 A→B 的 3 次确认，用户在 B 点要停近 3 秒，
                        #   演示时太难做到。降到 3 拍后约 1~1.5 秒就能判出。


def b_target_path(path: str | None = None) -> str:
    """B 点文件的位置（默认放在项目根目录）。"""
    if path:
        return path if os.path.isabs(path) else os.path.join(_ROOT, path)
    return os.path.join(_ROOT, B_TARGET_FILE)


def load_b_target(path: str | None = None) -> dict | None:
    """
    读 B 点。返回 {"x_norm","y_norm","tol_px","frame"} 或 None。

    ★ 存"占画面的比例"而不是绝对像素：
      换分辨率（1920→1280）时绝对坐标会全错，比例不会。
    """
    import json
    p = b_target_path(path)
    try:
        with open(p, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        if "x_norm" not in d or "y_norm" not in d:
            return None
        return d
    except (OSError, ValueError):
        return None


def save_b_target(x_norm: float, y_norm: float, frame, tol: float,
                  path: str | None = None) -> str:
    """写 B 点。"""
    import json
    p = b_target_path(path)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump({"x_norm": round(float(x_norm), 5),
                   "y_norm": round(float(y_norm), 5),
                   "tol_px": float(tol),
                   "frame_w": int(frame[0]),
                   "frame_h": int(frame[1])}, fh,
                  ensure_ascii=False, indent=2)
    return p

# ---- 2026-11-06 第二次实抓暴露的问题，对应三个补丁 ----
#
# ① 力冲过头：实测目标 1.2N，实际冲到 2.26N。
#    原因：闭合循环在力于目标附近抖动时"只进不退"（棘轮）。
#    补丁：到目标就**锁死**不再走；超过 CEIL_N 反而退一步。
CEIL_MARGIN = 0.30      # 力上限 = 目标力 + 该余量；超过说明两指压在一起
#   ⚠️ 2026-11-06 二次修订：这个量**只用于闭合控制**（防止把葡萄压坏），
#      **不再用于判定**。原因：PS01 单样本 σ≈2.6N 且有周期性尖峰、
#      舵机本身也抖，用力大小当判定阈值会被噪声顶穿。
#
# ---- 所有判定统一改成「连续 N 拍确认」，避免单拍噪声误触发 ----
CONTACT_CONFIRM = 2     # 接触到：连续满足
GRASP_CONFIRM = 3       # 夹到了：连续满足
DROP_FORCE_CONFIRM = 3  # 力骤降：连续 3 拍
DROP_SQUEEZE_CONFIRM = 4  # 扁平度回落：连续 4 拍（视觉主判据）
SLIP_CONFIRM = 3        # 滑移：连续 3 拍
MOVED_CONFIRM = 3       # A→B：连续 3 拍
#
# ② 葡萄掉了却判"已移到B"：葡萄掉落也会让质心移动，于是误判成功。
#    补丁：A→B 必须**同时**还有"被挤压"的视觉证据（扁平度仍高于基线）。
#    葡萄掉落后扁平度会回落到基线附近 —— 这正好是判据。
SQUEEZE_KEEP = 0.55     # A→B 时要求 扁平度涨幅 ≥ FLAT_RISE × 该系数
#
# ③ 明明夹住了却一直停在"闭合中"：
#    扁平度在接触后被夹爪遮挡会回落，只比较"当前值"就永远不触发。
#    补丁：记录**最大涨幅**（曾经涨到过就算数，视觉证据是历史事实）。

# ---------------------------------------------------------------- 状态名
S_WAIT = "采基线"        # 张开、采扁平度基线（此时不闭合）
S_CLOSING = "闭合中"
S_TOUCHED = "接触到"     # 力上升 + 边缘变直（传感器与视觉一致）
S_HOLDING = "夹到了"     # ★ 力达到门槛且视觉证据成立 —— 这就是导师说的"夹到了"
S_SLIPPING = "滑移中"
S_MOVED = "已移到B"
S_DROPPED = "脱落"
S_AIR = "夹空"
S_FAIL = "失败"


class VisionWorker:
    """
    后台线程：持续从摄像头取帧并判状态。

    ⚠️ 为什么单独开线程：一次检测约 133 ms，如果放在主循环里，
       力采样就会被卡住（力要 20 Hz）。分开之后互不影响。

    ⚠️ 为什么限速：16 fps 连续推流会把板子跑到 79°C（2026-11-05 事故）。
       这里每帧之间至少休息 ``min_interval`` 秒，实测约 3~4 Hz，
       约占半个 CPU 核；而且只在抓取期间运行。
    """

    def __init__(self, detector, min_interval: float = 0.06,
                 width: int = 1920, height: int = 1080,
                 stab_window: int = 7, stab_gate_px: float = 60.0):
        self.det = detector
        self.min_interval = min_interval
        self.width = width
        self.height = height
        # ★ 时间稳定器：单帧扁平度抖动 σ≈0.0084，和状态间距(0.026)同量级，
        #   直接用单帧会反复横跳。这里做空间门限 + 时间中值。
        #   详见 grape_state.StateStabilizer 的说明。
        self.stab = StateStabilizer(window=stab_window, gate_px=stab_gate_px)
        #: 锁定的感兴趣区域 (x0,y0,x1,y1)；首次检出后自动设定。
        #: 用处：① 防止锁到画面别处的噪点（实测锁到过 23px 的噪点）
        #:      ② 检测区域变小，速度快
        self.roi = None
        self.roi_n = 0
        #: 连续检不到的帧数；到 3 就清空 ROI 退回全画面（防止葡萄移出框后死锁）
        self.roi_miss = 0
        #: 是否已经"捕获"到葡萄。捕获前用严格阈值（保证锁对目标），
        #: 捕获后切到宽松阈值（葡萄被夹扁也能跟住，A→B 才判得出来）。
        self.acquired = False
        #: 最新预览帧（JPEG 字节）与叠加文字（由主循环更新）
        self.latest_jpg = None
        self.overlay_force = ""
        self.overlay_state = ""
        self.overlay_note = ""
        self.overlay_pulse = 0.0
        self.cam = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest = None          # 最近一次结果
        self._fps_t = time.time()
        self._fps_n = 0
        self.fps = 0.0
        self.frames = 0
        self.errors = 0
        self.thread = None

    def start(self):
        from k1_camera_stream import K1CameraStream
        print("  启动视觉：打开连续视频流…", flush=True)
        self.cam = K1CameraStream(width=self.width, height=self.height,
                                  use_sudo=False, verbose=False)
        self.cam.open()
        self._stop.clear()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def _loop(self):
        while not self._stop.is_set():
            t0 = time.time()
            try:
                bgr = self.cam.read_frame(timeout=4.0)
            except Exception as exc:  # noqa: BLE001
                self.errors += 1
                if self.errors <= 3:
                    print(f"  ⚠️ 取帧失败：{exc}", flush=True)
                time.sleep(0.5)
                continue
            if bgr is None:
                self.errors += 1
                time.sleep(0.2)
                continue
            try:
                # ★ ROI 只在"连续失败"之前用；一旦失败会自动退回全画面。
                r = self.det.detect(bgr, roi=self.roi, relaxed=self.acquired)
            except Exception as exc:  # noqa: BLE001
                self.errors += 1
                if self.errors <= 3:
                    print(f"  ⚠️ 检测失败：{exc}", flush=True)
                r = {"ok": False, "state": None, "reason": str(exc)}

            # ★ 维护 ROI（2026-11-06 大幅修正）
            #
            #   旧逻辑有两个致命问题，用户实测踩到：
            #     ① ROI 只在"每 10 次成功检出"才更新一次。
            #        用户把葡萄横移时，葡萄走出 ROI → 检出全失败 →
            #        roi_n 不再增长 → **ROI 永远更新不了 → 死锁**，
            #        "已移到B"再也触发不了（表现为"横移没反应"）。
            #     ② 一旦出框，后续帧都判"葡萄消失"，
            #        而脱落检测又会把它判成"脱落"。
            #
            #   现在改成：
            #     · 每次成功检出都**平滑跟随**（0.35 系数），葡萄动了 ROI 就跟着动
            #     · 连续 3 帧检不到 → **清空 ROI，退回全画面搜索**重新锁定
            if r.get("ok"):
                self.acquired = True      # 捕获成功后切到宽松跟踪模式
                m = r.get("metrics") or {}
                c = m.get("centroid")
                wid = m.get("width_px")
                if c and wid:
                    half = max(140.0, float(wid) * 1.8)
                    H, W = bgr.shape[:2]
                    if self.roi is None:
                        self.roi = (int(max(0, c[0] - half)),
                                    int(max(0, c[1] - half)),
                                    int(min(W, c[0] + half)),
                                    int(min(H, c[1] + half)))
                    else:
                        # 平滑跟随：ROI 中心朝当前质心移动
                        rx0, ry0, rx1, ry1 = self.roi
                        rcx = (rx0 + rx1) / 2.0
                        rcy = (ry0 + ry1) / 2.0
                        rcx += 0.35 * (c[0] - rcx)
                        rcy += 0.35 * (c[1] - rcy)
                        self.roi = (int(max(0, rcx - half)),
                                    int(max(0, rcy - half)),
                                    int(min(W, rcx + half)),
                                    int(min(H, rcy + half)))
                    self.roi_miss = 0
            else:
                self.roi_miss += 1
                if self.roi is not None and self.roi_miss >= 3:
                    # 连续检不到 → 可能葡萄移动出了 ROI，退回全画面再找
                    self.roi = None
                    self.stab.reset()      # 允许重新锁定
                    self.roi_miss = 0

            # ★ 过一遍时间稳定器：拿到平滑后的扁平度和稳定状态
            s = self.stab.update(r)
            r["raw_state"] = r.get("state")
            r["raw_flatness"] = r.get("flatness")
            r["stable"] = s["stable"]
            r["stab_n"] = s["n"]
            if s["state"] is not None:
                r["state"] = s["state"]        # 用平滑后的状态对外输出
                r["flatness"] = s["flatness"]
                r["ok"] = True
            else:
                # 有效样本还不够（刚启动），保持"暂时不可信"
                #
                # ★ 保留 detect() 给出的**真实原因**。
                #   以前这里无条件覆盖成"稳定器预热中"，把
                #   "葡萄被画面下缘切掉"这种关键信息盖住了，
                #   用户只看到一直卡在采基线，无法判断该改什么。
                r["ok"] = False
                det_reason = r.get("reason")
                if det_reason:
                    r["reason"] = ("%s（稳定器 %d/%d）"
                                   % (det_reason, s["n"], self.stab.lock_after))
                else:
                    r["reason"] = ("稳定器预热中（%d/%d）"
                                   % (s["n"], self.stab.lock_after))
            r["t"] = time.time()
            with self._lock:
                self._latest = r
            # ★ 生成预览帧（给网页实时看整个夹取过程用）
            try:
                self._make_preview(bgr, r)
            except Exception:  # noqa: BLE001
                pass
            self.frames += 1
            self._fps_n += 1
            if time.time() - self._fps_t >= 2.0:
                self.fps = self._fps_n / (time.time() - self._fps_t)
                self._fps_t = time.time()
                self._fps_n = 0
            dt = time.time() - t0
            if dt < self.min_interval:
                time.sleep(self.min_interval - dt)
            else:
                time.sleep(0.02)     # 给系统一点喘息，别把核占满

    def latest(self):
        with self._lock:
            return dict(self._latest) if self._latest else None

    # ------------------------------------------------------------ 预览画面
    def _make_preview(self, bgr, r):
        """
        把当前的检测结果画到缩小图上，编码成 JPEG，供网页实时查看。

        ★ 为什么在视觉线程里做：这一帧本来就要处理，
          顺手画个框、编码一次（960x540 约 10ms），
          就能让用户**实时看到整个夹取过程**，
          而且不需要另外打开摄像头（摄像头是独占的）。
          3 fps × 10ms = 3% CPU，可以忽略。
        """
        if not _CV_OK:
            return
        h, w = bgr.shape[:2]
        sx = 960.0 / w
        vis = cv2.resize(bgr, (960, int(h * sx)),
                         interpolation=cv2.INTER_AREA)
        m = r.get("metrics") or {}
        c = m.get("centroid")
        wid = m.get("width_px")
        ok = r.get("ok")
        # 检测到的葡萄：画一个圆
        if ok and c and wid:
            cx = int(c[0] * sx)
            cy = int(c[1] * sx)
            rad = int(float(wid) * sx / 2.0)
            cv2.circle(vis, (cx, cy), rad, (0, 255, 0), 2)
            cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1)
        # ROI 框
        if self.roi:
            rx0, ry0, rx1, ry1 = self.roi
            cv2.rectangle(vis, (int(rx0 * sx), int(ry0 * sx)),
                          (int(rx1 * sx), int(ry1 * sx)), (255, 200, 0), 1)
        # 文字：状态 / 力 / 扁平度 / 说明
        #
        # ⚠️ 必须用 ASCII：OpenCV 的 putText 用的是 Hershey 字体，
        #    **不支持中文**，直接写中文会全部变成 "?????"（实测）。
        #    所以这里把状态映射成英文缩写再画。
        _EN = {
            "采基线": "BASELINE", "闭合中": "CLOSING", "接触到": "CONTACT",
            "夹到了": "GRASPED", "滑移中": "SLIPPING", "已移到B": "MOVED",
            "脱落": "DROPPED", "夹空": "AIR", "失败": "FAIL",
            "自由": "FREE", "贴住": "TOUCH", "夹紧": "TIGHT",
        }
        vstate = r.get("state")
        vstate_en = (_EN.get(vstate, "?") if vstate
                     else ("NO-GRAPE" if not ok else "--"))
        gstate_en = _EN.get(self.overlay_state, "?")
        flat_s = ("%.3f" % r["flatness"]) if r.get("flatness") else "--"
        wid_s = ("%.0fpx" % wid) if wid else "--"
        line1 = "VISION %-9s   STATE %s" % (vstate_en, gstate_en)
        line2 = "FORCE %-9s  PULSE %4.0fus" % (self.overlay_force or "--",
                                               self.overlay_pulse or 0)
        line3 = "FLAT %-6s  WIDTH %-7s  ROI %s" % (
            flat_s, wid_s, "ON" if self.roi else "OFF")
        for i, (txt, color) in enumerate((
                (line1, (255, 255, 255)),
                (line2, (180, 255, 180)),
                (line3, (120, 220, 255)))):
            if not txt:
                continue
            y = 26 + i * 26
            cv2.putText(vis, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, color, 1, cv2.LINE_AA)
        okj, buf = cv2.imencode(".jpg", vis,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if okj:
            with self._lock:
                self.latest_jpg = buf.tobytes()

    def stop(self):
        self._stop.set()
        if self.thread is not None:
            self.thread.join(timeout=3.0)
        try:
            if self.cam is not None:
                self.cam.close()
        except Exception:  # noqa: BLE001
            pass


class PreviewServer:
    """
    把视觉线程的最新一帧用网页实时播出来。

    为什么这么做
    ------------
    摄像头是**独占**的：`cam-test` 同一时间只能有一个进程用。
    所以不能"另开一个预览程序" —— 那会和抓取主程序抢摄像头。

    但视觉线程本来就在取帧、本来就在检测，
    **顺手把这一帧画上框、编码成 JPEG** 就能实时看到整个夹取过程，
    额外开销约 3% CPU（960x540，3 fps）。

    这样既能看到画面，又不用额外操作摄像头，也不会像
    2026-11-05 那次（16 fps 连续推流）那样把板子跑到 79°C。

    打开 http://<板子IP>:8082/ 即可。
    """

    def __init__(self, worker: "VisionWorker", port: int = 8082):
        self.worker = worker
        self.port = port
        self.httpd = None

    def start(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        worker = self.worker

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    body = (b"<!DOCTYPE html><html><head><meta charset='utf-8'>"
                            b"<title>K1 \xe5\xae\x9e\xe6\x97\xb6"
                            b"\xe9\xa2\x84\xe8\xa7\x88</title></head>"
                            b"<body style='margin:0;background:#0d0d0d'>"
                            b"<img src='/stream' style='width:100%;"
                            b"display:block'>"
                            b"</body></html>")
                    self.send_response(200)
                    self.send_header("Content-Type",
                                     "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type",
                                     "multipart/x-mixed-replace; "
                                     "boundary=frame")
                    self.end_headers()
                    last = None
                    try:
                        while True:
                            jpg = worker.latest_jpg
                            if jpg is not None and jpg is not last:
                                last = jpg
                                self.wfile.write(b"--frame\r\n")
                                self.wfile.write(
                                    b"Content-Type: image/jpeg\r\n")
                                self.wfile.write(
                                    ("Content-Length: %d\r\n\r\n"
                                     % len(jpg)).encode())
                                self.wfile.write(jpg)
                                self.wfile.write(b"\r\n")
                            time.sleep(0.1)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    return
                self.send_error(404)

        self.httpd = ThreadingHTTPServer(("0.0.0.0", self.port), Handler)
        threading.Thread(target=self.httpd.serve_forever,
                         daemon=True).start()
        return self

    def stop(self):
        try:
            if self.httpd is not None:
                self.httpd.shutdown()
        except Exception:  # noqa: BLE001
            pass


class EmbodiedGrasp:
    """状态机：把视觉状态和力读数融合成一条状态序列。"""

    def __init__(self, vision: VisionWorker, dry_run: bool = False,
                 verbose: bool = False, target_force: float = 1.0,
                 b_target: dict | None = None,
                 frame_size: tuple | None = None):
        self.vision = vision
        self.dry_run = dry_run
        self.verbose = verbose
        self.target_force = float(target_force)
        #: 预学的 B 点（None 表示没学过 → 退回"位移够远"判据）
        self.b_target = b_target
        self.frame_size = frame_size or (1920, 1080)
        #: 最近几拍的质心，用来判断"是否停稳"
        self.center_hist = []
        #: 连续"位置对且已停稳"的拍数
        self.moved_hits = 0
        #: 当前质心距 B 点的像素距离（没有 B 点时为 None）
        self.dist_b = None
        self.events = []          # 状态变化时间线
        self.state = S_WAIT
        self.force = 0.0
        self.flat = None
        self.width = None         # 葡萄可见宽度（像素）= 两指间距的视觉估计
        self.center = None        # 当前质心（像素）
        self.center_a = None      # 夹住时的质心（A 点）
        self.move_px = 0.0        # 相对 A 点的位移
        self.air_confirm = 0
        self.vision_lost = 0
        self.width_at_grasp = None
        # ★ 扁平度基线：在"张开、未接触"状态下实测，之后用**相对变化**判夹到。
        #   用相对量而不是绝对阈值，是因为不同距离/光照/葡萄品种的
        #   绝对值差别很大（实测实时流 0.138，存档照片 0.130）。
        self.flat_baseline = None
        self._base_samples = []
        self.baseline_done = False
        #: 采基线阶段连续失败的拍数（用于超时报错，避免无限等待）
        self.base_fail = 0
        # ★ 当前脉宽（由主循环每拍写入）。用来诊断"舵机明明走到头了却没碰到东西"：
        #   如果脉宽已经到 pmin、力却还是 0，那就不是"还在闭合中"，
        #   而是**根本没夹到葡萄**。以前这种情况会静默等到超时，很难查。
        self.pulse = None
        self.pulse_min = None
        self.empty_ticks = 0
        # ★ 扁平度的"历史最大涨幅"。视觉证据是**发生过**的事实：
        #   接触瞬间边缘确实被压直了，之后被夹爪遮挡导致读数回落，
        #   不代表"没夹到"。所以判据用 max_rise 而不是当前 rise。
        #   （2026-11-06 实测：夹住了却一直停在"闭合中"，就是这个原因）
        self.max_rise = 0.0
        # ★ 力的历史最大观测值。
        #   实测踩过：力瞬间冲到 1.87N 就掉回 0.5N 以下，
        #   用"当拍力值"判夹到了，计数器每拍被清零、41 秒都判不出来。
        #   用历史最大值 + 锁存，符合物理事实（**确实来过这么大的力**）。
        self.max_force = None
        self.grasp_latched = False
        # ★ 力到达目标后锁死，不再闭合。防止"力在目标附近抖动 → 又走一步"
        #   的棘轮效应（实测冲到 2.26N，目标只有 1.2N）。
        self.force_latched = False
        # ★ 各类判定的"连续确认"计数器。
        #   单拍读数可能被噪声影响，必须连续多拍一致才认。
        self.contact_hits = 0
        self.grasp_hits = 0
        self.drop_force = 0
        self.drop_squeeze = 0
        self.slip_hits = 0
        self.t0 = time.time()

    # ---------------------------------------------------------------- 记录
    def _emit(self, state, note=""):
        if state != self.state:
            self.state = state
            v = self.vision.latest()
            self.events.append({
                "t": time.time() - self.t0,
                "state": state,
                "force": self.force,
                "vision": v.get("state") if v and v.get("ok") else None,
                "flat": self.flat,
                "width": self.width,
                "note": note,
            })
            f = ("力 %+6.2fN" % self.force) if self.force is not None else "力 --"
            vs = (v.get("state") if v and v.get("ok") else "--")
            cls = (v.get("metrics", {}).get("cls")
                   if v and v.get("ok") else "-")
            fl = ("%.3f" % self.flat) if self.flat else "--"
            wd = ("%.0fpx" % self.width) if self.width else "--"
            print("  [%5.1fs] %-8s %s  视觉 %-2s(%s)  扁平度 %s  宽度 %s  %s"
                  % (time.time() - self.t0, state, f, vs, cls, fl, wd, note),
                  flush=True)

    # ---------------------------------------------------------------- 一拍
    def tick(self, force: float) -> str:
        """
        喂入一次力读数，推进状态机，返回当前状态。

        ★ 关键判断："夹到了" = 传感器 **且** 视觉边缘
        ------------------------------------------------
            传感器  ：力 > CONTACT_N（确实受力了）
            AND
            视觉边缘：扁平度 − 基线 ≥ FLAT_RISE（轮廓被平板压出直边）

        **两者缺一不判。** 这就是导师说的"传感器和视觉边缘检测相结合"。

        为什么这样组合能互救：
            · 只有力上升 → 可能把葡萄推走了、或夹到夹爪自身，
              葡萄并没被夹住（边缘不会变直）
            · 只有边缘变直 → 可能是视角/光照变化造成的错觉（力不会上升）
            两个不同物理量同时变化，才可信。

        刻意不判"夹得多紧"：实测力 >1.0N 后葡萄被夹爪遮挡，
        扁平度反而下降（0.159 → 0.150），这个量在重夹区测不准。
        而"夹到了"发生在接触瞬间，那时葡萄完整可见，信号最清晰。
        """
        self.force = force
        v = self.vision.latest()
        v_ok, vision_ok, v_state = self._judge_vision(v)
        if v_ok:
            self.flat = v.get("flatness")
            self.center = v.get("metrics", {}).get("centroid")
            self.width = v.get("metrics", {}).get("width_px")
            # 记录质心轨迹（近几拍），用于"是否停稳"判断
            if self.center is not None:
                self.center_hist.append((float(self.center[0]),
                                         float(self.center[1])))
                if len(self.center_hist) > STILL_CONFIRM + 2:
                    self.center_hist.pop(0)
                # ★ 距 B 点的距离**每拍都算**，不管当前是什么状态。
                #   以前只在 A→B 那段算，一旦前面的条件不满足，
                #   dist_b 就一直是 None，界面上看不到"还差多远"，
                #   排查时完全没有抓手。
                if self.b_target is not None:
                    fw, fh = self.frame_size
                    bx = float(self.b_target["x_norm"]) * fw
                    by = float(self.b_target["y_norm"]) * fh
                    self.dist_b = ((self.center[0] - bx) ** 2
                                   + (self.center[1] - by) ** 2) ** 0.5
            else:
                self.center_hist.clear()

        # ── 阶段 0：采扁平度基线（此时夹爪张开、未接触）──
        #
        # ⚠️ 2026-11-06 用户反馈"有时候一直卡在采基线"。
        #   原因：这一步需要"视觉连续给出有效葡萄"，任何一环断了就无限等：
        #     ① detect() 要能检出（葡萄完整、未被画面切掉）
        #     ② 稳定器要攒够 3 帧，且质心不能偏离锁定点 >60px
        #     ③ 攒够 4 个样本才开始闭合
        #   而旧代码在这期间**什么都不打印**，只能看到"卡住"。
        #
        #   现在补三件事：
        #     · 每约 3 秒打印一次真实原因（是没检出、还是被门限拒了）
        #     · 超过 6 秒还采不到 → 明确判失败，不再空等到超时
        #     · 失败时**重置稳定器**，允许它重新锁定目标
        #       （否则一旦先锁到别的绿色物体，真葡萄就永远被门限拒掉）
        if not self.baseline_done:
            if v_ok and self.flat is not None:
                self._base_samples.append(float(self.flat))
                self.base_fail = 0
                n = len(self._base_samples)
                if n == 1:
                    self._emit(S_WAIT, "开始采扁平度基线（保持张开）")
                if n >= BASELINE_MIN_SAMPLES:
                    self.flat_baseline = statistics.median(self._base_samples)
                    self.baseline_done = True
                    self._emit(S_CLOSING,
                               "基线 %.4f（%d 个样本）→ 开始闭合"
                               % (self.flat_baseline, n))
            else:
                self.base_fail += 1
                if self.base_fail % 10 == 1:
                    why = (v or {}).get("reason") or "视觉未就绪"
                    st = self.vision.stab
                    print("  ⏳ 采基线中（第 %d 拍）：%s"
                          "　[稳定器 已接受 %d / 已拒绝 %d]"
                          % (self.base_fail, why,
                             st.accepted, st.rejected), flush=True)
                if self.base_fail >= BASELINE_FAIL_TICKS:
                    # ★ 不要退出 —— 保持进程活着，网页预览才能继续显示画面，
                    #   用户可以边看边调整机位，等葡萄出现在画面里就会自动开始。
                    #   （2026-11-06：原来这里直接判失败退出，导致
                    #     "用预览调取景"这件事根本做不到）
                    print("  ⚠️ 已经 %.0f 秒没能锁定葡萄：%s"
                          % (self.base_fail * 0.1,
                             (v or {}).get("reason") or "未知"),
                          flush=True)
                    print("     → 保持张开等待中，请调整机位/葡萄位置；"
                          "网页预览能看到实时画面", flush=True)
                    self.vision.stab.reset()   # 允许重新锁定
                    self.base_fail = 0
            return self.state

        # ── 计算扁平度涨幅 + 记录历史最大涨幅 ──
        rise = None
        if self.flat is not None and self.flat_baseline is not None:
            rise = self.flat - self.flat_baseline
            if rise > self.max_rise:
                self.max_rise = rise
        # 记录力的历史最大值（波动大，峰值才反映"确实受过力"）
        if force is not None:
            if self.max_force is None or force > self.max_force:
                self.max_force = force
        # 力上限：超过说明两指已经压在一起（葡萄掉了或根本没夹到）
        ceil_n = self.target_force + CEIL_MARGIN

        # ── 已夹住之后：滑移 / A→B / 脱落 ──
        if self.state in (S_HOLDING, S_MOVED, S_SLIPPING):
            # ══════════════════════════════════════════════════════
            # 脱落判定：**以视觉为主，力只做辅助**
            #
            # 为什么不用"力超过上限"：
            #   ① PS01 单样本 σ≈2.6N，还有周期性尖峰
            #   ② 舵机本身会抖，脉宽微小变化就带动力变化
            #   用力大小当阈值，噪声就能把它顶穿（实测误判过）
            #
            # 视觉为什么可靠：葡萄在两指间时，边缘**一定**被压平；
            # 掉落后两指闭到彼此，中间没东西可压，扁平度回落到基线。
            # 这是几何事实，不受力噪声和舵机抖动影响。
            # ══════════════════════════════════════════════════════

            # ① 视觉主判据：挤压证据消失（连续多拍）
            #
            # ★ 2026-11-06 重要放宽：**只要力还很足、葡萄也还看得见，
            #   扁平度回落就不算脱落的证据。**
            #   原因：葡萄被夹住后，扁平度会因为遮挡/曝光变化而掉回基线附近
            #   （实测多次：夹得好好的，涨幅却变成 0.00x）。
            #   旧逻辑会让它在 A→B 之前就误判"脱落"，
            #   于是用户"移到 B 点"永远等不到判定 —— 这就是他反馈的现象。
            #
            #   现在"还夹着"的证据取**两者之一**：
            #     · 扁平度仍高于基线（边缘被压直）
            #     · 力仍 ≥ HOLD_N 且视觉能看见葡萄（更可靠）
            held_solid = (force >= HOLD_N and vision_ok)
            if (rise is not None and rise < FLAT_RISE * SQUEEZE_KEEP
                    and not held_solid):
                self.drop_squeeze += 1
            else:
                self.drop_squeeze = 0
            if self.drop_squeeze >= DROP_SQUEEZE_CONFIRM:
                self._emit(S_DROPPED,
                           "扁平度涨幅回落到 %+.4f（<%.3f）且力只有 %.2fN，"
                           "连续 %d 拍 → 已不再被挤压，葡萄脱落"
                           % (rise or 0.0, FLAT_RISE * SQUEEZE_KEEP,
                              force, self.drop_squeeze))
                return self.state

            # ② 视觉判据：葡萄离开视野
            #
            # ⚠️ 2026-11-06 实测事故（视频 + 日志确认）：
            #   葡萄被紧紧夹在两指之间时，**视觉会连续 30 秒认不出来**
            #   （压扁、被黑块遮挡、绿色区域被切碎）。
            #   当时这条规则把它判成"脱落"，但葡萄其实好好夹着。
            #
            #   所以"看不见"**不能单独作为脱落的证据** ——
            #   夹持状态下看不见是正常的。
            #   只有"看不见 **且** 力也很低"才说明真的掉了。
            if not vision_ok:
                self.vision_lost += 1
            else:
                self.vision_lost = 0
            if (self.vision_lost >= DROP_CONFIRM
                    and force < GRASP_N):
                self._emit(S_DROPPED,
                           "葡萄连续 %d 拍不在视野内，且力只有 %.2fN "
                           "→ 判断为脱落"
                           % (self.vision_lost, force))
                return self.state

            # ③ 力辅助判据（力骤降）已在上面统一处理，这里不再重复
            #    否则 drop_force 会被计两次，确认拍数减半

            if self.state != S_MOVED:
                # ④ 滑移：可见宽度变大（葡萄在往外挤），连续确认
                if (self.width is not None and self.width_at_grasp
                        and self.width - self.width_at_grasp > SLIP_WIDTH_PX):
                    self.slip_hits += 1
                else:
                    self.slip_hits = 0
                if self.slip_hits >= SLIP_CONFIRM:
                    self._emit(S_SLIPPING,
                               "可见宽度 %.0f→%.0f px 连续 %d 拍"
                               % (self.width_at_grasp, self.width,
                                  self.slip_hits))
                    return self.state

                # ⑤ A→B：**到了 B 点并且停稳了**（学过 B）或 **位移够远**（没学过 B）
                #   "还夹着"的证据和上面一样取两者之一：
                #   扁平度仍高于基线，**或** 力还足且葡萄可见。
                #   （只用扁平度会把"夹得好好的但读数回落"挡在门外，
                #     用户实测"移到 B 点没反应"就是这个原因）
                held_ok = ((rise is not None
                            and rise >= FLAT_RISE * SQUEEZE_KEEP)
                           or (force >= HOLD_N and vision_ok))
                if (self.center is not None and force >= GRASP_N
                        and held_ok):
                    still = self._is_still()
                    if self.b_target is not None:
                        # ── 有预学的 B 点：判"是否真的到了 B" ──
                        fw, fh = self.frame_size
                        bx = float(self.b_target["x_norm"]) * fw
                        by = float(self.b_target["y_norm"]) * fh
                        tol = float(self.b_target.get("tol_px", B_TOL_PX))
                        db = ((self.center[0] - bx) ** 2
                              + (self.center[1] - by) ** 2) ** 0.5
                        self.dist_b = db
                        at_b = db <= tol
                        if at_b and still:
                            self.moved_hits += 1
                        else:
                            self.moved_hits = 0
                        if self.moved_hits >= MOVED_CONFIRM:
                            self._emit(S_MOVED,
                                       "距 B 点 %.0fpx ≤ 容差 %.0fpx，"
                                       "且已停稳 %d 拍；力 %.2fN，"
                                       "扁平度仍 +%.4f → 已移到B"
                                       % (db, tol, self.moved_hits,
                                          force, rise))
                    else:
                        # ── 没学过 B 点：退回"位移够远" ──
                        if self.center_a is not None:
                            d = ((self.center[0] - self.center_a[0]) ** 2
                                 + (self.center[1] - self.center_a[1]) ** 2
                                 ) ** 0.5
                            self.move_px = d
                            if d >= A2B_PX and still:
                                self.moved_hits += 1
                            else:
                                self.moved_hits = 0
                            if self.moved_hits >= MOVED_CONFIRM:
                                self._emit(S_MOVED,
                                           "位移 %.0f px（未学 B 点，用位移判定），"
                                           "已停稳 %d 拍；力 %.2fN，"
                                           "扁平度仍 +%.4f → 已移到B"
                                           % (d, self.moved_hits, force, rise))
            return self.state

        # ── 脱落检测：一旦"接触过"，就一直生效 ──
        #
        # ★ 2026-11-06 实测事故：用户中途把葡萄弄掉了，程序却没结束，
        #   一直跑到 45 秒超时。原因 —— 端落检测原来只写在
        #   "夹到了/已移到B/滑移中"这三个分支里；而那一次视觉锁到了
        #   黑色传感器块（矩形），扁平度全程不变，导致"接触到"从未触发，
        #   状态一直停在"闭合中"，脱落检测根本没被启用。
        #
        #   正确做法：**只要确实接触过**（力曾经上过 CONTACT_N），
        #   之后力掉没了就判脱落，不管有没有成功判出"夹到了"。
        if (self.state in (S_CLOSING, S_TOUCHED, S_HOLDING,
                           S_MOVED, S_SLIPPING)
                and self.max_force is not None
                # ★ 门槛必须够高。原来用 CONTACT_N(0.05N)，
                #   但力的噪声有 ±0.6N，这个门槛会被噪声顶穿 ——
                #   用户实测：**画面里根本没葡萄**，也被判成"脱落"。
                #   现在要求"确实夹到过"（力峰值 ≥ HOLD_N）才启用脱落检测；
                #   没夹到的情况由"夹空"路径处理（脉宽到下限 + 扁平度没上升）。
                and self.max_force >= HOLD_N):
            if force < RELEASE_N:
                self.drop_force += 1
            else:
                self.drop_force = 0
            if self.drop_force >= DROP_FORCE_CONFIRM:
                self._emit(S_DROPPED,
                           "接触过（力峰值 %.2fN）但力连续 %d 拍 "
                           "低于 %.2fN（当前 %+.2fN）→ 葡萄脱落"
                           % (self.max_force, self.drop_force,
                              RELEASE_N, force))
                return self.state

        # ── 闭合中 ──
        if self.state == S_CLOSING:
            # ★ 诊断：脉宽已经到最小、力却一直是 0 → 根本没夹到东西
            #   这比"静默等到超时"有用得多（2026-11-06 实测踩过：
            #   葡萄被上一轮推走，45 秒里力一直 -0.27N，看不出原因）
            if (self.pulse is not None and self.pulse_min is not None
                    and self.pulse <= self.pulse_min + 1.0
                    and force < CONTACT_N):
                self.empty_ticks += 1
                if self.empty_ticks >= 5:
                    self._emit(S_AIR,
                               "脉宽已到下限 %.0fµs，力仍 %.2fN → "
                               "两指之间没有葡萄（或葡萄被推走了）"
                               % (self.pulse, force))
                return self.state
            else:
                self.empty_ticks = 0

            # ★ 夹空判定：改成**机械 + 视觉**，不看力的大小
            #   ① 脉宽走到下限 = 两指物理上已经完全闭拢（这是机械事实）
            #   ② 扁平度从未明显上升 = 中间没有东西被压过（视觉）
            #   两条同时成立 → 夹空。
            #   ⚠️ 不用"力超过上限"来判，因为力本身不稳（PS01 σ≈2.6N）。
            if (self.pulse is not None and self.pulse_min is not None
                    and self.pulse <= self.pulse_min + 1.0):
                self.empty_ticks += 1
                if self.empty_ticks >= 3:
                    if self.max_rise >= FLAT_RISE:
                        # 曾经碰到过，只是力没上来 —— 可能是接触很轻
                        self._emit(S_AIR,
                                   "脉宽到下限 %.0fµs；扁平度曾涨 %+.4f "
                                   "但力不足 → 接触过轻，未夹住"
                                   % (self.pulse, self.max_rise))
                    else:
                        self._emit(S_AIR,
                                   "脉宽到下限 %.0fµs，且扁平度从未上升"
                                   "（最大 %+.4f < %.3f）→ 两指之间没有葡萄"
                                   % (self.pulse, self.max_rise, FLAT_RISE))
                return self.state
            else:
                self.empty_ticks = 0

            # ★ "接触到"：力上升 AND 边缘曾变直，两者结合
            #   两个量都用**历史最大值**，因为力和扁平度都会波动：
            #   · 力会瞬间冲高再回落（实测 1.87N → 0.5N 以下）
            #   · 扁平度在接触后被遮挡会回落
            #   用峰值判断符合物理事实：**确实来过这么大的力 / 确实被压平过**。
            if self.max_force > CONTACT_N and self.max_rise >= FLAT_RISE:
                self.contact_hits += 1
            else:
                self.contact_hits = 0
            if self.contact_hits >= CONTACT_CONFIRM:
                self._emit(S_TOUCHED,
                           "力峰值 %.2fN(>%.2f) + 扁平度最大涨幅 %+.4f(≥%.3f)"
                           " 连续 %d 拍 → 传感器与视觉边缘一致"
                           % (self.max_force, CONTACT_N, self.max_rise,
                              FLAT_RISE, self.contact_hits))
            return self.state

        if self.state == S_TOUCHED:
            # ★ "夹到了"：**用峰值判，并锁存**
            #   力达到 HOLD_N 且视觉证据成立时，就认定夹住了。
            #   不再要求"连续 3 拍都满足当前值" —— 因为力会掉，
            #   那样会永远判不出来（实测踩过：41 秒卡在"接触到"）。
            if (self.max_force is not None and self.max_force >= HOLD_N
                    and self.max_rise >= FLAT_RISE):
                self.grasp_hits += 1
            else:
                self.grasp_hits = 0
            if self.grasp_hits >= GRASP_CONFIRM:
                self.grasp_latched = True
                self.center_a = self.center
                self.width_at_grasp = self.width
                self._emit(S_HOLDING,
                           "力峰值 %.2fN ≥ %.2fN，扁平度涨幅 %+.4f ≥ %.3f "
                           "连续 %d 拍 → 夹到了（已锁存）"
                           % (self.max_force, HOLD_N, self.max_rise,
                              FLAT_RISE, self.grasp_hits))
            return self.state

        return self.state

    # ------------------------------------------------------------ 停稳判定
    #
    # ⚠️ 这个方法必须放在 tick() **之外**。
    #    2026-11-06 我把它的定义插到了 tick() 中间，
    #    结果 tick() 在插入点就结束了，"脱落检测 / 闭合中 / 接触到"
    #    整段代码变成了本方法的函数体（死代码）。
    #    表现是：tick() 在"闭合中"状态落空返回 None，
    #    主循环里 st == S_CLOSING 不成立 → 脉宽永远不动 →
    #    力一直 0、也不报夹空，卡 60 秒。
    def _is_still(self) -> bool:
        """
        葡萄是否"停下来了"。

        ★ 为什么需要：用户反馈"我一移动它就判定为 A 到 B"。
          只有"到达位置"不够 —— 路过也算到达。必须**停下来**才算到位。
          判据：相邻两拍质心移动都很小，且连续够多拍（STILL_CONFIRM）。
        """
        h = self.center_hist
        if len(h) < STILL_CONFIRM + 1:
            return False
        for i in range(1, len(h)):
            step = ((h[i][0] - h[i - 1][0]) ** 2
                    + (h[i][1] - h[i - 1][1]) ** 2) ** 0.5
            if step > STILL_PX:
                return False
        return True

    # ------------------------------------------------------------ 视觉判定
    @staticmethod
    def _judge_vision(v):
        """
        把视觉结果压缩成两个布尔量。

        返回 (v_ok, vision_ok, v_state)：
            v_ok      视觉这一拍有没有产出有效结果（稳定器已热身）
            vision_ok 葡萄是否还在视野里且轮廓完整
            v_state   "自由"/"贴住"/"夹紧"，仅用于记录，**不参与判定**
        """
        if not v or not v.get("ok"):
            return False, False, None
        m = v.get("metrics") or {}
        solidity = m.get("solidity")
        # 轮廓被切碎/遮挡严重时实心度会掉 —— 这是"还看得见吗"的判据
        if solidity is not None and solidity < SOLID_OK:
            return True, False, v.get("state")
        return True, True, v.get("state")

    # ---------------------------------------------------------------- 汇总
    def summary(self) -> str:
        n = len(self.events)
        return ("  状态序列共 %d 个事件，耗时 %.1fs\n" % (n,
                                                         time.time() - self.t0))


def main() -> int:
    ap = argparse.ArgumentParser(description="具身抓取闭环（视觉+力双模）")
    ap.add_argument("--object", default="grape")
    ap.add_argument("--target", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="不驱动舵机，只跑观测循环")
    ap.add_argument("--max-seconds", type=float, default=45.0,
                    help="总时长上限（默认 45 秒，超时自动收尾）")
    ap.add_argument("--vision-hz", type=float, default=3.0,
                    help="视觉判定频率上限（默认 3 Hz，约半核）")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--no-preview", action="store_true",
                    help="不启动网页实时预览（默认启动，端口 8082）")
    ap.add_argument("--preview-port", type=int, default=8082)
    ap.add_argument("--learn-b", action="store_true",
                    help="预学 B 点：把葡萄放在目标位置后运行，按回车记录")
    ap.add_argument("--b-file", default=None,
                    help="B 点文件路径（默认项目根目录 b_target.json）")
    ap.add_argument("--b-tol", type=float, default=B_TOL_PX,
                    help="到 B 点的容差（像素，默认 %.0f）" % B_TOL_PX)
    args = ap.parse_args()

    cfg = Config.load(default_config_path())
    base = "grasp_policy.%s" % args.object
    target = (args.target if args.target is not None
              else cfg.get_float(base + ".target_force_n", 0.30))
    approach = cfg.get_float(base + ".approach_pulse", 940.0)
    pmin = cfg.get_float("servo.pulse_min_us", 750.0)
    pmax = cfg.get_float("servo.pulse_max_us", 1450.0)
    k = cfg.get_float("sensor.calib.right.counts_per_n", 1564.0)

    print("=" * 74)
    print("具身抓取闭环：视觉持续观测 + 力传感 + 双模判定")
    print("=" * 74)
    print("  策略：目标力 %.2fN，预闭合脉宽 %.0fµs，脉宽范围 %.0f~%.0f"
          % (target, approach, pmin, pmax))
    print("  ── 判定规则（传感器与视觉边缘结合，全部需连续 N 拍确认）──")
    print("  「接触到」= 力 > %.2fN  AND  扁平度最大涨幅 ≥ %+.3f  （连续 %d 拍）"
          % (CONTACT_N, FLAT_RISE, CONTACT_CONFIRM))
    print("  「夹到了」= 力 ≥ %.2fN  AND  扁平度最大涨幅 ≥ %+.3f  （连续 %d 拍）"
          % (HOLD_N, FLAT_RISE, GRASP_CONFIRM))
    _b_now = load_b_target(args.b_file)
    if _b_now:
        print("  「已移到B」= 距 B 点 ≤ %.0fpx AND 已停稳 AND 力 ≥ %.2fN"
              " AND 扁平度涨幅仍 ≥ %+.3f"
              % (_b_now.get("tol_px", B_TOL_PX), GRASP_N,
                 FLAT_RISE * SQUEEZE_KEEP))
    else:
        print("  「已移到B」= 位移 ≥ %.0fpx AND 已停稳 AND 力 ≥ %.2fN"
              " AND 扁平度涨幅仍 ≥ %+.3f"
              % (A2B_PX, GRASP_N, FLAT_RISE * SQUEEZE_KEEP))
    print("                （位置对 + 停稳 需连续 %d 拍）" % MOVED_CONFIRM)
    print("  「脱落」  = 扁平度涨幅回落 < %+.3f（视觉主判据，连续 %d 拍）"
          % (FLAT_RISE * SQUEEZE_KEEP, DROP_SQUEEZE_CONFIRM))
    print("              或 葡萄连续 %d 拍不在视野内" % DROP_CONFIRM)
    print("              或 力连续 %d 拍低于 %.2fN（力辅助）"
          % (DROP_FORCE_CONFIRM, RELEASE_N))
    print("  ⚠️ 力的上限 %.2fN **只用于控制**（防压坏），不参与判定 ——"
          % (target + CEIL_MARGIN))
    print("     因为 PS01 单样本 σ≈2.6N 且有尖峰、舵机也会抖，")
    print("     用力大小当判定阈值会被噪声顶穿。")
    print("  力上限 %.2fN = 目标 %.2fN + 余量 %.2fN；到目标后锁死不再闭合"
          % (target + CEIL_MARGIN, target, CEIL_MARGIN))
    print("  视觉频率上限 %.1f Hz（每帧约 133 ms；限速是为了不过热）"
          % args.vision_hz)
    print("  总时长上限 %.0f 秒" % args.max_seconds)
    # 显示 B 点状态
    _b = load_b_target(args.b_file)
    if args.learn_b:
        print("  📌 模式：预学 B 点（把葡萄放到目标位置，按回车）")
    elif _b:
        print("  🎯 B 点已预学：画面比例 (%.3f, %.3f)，容差 ±%.0fpx"
              % (_b["x_norm"], _b["y_norm"],
                 _b.get("tol_px", B_TOL_PX)))
        print("     → 「已移到B」= 到达 B 点 且 停稳 且 仍夹着")
    else:
        print("  ⚠️ 未预学 B 点 → 「已移到B」退回「位移 ≥ %.0fpx」判定"
              % A2B_PX)
        print("     想让它判「真的到了B」，运行一次：--learn-b")
    print()

    # ---------------- 视觉 ----------------
    det = GrapeStateDetector()
    vision = VisionWorker(det, min_interval=1.0 / max(args.vision_hz, 0.5))
    try:
        vision.start()
    except Exception as exc:  # noqa: BLE001
        print(f"❌ 视觉启动失败：{exc}")
        return 1

    # ---------------- 实时预览（网页）----------------
    #   ⚠️ 必须放在"预学 B 点"**之前**启动：
    #      学习 B 点时正是需要一边看画面一边摆葡萄的时候，
    #      第一版把它放在后面，学习模式下压根没有预览（已被用户指出）。
    preview = None
    if not args.no_preview:
        try:
            preview = PreviewServer(vision, port=args.preview_port).start()
            print("  📺 实时预览：  http://<板子IP>:%d/   （浏览器打开即可看全过程）"
                  % args.preview_port)
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠️ 预览服务启动失败（不影响抓取）：{exc}")

    # ---------------- 预学 B 点模式 ----------------
    #   把葡萄放到目标位置 → 记录此刻的质心（存成画面比例）
    #   之后「已移到B」就判"是否真的到了这个点"，而不是"动了多远"。
    if args.learn_b:
        print("\n📌 学习 B 点")
        print("   ① 把葡萄放到你希望它最终到达的位置（B 点）")
        print("   ② 等下面显示「已锁定葡萄」")
        print("   ③ **随时**按回车记录\n")

        # ★ 回车由后台线程收，主循环只管持续刷新状态。
        #   这样"想按就能按"，不用等固定时长。
        #   （第一版写的是"先等 40 秒再 input()"，用户必须干等，
        #     自动化测试也喂不进回车 —— 已改掉）
        entered = threading.Event()

        def _wait_enter():
            try:
                input()
            except (EOFError, KeyboardInterrupt):
                pass
            entered.set()

        threading.Thread(target=_wait_enter, daemon=True).start()

        is_tty = sys.stdout.isatty()
        t0 = time.time()
        last_print = 0.0
        c = None
        while not entered.is_set() and time.time() - t0 < 120:
            v = vision.latest()
            m = (v or {}).get("metrics") or {}
            c = m.get("centroid") or c
            now = time.time()
            if now - last_print >= (0.5 if is_tty else 2.0):
                last_print = now
                if v and v.get("ok") and m.get("centroid"):
                    txt = ("   ✅ 已锁定葡萄  质心 (%.0f, %.0f)  "
                           "宽 %.0fpx  扁平度 %.3f  （按回车记录）"
                           % (m["centroid"][0], m["centroid"][1],
                              m.get("width_px") or 0,
                              v.get("flatness") or 0))
                else:
                    txt = "   ⏳ 等待视觉锁定葡萄…"
                if is_tty:
                    print("\r" + txt + " " * 6, end="", flush=True)
                else:
                    print(txt, flush=True)
            time.sleep(0.3)
        if is_tty:
            print()
        if not entered.is_set():
            print("   ⏱ 等了 120 秒没收到回车，放弃")
        if c is None:
            print("❌ 没能锁定葡萄，请检查取景后重试")
            vision.stop()
            return 1
        v = vision.latest()
        m = (v or {}).get("metrics") or {}
        c = m.get("centroid") or c
        H, W = 1080, 1920
        p = save_b_target(c[0] / W, c[1] / H, (W, H), args.b_tol,
                          args.b_file)
        print("\n   ✅ 已记录 B 点：")
        print("      质心像素 (%.0f, %.0f)  =  画面比例 (%.4f, %.4f)"
              % (c[0], c[1], c[0] / W, c[1] / H))
        print("      容差 ±%.0fpx" % args.b_tol)
        print("      已保存到 %s" % p)
        print("\n   以后直接运行（不带 --learn-b）就会判「是否真的到了 B」")
        if preview is not None:
            preview.stop()
        vision.stop()
        return 0

    # ---------------- 力 ----------------
    reader = None
    raw_zero = None
    if not args.dry_run:
        try:
            from grape_gripper.ps01 import Ps01Reader
            port = cfg.get_str("sensor.ps01.port", "/dev/ttyUSB0")
            reader = Ps01Reader(port)
            reader.open()
            if not reader.wait_first_sample(timeout=3.0):
                raise RuntimeError("传感器无数据")
            # 实测空载基线（不用配置常量：配置里是传感器 B 的基线）
            vals = []
            t0 = time.time()
            while time.time() - t0 < 1.2:
                v = reader.channel(1)
                if v is not None:
                    vals.append(float(v))
                time.sleep(0.002)
            raw_zero = statistics.median(vals) if vals else None
            print(f"  力基线实测：{raw_zero:.0f} counts（{len(vals)} 个样本）")
        except Exception as exc:  # noqa: BLE001
            print(f"❌ 力传感器启动失败：{exc}")
            vision.stop()
            return 1

    servo = None
    if not args.dry_run:
        try:
            from grape_gripper.actuator import BusyPwmServo
            servo = BusyPwmServo(gpio_pin=cfg.get_int("servo.gpio", 49),
                                 period_us=20000.0, pulse_min_us=pmin,
                                 pulse_max_us=pmax, angle_min=0.0,
                                 angle_max=180.0)
        except Exception as exc:  # noqa: BLE001
            print(f"❌ 舵机启动失败：{exc}")
        else:
            # 显式确认舵机对象已建立（2026-11-06：之前只在不报错时静默通过，
            # 出问题时无法从日志判断舵机是否可用）
            print("  ✅ 舵机已就绪（GPIO %d，脉宽 %.0f~%.0fµs）"
                  % (cfg.get_int("servo.gpio", 49), pmin, pmax))

    def measure(window=0.35, n_ret=1):
        """
        测力（取窗口中位数）。

        ★ window 从 0.22 提到 0.35（2026-11-06：用户反馈"抖动大"）。
          PS01 单样本 σ≈2.6N，中位数的标准误差 ≈ 1.253σ/√n。
          0.22s 约 110 个样本 → SE≈0.31N；
          0.35s 约 175 个样本 → SE≈0.25N。窗口再长就跟不上状态推进了。
        """
        if reader is None or raw_zero is None:
            return 0.0
        vals = []
        t0 = time.time()
        while time.time() - t0 < window:
            v = reader.channel(1)
            if v is not None:
                vals.append(float(v))
            time.sleep(0.002)
        if len(vals) < 4:
            return 0.0
        vals.sort()
        return (vals[len(vals) // 2] - raw_zero) / k

    # ---------------- 主循环 ----------------
    machine = EmbodiedGrasp(vision, dry_run=args.dry_run,
                            verbose=args.verbose, target_force=target,
                            b_target=load_b_target(args.b_file),
                            frame_size=(1920, 1080))
    # ★ 采基线阶段必须**全开**：基线要的是"葡萄完全没被压"时的扁平度。
    #   如果一上来就预闭合到 approach（940µs），两指可能已经贴到葡萄，
    #   基线就被污染了，"扁平度上升"这个判据会失效。
    pulse = pmax
    if servo is not None:
        servo.set_pulse_us(pmax)
        print(f"  张开到 {pmax:.0f}µs，采集扁平度基线…")
        # ★ 重新测力基线：上面的基线是在舵机还没动时测的，
        #   舵机张开后两指位置变了，传感器读数会跟着变。
        #   基线要在**最终的开爪姿态**下测，否则会引入固定的偏置
        #   （实测出现过张开状态显示 -3.4N 的偏置）。
        time.sleep(1.0)
        try:
            vals = []
            t0b = time.time()
            while time.time() - t0b < 1.2:
                v = reader.channel(1)
                if v is not None:
                    vals.append(float(v))
                time.sleep(0.002)
            if len(vals) >= 20:
                old = raw_zero
                raw_zero = statistics.median(vals)
                print("  力基线（张开姿态下重测）：%.0f counts"
                      "（原 %.0f，差 %+.0f）"
                      % (raw_zero, old, raw_zero - old))
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠️ 重测力基线失败（沿用旧值）：{exc}")
    machine.pulse_min = pmin

    ok = False
    #: 主循环节拍（秒）。10 Hz 足够跟住夹爪动作，又不抢视觉线程的 GIL。
    #: ⚠️ 没有节拍会出两个问题：
    #:   ① dry-run 时 measure() 立即返回 → 空转把 GIL 占满，
    #:      视觉线程被饿到 1 fps（实测）；
    #:   ② verbose 时同一时刻刷出几百行。
    TICK = 0.10
    force_hist = []        # 最近几拍的力，用于"持续中位数"判到目标
    try:
        while True:
            t_tick = time.time()
            el = time.time() - machine.t0
            if el > args.max_seconds:
                print("  ⏱ 到达时长上限，收尾")
                break
            f = measure()
            machine.pulse = pulse          # 让状态机能判断"脉宽到下限了没"
            st = machine.tick(f)
            # 让网页预览上的文字跟着状态走
            vision.overlay_force = "%+.2fN" % f
            vision.overlay_state = st
            vision.overlay_pulse = pulse

            if args.verbose:
                v = vision.latest()
                _m = (v.get("metrics") or {}) if v else {}
                print("    t=%5.1fs 力=%+6.3fN 视觉=%s(%s) 扁平度=%s "
                      "宽=%s 圆度=%s 顶点=%s 周长=%s 状态=%s"
                      % (el, f,
                         (v.get("state") if v and v.get("ok") else "--"),
                         _m.get("cls") or "-",
                         ("%.3f" % v["flatness"]) if v and v.get("ok")
                         and v.get("flatness") else "--",
                         ("%.0f" % _m["width_px"]) if _m.get("width_px")
                         else "--",
                         ("%.3f" % _m["circ"]) if _m.get("circ") else "--",
                         v.get("poly_points") if v else "--",
                         ("%.0f" % v["perimeter_px"]) if v
                         and v.get("perimeter_px") else "--",
                         st))

            # 闭合：力还没到目标就继续收紧（脉宽减小 = 更紧）
            # 注意：采基线阶段（S_WAIT）**不能闭合**，否则基线会被污染
            #
            # ★ 2026-11-06 关键修正（用户实测"贴住之后脉宽不动、夹不起来"）：
            #   旧逻辑用**单拍峰值**判"到目标"，然后还**回退 8µs**。
            #   实测：脉宽 916µs 时力冲到 1.602N → 立刻锁存 → 退到 924 → 932，
            #   结果把夹爪松开了 16µs，稳态力掉到 0.25N，葡萄滑落。
            #   现在改成：
            #     ① 用**最近 3 拍的持续中位数**判到目标（峰值不算数）
            #     ② 到目标**只停、绝不回退**
            force_hist.append(f)
            if len(force_hist) > 3:
                force_hist.pop(0)
            sustained = (statistics.median(force_hist)
                         if len(force_hist) >= 2 else f)

            if st == S_CLOSING and pulse > approach:
                # 基线采完了，先快速预闭合到 approach，再逐步收紧
                pulse = approach
                if servo is not None:
                    servo.set_pulse_us(pulse)
            elif servo is not None and st in (S_CLOSING, S_TOUCHED,
                                              S_HOLDING):
                if not machine.force_latched:
                    if sustained >= target:
                        machine.force_latched = True
                        print("  ⏹ 力持续 %.2fN ≥ 目标 %.2fN → 停止闭合"
                              "（脉宽保持 %.0fµs，不回退）"
                              % (sustained, target, pulse), flush=True)
                    elif pulse > pmin:
                        # 接近目标时用小步长，避免过冲
                        step = 4.0 if sustained < target * 0.7 else 2.0
                        pulse -= step
                        servo.set_pulse_us(pulse)

            if args.verbose:
                print("        脉宽 %.0fµs  持续力 %.3fN  已锁定=%s"
                      % (pulse, sustained, machine.force_latched))

            # ★ 诊断：不用 -v 也每 2 秒报一次脉宽。
            #   2026-11-06 用户实测"60 秒一直闭合中、力一直是 0、也没报夹空"，
            #   从日志完全看不出**脉宽到底有没有在减小** ——
            #   如果是舵机没动（pulse 不变），现象就完全吻合。
            #   所以把脉宽打出来，一眼就能分辨：
            #     脉宽在降、力不涨  → 葡萄不在这条闭合路径上（位置问题）
            #     脉宽不动          → 舵机没收到指令（硬件/驱动问题）
            if el - getattr(machine, "_last_pulse_log", -99.0) >= 2.0:
                machine._last_pulse_log = el
                # 距 B 点 / 停稳 / 扁平度涨幅 —— 这三项决定 A→B 能不能触发
                if machine.b_target is not None:
                    _tol = float(machine.b_target.get("tol_px", B_TOL_PX))
                    _pos = ("%.0fpx(容差%.0f)" % (machine.dist_b, _tol)
                            if machine.dist_b is not None else "--")
                else:
                    _pos = "无B点"
                _still = "是" if machine._is_still() else "否"
                _rise = ("%+.3f" % (machine.flat - machine.flat_baseline)
                         if machine.flat is not None
                         and machine.flat_baseline is not None else "--")
                print("    [t=%4.1fs] 脉宽 %.0fµs 力 %+.2fN 状态 %s"
                      "  距B=%s 停稳=%s 扁平度涨幅=%s"
                      % (el, pulse, f, st, _pos, _still, _rise),
                      flush=True)

            if st == S_MOVED:
                ok = True
                break
            if st in (S_DROPPED, S_FAIL, S_AIR):
                break

            # ★ 节拍：把这一拍剩余的时间睡掉，别空转
            sleep_left = TICK - (time.time() - t_tick)
            if sleep_left > 0:
                time.sleep(sleep_left)
    except KeyboardInterrupt:
        print("\n  中断")
    finally:
        print()
        print("=" * 74)
        print("状态序列")
        print("=" * 74)
        if machine.events:
            print("  %-8s %-10s %-9s %-8s %-9s %s"
                  % ("时间", "状态", "力(N)", "视觉", "扁平度", "宽度"))
            for e in machine.events:
                print("  %6.1fs  %-10s %+8.2f  %-8s %-9s %s"
                      % (e["t"], e["state"], e["force"],
                         e["vision"] or "--",
                         ("%.3f" % e["flat"]) if e["flat"] else "--",
                         ("%.0fpx" % e["width"]) if e.get("width") else "--"))
            print()
            print("  ── 关键判定的依据（每一条都有力和视觉两个信号）──")
            for e in machine.events:
                if e["state"] in (S_TOUCHED, S_HOLDING, S_AIR, S_DROPPED,
                                  S_SLIPPING, S_MOVED):
                    print("    %-8s  力 %+6.2fN  视觉 %s  →  %s"
                          % (e["state"], e["force"], e["vision"] or "--",
                             e["note"]))
        else:
            print("  （没有状态变化）")
        print()
        print(f"  视觉：{vision.frames} 帧，{vision.fps:.1f} fps，"
              f"错误 {vision.errors}")
        # 收尾：张开、关流
        try:
            if servo is not None:
                servo.set_pulse_us(pmax)
                time.sleep(0.3)
                servo.close()
        except Exception:  # noqa: BLE001
            pass
        if preview is not None:
            preview.stop()
        vision.stop()
        try:
            if reader is not None:
                reader.close()
        except Exception:  # noqa: BLE001
            pass

    print()
    print("  结论：%s" % ("✅ 已移到 B" if ok else "未完成 A→B"))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())