#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拍照 + 调机位的网页工具（参考 8081，与旧的 camera_preview.py 无关）。

为什么这样设计
--------------
2026-11-05 用旧的 camera_preview.py（16 fps 连续推流）把板子跑到 79°C，
拔电后 UEFI 启动项损坏、掉进 Shell，折腾两小时才恢复。

**根本原因不是"网页"这个形式，是"连续推流"。** 所以这里改成：

    只有点「拍一张」时才启动 cam-test（约 1.2 秒）
    其余时间相机完全关闭，CPU 占用 0

舵机是另一回事：BusyPwmServo 在 20ms 周期里只忙等约 0.85ms
（高电平那一下），低电平用 sleep —— **实测约 4% 单核**，
所以"保持某个脉宽"是安全的。但为了保险，闲置 3 分钟会自动张开并释放。
（见 actuator.py:597 的 _loop）

功能
----
    ① 实时看力（相对基线的变化量）和温度
    ② 设舵机脉宽 / 张开 / 自动扫接触点
    ③ 点一下拍一张，显示 5 个指标 + 判定
    ④ 按类别保存（自由/贴住/夹紧/标记）到 debug/set_a..set_d
       —— 上次拍 31 张时把目录名打成了 set_ ，这次用按钮点，不会错

安全措施
--------
    · 相机只在拍照时开，无后台循环
    · 舵机闲置 180 秒自动张开 + 释放
    · 总运行 --max-minutes（默认 20）分钟自动退出
    · 温度 > 70°C 直接拒绝拍照并提示
    · 页面显示当前温度，随时能看到

用法::

    sudo systemctl stop grape-gripper      # 先停服务，避免抢串口
    sudo python3 apps/shot_web.py --port 8081

    # 浏览器打开 http://192.168.137.125:8081/
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

try:
    import cv2
    import numpy as np
    _CV_OK = True
except ImportError:
    _CV_OK = False

from grape_gripper.config import Config, default_config_path   # noqa: E402

GRAPE_HSV = ((120, 15, 15), (180, 255, 130))

#: 清晰度参考线（实测本机最好约 50；边缘分析希望更高）
FOCUS_GOOD = 80.0
FOCUS_MIN = 40.0

#: 温度上限（°C）—— 超过就拒绝拍照
TEMP_REFUSE = 70.0
TEMP_WARN = 65.0

#: 舵机闲置多久后自动放开（秒）
SERVO_IDLE_RELEASE = 180.0


# --------------------------------------------------------------------------
# 硬件封装（懒加载 + 空闲释放）
# --------------------------------------------------------------------------
class Hardware:
    def __init__(self):
        self.cfg = Config.load(default_config_path())
        self.cam = None
        self.servo = None
        self.reader = None
        self.raw_zero = None          # 空载基线（实测，不用配置常量）
        self.k = 1564.0               # counts/N，标定值
        self.pulse = None             # 当前脉宽
        self.last_use = 0.0
        # ★ 力值缓存：/status 只返回上次测的结果，**不主动测力**。
        #   浏览器每 2 秒轮询一次，如果每次都测 0.3 秒，
        #   就等于持续 15% CPU + 串口被一直占着。
        #   这正是 2026-11-05 那次事故的同一类错误（无意的持续负载）。
        self.force_cache = None
        self.lock = threading.Lock()
        try:
            self.k = self.cfg.get_float("sensor.calib.right.counts_per_n", 1564.0)
        except Exception:  # noqa: BLE001
            pass

    # ---- 温度 ----
    @staticmethod
    def temp_c() -> float | None:
        for z in ("/sys/class/thermal/thermal_zone0/temp",
                  "/sys/class/thermal/thermal_zone1/temp"):
            try:
                with open(z, "r") as fh:
                    return int(fh.read().strip()) / 1000.0
            except Exception:  # noqa: BLE001
                continue
        return None

    # ---- 相机 ----
    def get_cam(self):
        if self.cam is None:
            from k1_camera_capture import K1Camera
            self.cam = K1Camera(use_sudo=False)
        return self.cam

    # ---- 舵机 ----
    def get_servo(self):
        if self.servo is None:
            from grape_gripper.actuator import BusyPwmServo
            pin = self.cfg.get_int("servo.gpio", 49)
            pmin = self.cfg.get_float("servo.pulse_min_us", 750.0)
            pmax = self.cfg.get_float("servo.pulse_max_us", 1450.0)
            self.servo = BusyPwmServo(gpio_pin=pin, period_us=20000.0,
                                      pulse_min_us=pmin, pulse_max_us=pmax,
                                      angle_min=0.0, angle_max=180.0)
        return self.servo

    def pulse_limits(self):
        return (self.cfg.get_float("servo.pulse_min_us", 750.0),
                self.cfg.get_float("servo.pulse_max_us", 1450.0))

    def set_pulse(self, p: float) -> float:
        lo, hi = self.pulse_limits()
        p = max(lo, min(hi, float(p)))
        self.get_servo().set_pulse_us(p)
        self.pulse = p
        self.last_use = time.time()
        return p

    def release_servo(self):
        """张开并释放舵机（省 CPU 和热量）。"""
        try:
            if self.servo is not None:
                lo, hi = self.pulse_limits()
                self.servo.set_pulse_us(hi)
                time.sleep(0.3)
                self.servo.close()
        except Exception:  # noqa: BLE001
            pass
        finally:
            self.servo = None
            self.pulse = None

    # ---- 力 ----
    def get_reader(self):
        if self.reader is None:
            from grape_gripper.ps01 import Ps01Reader
            port = self.cfg.get_str("sensor.ps01.port", "/dev/ttyUSB0")
            r = Ps01Reader(port)
            r.open()
            if not r.wait_first_sample(timeout=3.0):
                raise RuntimeError(f"传感器无数据：{r.stats()}")
            self.reader = r
        return self.reader

    def force(self, window: float = 0.5):
        """
        测力：返回 (相对基线的力N, 原始中位数 counts, 样本数)。

        ⚠️ 用**中位数**，不能用均值 —— PS01 每约 2 秒冒一个
        +11,000 counts（约 +7 N）的尖峰，均值会被拉高 1.2 N。
        """
        r = self.get_reader()
        vals = []
        t0 = time.time()
        while time.time() - t0 < window:
            v = r.channel(1)
            if v is not None:
                vals.append(float(v))
            time.sleep(0.002)
        if len(vals) < 8:
            return None, None, len(vals)
        med = statistics.median(vals)
        if self.raw_zero is None:
            self.force_cache = 0.0
            return 0.0, med, len(vals)
        f = (med - self.raw_zero) / self.k
        self.force_cache = f
        return f, med, len(vals)

    def measure_baseline(self):
        """
        实测空载基线：张开爪子，测 1.2 秒取中位数。

        ⚠️ **不能用配置里的 sensor.calib.right.raw_zero（693,300）** ——
        那是传感器 B 的基线，而 channel(1) 读的是传感器 A（约 682,700），
        两者差 10,600 counts ≈ 6.8 N。之前踩过，张开状态显示 -6.8N。
        """
        self.set_pulse(self.pulse_limits()[1])
        time.sleep(1.0)
        r = self.get_reader()
        vals = []
        t0 = time.time()
        while time.time() - t0 < 1.2:
            v = r.channel(1)
            if v is not None:
                vals.append(float(v))
            time.sleep(0.002)
        if len(vals) < 10:
            raise RuntimeError("基线采样不足")
        self.raw_zero = statistics.median(vals)
        return self.raw_zero, len(vals)

    def find_contact(self, hi: float = 940.0, lo: float = 780.0,
                     step: float = 5.0, win: float = 0.4):
        """
        自动扫一遍找接触点。

        判据用**中位数的标准误差**，不是单样本 σ：
        单样本 σ 约 2.6 N，而力是几百个样本的中位数，
        其标准误差约 1.253·σ/√n ≈ 0.15 N。用单样本 σ 会把阈值定到 5 N，
        永远判不出接触（实测踩过）。
        """
        self.measure_baseline()
        rows = []
        p = hi
        while p >= lo - 1e-6:
            self.set_pulse(p)
            time.sleep(0.35)
            f, med, n = self.force(win)
            rows.append((p, f, n))
            p -= step
        return rows


HW = Hardware()


# --------------------------------------------------------------------------
# 图像分析
# --------------------------------------------------------------------------
def imwrite_unicode(path, img):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if ok:
        buf.tofile(path)


def color_masks(bgr, fast: bool = False, only: str | None = None):
    """
    分别返回三种颜色的掩码：黑 / 紫 / 绿。

    ⚠️ **绝对不要把它们合并成一张** —— 2026-11-06 实测踩过：

       合并（黑|紫|绿）：
           绿色像素 4.16%，但最大连通域变成 11%（1705x561），
           圆度 0.03 → 被筛掉 → 页面显示"没检出葡萄"

       只用绿色：
           最大连通域 2.27%（约 47,000 px），长宽比 0.98，
           圆度 0.73~0.84，质心正好落在葡萄上 → ✅ 完美

    原因：黑色传感器块 + 阴影 + 绿色/黄色导线（H≈25~35，
    和绿葡萄同色）在合并掩码里**桥接成一片**，
    把葡萄吞进一个巨大的非圆区域。

    分开之后每种颜色的连通关系独立，葡萄在"绿色"那张里是干净的圆块。
    调用方应**逐类尝试**，取最像葡萄的那个。

    fast 模式（给实时循环用）
    ------------------------
    形态学是这个函数最贵的部分。实测 1920x1080 下（单个掩码）：

        OPEN 11x11 椭圆     564 ms   ← 椭圆核不可分离，逐像素 121 次
        OPEN 11x11 矩形     102 ms   ← 可分离，快 5.5 倍
        CLOSE 5x5 椭圆 x2   233 ms

    耗时几乎完全由像素数决定（480x270 时只有 1/16）。

    fast=True 时：
        · 矩形核代替椭圆核（同样能切断约 10px 粗的导线）
        · 跳过 CLOSE —— 葡萄是实心的，不需要填洞；
          实测跳过它不影响三档判定
    配合调用方先降采样，整个检测可压到几十毫秒。
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0].astype(np.int16)
    S = hsv[:, :, 1].astype(np.int16)
    V = hsv[:, :, 2].astype(np.int16)

    masks = {}
    masks["black"] = ((V < 110) & (S < 110)).astype(np.uint8) * 255
    masks["purple"] = ((H >= 110) & (H <= 180) & (S >= 60) & (V >= 20)
                       & (V <= 170)).astype(np.uint8) * 255
    # 实测绿葡萄：H 中位 29（范围 16~31）、S 202、V 134
    masks["green"] = ((H >= 18) & (H <= 55) & (S >= 55) & (V >= 25)
                      & (V <= 250)).astype(np.uint8) * 255

    # ★ only：只算指定的那一类。两阶段检测的精测阶段已经知道
    #   葡萄是哪个颜色，没必要把另外两张掩码也算一遍（省 2/3 时间）。
    if only is not None:
        if only not in masks:
            return {}
        masks = {only: masks[only]}

    if fast:
        k11 = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        return {n: cv2.morphologyEx(m, cv2.MORPH_OPEN, k11)
                for n, m in masks.items()}

    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # ★ 开运算用 11：专门切断"导线"这类细连接（约 10px 粗）。
    #   葡萄直径约 230px，开运算对它几乎无损（实测面积 2.27% 不变）。
    k11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    out = {}
    for name, m in masks.items():
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k5, iterations=2)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k11)
        out[name] = m
    return out


def build_mask(bgr):
    """兼容旧接口，返回合并掩码（不推荐用于检测，原因见 color_masks）。"""
    ms = color_masks(bgr)
    m = ms["black"]
    for k in ("purple", "green"):
        m = cv2.bitwise_or(m, ms[k])
    return m


def analyze(bgr):
    """
    返回 6 个指标 + 判定。

    ⚠️ 两个判据的来历（都是 2026-11-05 被实测推翻后改的）：

    ① **不再用"是否靠画面中心"判断构图**
       我原来的规则是"葡萄质心偏离中心 ≤35%"，但实际布局是：
       相机看桌面、夹爪从左侧进、葡萄在桌面偏下 —— 葡萄**不可能在中心**。
       现在改成"**不贴画面边缘**（四边各留 2%）+ 尺寸合适"。

    ② **清晰度不再做硬性门槛**
       Laplacian 方差衡量的是"画面里有多少锐利边缘"，不是"对焦准不准"：
       满屏细密的场景能到 200+，而大片白桌面+几个物体只有 46 ——
       但这 46 完全可能是对焦准确的。所以这里只**报数**，
       真正的用法是在**同一场景下横向比较**（挪远挪近，看哪个更大）。
    """
    h, w = bgr.shape[:2]
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    focus = float(cv2.Laplacian(g, cv2.CV_64F).var())

    m = build_mask(bgr)

    # ★ 逐颜色类尝试，取最像葡萄的那个（**不要用合并掩码**，见 color_masks）
    best = None
    best_cls = None
    for cls_name, cm in color_masks(bgr).items():
        n, labels, stats, cents = cv2.connectedComponentsWithStats(cm, 8)
        for idx in range(1, n):
            a = int(stats[idx, cv2.CC_STAT_AREA])
            if a < w * h * 0.002:          # 太小
                continue
            if a > w * h * 0.35:           # 太大（大块阴影/墙面）
                continue
            bw = int(stats[idx, cv2.CC_STAT_WIDTH])
            bh = int(stats[idx, cv2.CC_STAT_HEIGHT])
            if bw <= 0 or bh <= 0:
                continue
            aspect = min(bw, bh) / max(bw, bh)
            if aspect < 0.62:              # 细长 → 导线
                continue
            blob_t = (labels == idx).astype(np.uint8)
            cnts, _ = cv2.findContours(blob_t * 255, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
            if not cnts:
                continue
            c = max(cnts, key=cv2.contourArea)
            perim = cv2.arcLength(c, True)
            if perim <= 0:
                continue
            circ = 4.0 * np.pi * a / (perim * perim)
            if circ < 0.60:                # 不够圆 → 方块/导线
                continue
            # 圆度权重更高：圆是葡萄最本质的特征
            score = a * circ * circ
            if best is None or score > best[1]:
                best = (idx, score, a, circ, labels, cents, cls_name)
                best_cls = cls_name

    out = {"w": w, "h": h, "focus": focus,
           "grape": None, "frac": None, "off": None,
           "solidity": None, "focus_grape": None,
           "cx": None, "cy": None, "r": None,
           "circ": None, "edge_w": None, "cls": None}

    if best is None:
        out["verdict"] = "没检出葡萄"
        out["advice"] = ("没找到圆形目标。检查：① 葡萄是否在画面内 "
                         "② 是否被夹爪挡住 ③ 是否太远（占比过小）")
        return out, None

    idx, _score, area, circ, labels, cents, cls_name = best
    out["cls"] = cls_name
    cx, cy = cents[idx]
    blob = (labels == idx).astype(np.uint8)
    out["grape"] = area
    out["frac"] = area / (w * h)
    out["cx"], out["cy"] = float(cx), float(cy)
    out["r"] = float(np.sqrt(area / np.pi))
    out["circ"] = float(circ)

    # ---- 构图：改成"不贴边"，不是"靠中心" ----
    #   实测布局：相机看桌面、夹爪从左侧进、葡萄在桌面偏下，
    #   葡萄永远不可能在画面中心，所以不能用偏心判据。
    ys, xs = np.nonzero(blob)
    m_edge = 0.02 * min(w, h)          # 四边各留 2%
    touch = []
    if xs.min() < m_edge:
        touch.append("左")
    if xs.max() > w - 1 - m_edge:
        touch.append("右")
    if ys.min() < m_edge:
        touch.append("上")
    if ys.max() > h - 1 - m_edge:
        touch.append("下")
    out["off"] = None                  # 不再用偏心率
    out["touch_edge"] = "、".join(touch) if touch else ""

    # ---- 轮廓实心度 ----
    cnts, _ = cv2.findContours(blob * 255, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        ca = cv2.contourArea(c)
        ha = cv2.contourArea(cv2.convexHull(c))
        out["solidity"] = ca / ha if ha > 0 else 0.0

    # ---- 边缘锐度：比 Laplacian 更直接 ----
    #   在葡萄边界附近看灰度梯度有多"陡"。
    #   对焦差时边缘会摊开成几像素宽，梯度峰值下降、过渡变宽。
    #   这里报"梯度最大值的对数尺度"，供**同一场景下横向比较**用。
    #   （绝对值依然依赖场景，不能当门槛。）
    try:
        ring = cv2.dilate(blob, np.ones((7, 7), np.uint8))
        ring = cv2.bitwise_and(ring, cv2.bitwise_not(blob))
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        if ring.any():
            out["edge_w"] = float(mag[ring > 0].mean())
    except Exception:  # noqa: BLE001
        pass

    # ---- 葡萄区域内的清晰度（同一场景横向比较用） ----
    if xs.size > 50:
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        crop = g[max(0, y0):y1, max(0, x0):x1]
        if crop.size:
            out["focus_grape"] = float(cv2.Laplacian(crop, cv2.CV_64F).var())

    # ---- 判定与建议 ----
    msgs = []
    # 清晰度不再做硬性门槛，只提示"这是相对指标"
    if out["focus_grape"] is not None and out["focus_grape"] < 15:
        msgs.append("葡萄区域几乎没细节（%.0f），可能太糊或太远"
                    % out["focus_grape"])
    if out["frac"] < 0.015:
        msgs.append("葡萄太小（%.2f%%），靠近一点" % (out["frac"] * 100))
    elif out["frac"] > 0.20:
        msgs.append("葡萄太大（%.2f%%），远一点" % (out["frac"] * 100))
    if touch:
        msgs.append("葡萄贴到画面边缘（%s），挪进画面内" % out["touch_edge"])
    if out["solidity"] is not None and out["solidity"] < 0.85:
        msgs.append("轮廓不完整（实心度 %.2f），被夹爪遮挡" % out["solidity"])
    if out["circ"] < 0.75:
        msgs.append("轮廓不够圆（圆度 %.2f），可能不是葡萄" % out["circ"])

    out["verdict"] = "合格" if not msgs else "需调整"
    out["advice"] = "；".join(msgs) if msgs else "可以拍"
    return out, blob


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>K1 拍照助手</title>
<style>
 body{margin:0;background:#161616;color:#ddd;font-family:monospace}
 .wrap{max-width:1180px;margin:0 auto;padding:14px}
 h2{margin:6px 0 12px;font-weight:normal;font-size:17px;color:#8cf}
 .bar{background:#222;border-radius:8px;padding:10px 14px;margin-bottom:10px}
 .row{display:flex;gap:18px;flex-wrap:wrap;align-items:center}
 button{font-size:15px;padding:9px 18px;cursor:pointer;border:0;
        border-radius:6px;background:#2a7;color:#fff}
 button:hover{background:#3b8}
 button.sec{background:#456;font-size:13px;padding:7px 12px}
 button.sec:hover{background:#567}
 button.big{font-size:20px;padding:14px 40px;background:#c60}
 button.big:hover{background:#e70}
 button:disabled{background:#444;color:#888;cursor:not-allowed}
 input{font-size:15px;width:90px;padding:7px;border-radius:5px;
       border:1px solid #444;background:#111;color:#eee}
 .flex{display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap}
 img{max-width:660px;border:1px solid #333;border-radius:4px}
 table{border-collapse:collapse;font-size:15px}
 td{padding:4px 10px;border-bottom:1px solid #2a2a2a}
 td.k{color:#9aa}
 .ok{color:#4d4}.bad{color:#f66}.warn{color:#fb4}
 #msg{color:#8cf;min-height:20px;margin-top:8px}
 .tags button{background:#357}
 .tags button:hover{background:#468}
</style></head><body><div class="wrap">
<h2>K1 拍照助手 &mdash; 点一下拍一张（相机只在点击时开启）</h2>

<div class="bar"><div class="row">
  <span>脉宽</span>
  <input id="pulse" type="number" value="940" min="750" max="1450">
  <button class="sec" onclick="setPulse()">设为该位置</button>
  <button class="sec" onclick="openJaw()">张开</button>
  <button class="sec" onclick="findContact()">扫一遍找接触点</button>
  <button class="sec" onclick="measure()">测力</button>
  <span id="fstat" style="color:#9aa">力：--</span>
  <span id="tstat" style="color:#9aa">温度：--</span>
</div></div>

<div class="bar"><div class="row">
  <button class="big" id="shoot" onclick="shoot()">📷 拍一张</button>
  <span class="tags row">
    <button class="sec" onclick="shoot('a')">存为「自由」</button>
    <button class="sec" onclick="shoot('b')">存为「贴住」</button>
    <button class="sec" onclick="shoot('c')">存为「夹紧」</button>
    <button class="sec" onclick="shoot('d')">存为「标记」</button>
  </span>
</div>
<div class="row" style="margin-top:8px">
  <span style="color:#9aa">连拍（每状态 10 张，约 12 秒）：</span>
  <button class="sec" onclick="burst('a')">连拍10张「自由」</button>
  <button class="sec" onclick="burst('b')">连拍10张「贴住」</button>
  <button class="sec" onclick="burst('c')">连拍10张「夹紧」</button>
</div><div id="msg"></div></div>

<div class="flex">
  <div><img id="pic" src="" alt="点「拍一张」看画面" width="660"></div>
  <div><table id="tab"><tr><td class="k">等待拍照…</td><td></td></tr></table></div>
</div>
</div>
<script>
let busy=false;
function fmt(v,d){return (v===null||v===undefined)?'--':Number(v).toFixed(d);}
function cls(v,good,bad){if(v===null)return '';return v>=good?'ok':(v<bad?'bad':'warn');}

async function poll(){
  try{
    const r=await fetch('/status'); const j=await r.json();
    document.getElementById('fstat').textContent =
      '力：'+(j.force===null?'--':(j.force>=0?'+':'')+j.force.toFixed(3)+'N')
      +'   基线：'+(j.baseline?Math.round(j.baseline):'--');
    const t=document.getElementById('tstat');
    t.textContent='温度：'+(j.temp===null?'--':j.temp.toFixed(0)+'°C')
      +(j.pulse?'   脉宽：'+Math.round(j.pulse)+'µs':'');
    t.className = j.temp>65?'bad':(j.temp>60?'warn':'');
  }catch(e){}
}
setInterval(poll,2000); poll();

async function shoot(tag){
  if(busy)return; busy=true;
  const b=document.getElementById('shoot'); b.disabled=true;
  document.getElementById('msg').textContent='拍照中（约 1.5 秒）…';
  try{
    const u='/shot'+(tag?('?tag='+tag):'');
    const r=await fetch(u); const j=await r.json();
    if(j.error){document.getElementById('msg').textContent='❌ '+j.error; return;}
    document.getElementById('pic').src='/img?t='+Date.now();
    const rows=[
      ['清晰度(全图)',fmt(j.focus,1),'','参考值：同一场景下挪远挪近比较，取最大'],
      ['清晰度(葡萄)',fmt(j.focus_grape,1),'','同上，越大事越好'],
      ['边缘梯度',fmt(j.edge_w,1),'','参考值：对焦越准，葡萄边界越陡'],
      ['葡萄占比',fmt(j.frac*100,2)+'%',
        (j.frac>=0.04&&j.frac<=0.15)?'ok':(j.frac<0.015?'bad':'warn'),'4~15% 可用'],
      ['圆度',fmt(j.circ,2),cls(j.circ,0.75,0.6),'≥0.75 才像葡萄'],
      ['实心度',fmt(j.solidity,2),cls(j.solidity,0.85,0.6),'≥0.85 未被遮挡'],
      ['贴边',(j.touch_edge?('贴'+j.touch_edge):'无'),
        (j.touch_edge?'bad':'ok'),'不贴边即可，**不要求居中**'],
    ];
    let h='';
    for(const [k,v,c,hint] of rows){
      h+='<tr><td class="k">'+k+'</td><td class="'+c+'">'+v+
         '</td><td class="k" style="font-size:12px">'+hint+'</td></tr>';
    }
    h+='<tr><td class="k">判定</td><td class="'+
       (j.verdict==='合格'?'ok':'warn')+'" colspan="2">'+j.verdict+'</td></tr>';
    h+='<tr><td class="k">建议</td><td colspan="2">'+j.advice+'</td></tr>';
    if(j.saved) h+='<tr><td class="k">已存</td><td colspan="2">'+j.saved+'</td></tr>';
    document.getElementById('tab').innerHTML=h;
    document.getElementById('msg').textContent='';
  }catch(e){
    document.getElementById('msg').textContent='❌ 请求失败：'+e;
  }finally{busy=false; b.disabled=false;}
}

async function setPulse(){
  const p=document.getElementById('pulse').value;
  document.getElementById('msg').textContent='设置 '+p+'µs …';
  const r=await fetch('/servo?pulse='+p); const j=await r.json();
  document.getElementById('msg').textContent = j.error?('❌ '+j.error)
    :('已设为 '+Math.round(j.pulse)+'µs');
  poll();
}
async function openJaw(){
  const r=await fetch('/servo?pulse=1450'); const j=await r.json();
  document.getElementById('msg').textContent='已张开';
  document.getElementById('pulse').value=1450; poll();
}
async function burst(tag){
  if(busy)return; busy=true;
  const b=document.getElementById('shoot'); b.disabled=true;
  const names={a:'自由',b:'贴住',c:'夹紧'};
  document.getElementById('msg').textContent =
    '连拍「'+names[tag]+'」10 张，约 12 秒，请不要移动葡萄…';
  try{
    const r=await fetch('/burst?tag='+tag+'&n=10'); const j=await r.json();
    if(j.error && !j.saved){
      document.getElementById('msg').textContent='❌ '+j.error; return;
    }
    let s='已存 '+j.saved+'/'+j.asked+' 张到 '+j.dir;
    if(j.focus_med) s+='　清晰度中位 '+j.focus_med.toFixed(1)+
      '（最低 '+j.focus_min.toFixed(1)+'）';
    if(j.frac_med) s+='　占比中位 '+(j.frac_med*100).toFixed(2)+'%';
    if(j.error) s+='　⚠️ '+j.error;
    document.getElementById('msg').textContent=s;
    document.getElementById('pic').src='/img?t='+Date.now();
  }catch(e){
    document.getElementById('msg').textContent='❌ 请求失败：'+e;
  }finally{busy=false; b.disabled=false;}
}

async function measure(){
  const r=await fetch('/measure'); const j=await r.json();
  document.getElementById('msg').textContent = j.error?('❌ '+j.error)
    :('力 '+(j.force>=0?'+':'')+j.force.toFixed(3)+'N  (中位数 '
      +Math.round(j.counts)+' counts, '+j.n+' 样本)');
  poll();
}
async function findContact(){
  document.getElementById('msg').textContent =
    '扫描中：940→780µs，共 33 档，约 25 秒，请等页面出结果…';
  // ⚠️ 上一版写"约 8 秒"，实际按 33 档 × (0.35+0.4) 秒 ≈ 25 秒，
  //    用户以为没反应就重复点了。这里改成准确时间。
  await new Promise(r=>setTimeout(r,150));
  const r=await fetch('/contact'); const j=await r.json();
  if(j.error){document.getElementById('msg').textContent='❌ '+j.error; return;}
  let s='基线 '+Math.round(j.baseline)+' counts。';
  if(j.contact){s+=' ★ 接触点约 '+Math.round(j.contact)+'µs'
    +'（建议：贴住 '+(Math.round(j.contact)+3)+'µs，夹紧 '
    +(Math.round(j.contact)-12)+'µs）';
    document.getElementById('pulse').value=Math.round(j.contact)+3;}
  else{s+=' 扫描范围内没检测到接触（葡萄没在两指之间？）';}
  document.getElementById('msg').textContent=s;
  poll();
}
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    last_jpeg = None

    def log_message(self, *a):
        pass

    def _json(self, obj):
        import json
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        qs = self.path.split("?", 1)[1] if "?" in self.path else ""

        if path in ("/", "/index.html"):
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/img":
            jpg = Handler.last_jpeg
            if not jpg:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpg)))
            self.end_headers()
            self.wfile.write(jpg)
            return

        if path == "/status":
            # ⚠️ 这里**只读温度文件、不碰任何硬件**。
            #    浏览器每 2 秒轮询；若每次测力 0.3 秒，
            #    就成了持续 15% CPU + 串口长期占用（同类错误）。
            #    力值用缓存，需要新读数时点「测力」按钮（/measure）。
            self._json({"temp": Hardware.temp_c(),
                        "force": HW.force_cache,
                        "baseline": HW.raw_zero,
                        "pulse": HW.pulse})
            return

        if path == "/measure":
            try:
                f, med, n = HW.force(0.6)
                self._json({"force": f, "counts": med, "n": n})
            except Exception as exc:  # noqa: BLE001
                self._json({"error": str(exc)})
            return

        if path == "/servo":
            try:
                p = float(qs.split("pulse=")[1].split("&")[0])
                p = HW.set_pulse(p)
                self._json({"pulse": p})
            except Exception as exc:  # noqa: BLE001
                self._json({"error": str(exc)})
            return

        if path == "/contact":
            try:
                rows = HW.find_contact()
                base = HW.raw_zero
                # 用中位数的标准误差判接触
                fs = [r[1] for r in rows if r[1] is not None]
                ns = [r[2] for r in rows if r[2]]
                if not fs:
                    self._json({"error": "扫描时读不到力"})
                    return
                n_typ = int(statistics.median(ns)) if ns else 100
                se = 1.253 * 2.6 / max(n_typ, 1) ** 0.5
                thr = 2 * se
                contact = None
                for p, f, _n in rows:
                    if f is not None and f > thr:
                        contact = p
                        break
                HW.set_pulse(HW.pulse_limits()[1])   # 扫完张开
                self._json({"baseline": base, "contact": contact,
                            "thr": thr, "rows": rows[:3]})
            except Exception as exc:  # noqa: BLE001
                self._json({"error": str(exc)})
            return

        if path == "/shot":
            t = Hardware.temp_c()
            if t is not None and t > TEMP_REFUSE:
                self._json({"error": "板子温度 %.0f°C 超过 %.0f°C，"
                                     "先散热再拍" % (t, TEMP_REFUSE)})
                return
            try:
                bgr = HW.get_cam().capture()
            except Exception as exc:  # noqa: BLE001
                self._json({"error": "拍照失败：%s" % exc})
                return
            try:
                info, _ = analyze(bgr)
            except Exception as exc:  # noqa: BLE001
                self._json({"error": "分析失败：%s" % exc})
                return

            ok, buf = cv2.imencode(".jpg", bgr,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            Handler.last_jpeg = buf.tobytes() if ok else None

            saved = ""
            tag = ""
            if "tag=" in qs:
                tag = qs.split("tag=")[1].split("&")[0]
                m = {"a": "set_a", "b": "set_b", "c": "set_c", "d": "set_d"}
                d = m.get(tag)
                if d:
                    outdir = os.path.join(_ROOT, "debug", d)
                    os.makedirs(outdir, exist_ok=True)
                    name = "%s_%s_f%.1f.jpg" % (
                        d, time.strftime("%H%M%S"), info["focus"])
                    p = os.path.join(outdir, name)
                    imwrite_unicode(p, bgr)
                    saved = "debug/%s/%s" % (d, name)
                    print("  📷 %s  focus=%.1f frac=%s" %
                          (saved, info["focus"],
                           ("%.2f%%" % (info["frac"] * 100))
                           if info["frac"] else "--"), flush=True)

            info["saved"] = saved
            self._json(info)
            return

        if path == "/burst":
            # 连拍 N 张（给"每个状态拍 10 张"用）
            #
            # 为什么需要：拍 31 张时每个状态要点 10 次，
            # 点一次挪一下手很容易让机位偏移，数据就不可比了。
            # 一次性拍完 10 张，位置只受舵机保持精度影响。
            #
            # 安全：N 上限 20（20×1.2s≈24s 相机开启，之后关闭）；
            #       每拍一张都查温度，超限立即中止并返回已拍数量。
            try:
                n = 10
                if "n=" in qs:
                    n = int(qs.split("n=")[1].split("&")[0])
                n = max(1, min(20, n))
                tag = qs.split("tag=")[1].split("&")[0] if "tag=" in qs else ""
                m = {"a": "set_a", "b": "set_b", "c": "set_c", "d": "set_d"}
                d = m.get(tag)
                if not d:
                    self._json({"error": "连拍必须指定 tag=a/b/c/d"})
                    return
                outdir = os.path.join(_ROOT, "debug", d)
                os.makedirs(outdir, exist_ok=True)

                saved_n = 0
                focuses = []
                fracs = []
                err = ""
                for i in range(n):
                    t = Hardware.temp_c()
                    if t is not None and t > TEMP_REFUSE:
                        err = "第 %d 张时温度 %.0f°C 超限，已中止" % (i + 1, t)
                        break
                    try:
                        bgr = HW.get_cam().capture()
                        info, _ = analyze(bgr)
                    except Exception as exc:  # noqa: BLE001
                        err = "第 %d 张失败：%s" % (i + 1, exc)
                        break
                    name = "%s_%02d_%s_f%.1f.jpg" % (
                        d, i + 1, time.strftime("%H%M%S"), info["focus"])
                    imwrite_unicode(os.path.join(outdir, name), bgr)
                    saved_n += 1
                    focuses.append(info["focus"])
                    if info["frac"]:
                        fracs.append(info["frac"])
                    print("  📷 [%d/%d] %s/%s  focus=%.1f  frac=%s" %
                          (i + 1, n, d, name, info["focus"],
                           ("%.2f%%" % (info["frac"] * 100))
                           if info["frac"] else "--"), flush=True)

                self._json({
                    "dir": "debug/%s" % d, "saved": saved_n, "asked": n,
                    "focus_med": (statistics.median(focuses)
                                  if focuses else None),
                    "focus_min": (min(focuses) if focuses else None),
                    "frac_med": (statistics.median(fracs)
                                 if fracs else None),
                    "error": err,
                })
            except Exception as exc:  # noqa: BLE001
                self._json({"error": str(exc)})
            return

        self.send_error(404)


def main() -> int:
    ap = argparse.ArgumentParser(description="拍照助手（点击式，安全）")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--max-minutes", type=float, default=20.0,
                    help="总运行多少分钟后自动退出（默认 20）")
    args = ap.parse_args()

    if not _CV_OK:
        print("❌ 需要 cv2")
        return 1

    print("=" * 70)
    print("K1 拍照助手")
    print("=" * 70)
    print("  相机：只在点「拍一张」时开启（约 1.2 秒），其余时间关闭")
    print("  舵机：保持位置约 4% CPU；闲置 180 秒自动张开并释放")
    print("  安全：总运行 %d 分钟后自动退出；温度 >%.0f°C 拒绝拍照" %
          (int(args.max_minutes), TEMP_REFUSE))
    print()
    print("  ⚠️ 先停服务，否则会抢串口：")
    print("       sudo systemctl stop grape-gripper")
    print()

    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print("=" * 70)
    print("  浏览器打开：  http://192.168.137.125:%d/" % args.port)
    print("=" * 70)

    stop_at = time.time() + args.max_minutes * 60

    def _idle_watch():
        while time.time() < stop_at:
            time.sleep(5)
            if (HW.servo is not None and HW.last_use > 0
                    and time.time() - HW.last_use > SERVO_IDLE_RELEASE):
                print("  舵机闲置 180 秒，自动张开并释放", flush=True)
                HW.release_servo()

    def _timeout():
        while time.time() < stop_at:
            time.sleep(5)
        print("\n  到达 %d 分钟上限，自动退出" % int(args.max_minutes),
              flush=True)
        HW.release_servo()
        os._exit(0)

    threading.Thread(target=_idle_watch, daemon=True).start()
    threading.Thread(target=_timeout, daemon=True).start()

    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n退出")
    finally:
        srv.shutdown()
        HW.release_servo()
        try:
            if HW.reader is not None:
                HW.reader.close()
        except Exception:  # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())