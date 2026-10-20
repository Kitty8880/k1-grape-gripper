#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
葡萄识别：从摄像头取一帧，找出葡萄的位置和大小。

策略：HSV 颜色分割 + 轮廓分析（快、无需模型、可解释）
      —— 适合比赛演示：能直接说清"为什么判定这是葡萄"。

用法
----
抓一帧并识别（交互模式摄像头）::

    sudo python3 grape_detect.py --save-raw /tmp/raw.jpg --save-ann /tmp/ann.jpg

识别已有的图片::

    python3 grape_detect.py --image /tmp/frame_0010.jpg --save-ann /tmp/ann.jpg

连续识别 20 帧看效果::

    sudo python3 grape_detect.py --frames 20 --save-ann /tmp/ann.jpg
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

try:
    import cv2
except ImportError:
    print("需要 opencv：sudo apt install -y python3-opencv")
    sys.exit(1)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass


# HSV 阈值（OpenCV 的 H 范围是 0~179）
COLOR_RANGES = {
    "purple": [
        # 紫葡萄：色相在 125~170，饱和度较高
        ((125, 40, 40), (170, 255, 255)),
    ],
    "green": [
        # 绿葡萄：色相在 35~90
        ((35, 40, 40), (90, 255, 255)),
    ],
    "red": [
        ((0, 60, 50), (12, 255, 255)),
        ((168, 60, 50), (179, 255, 255)),
    ],
    "any": [
        ((0, 40, 40), (179, 255, 255)),   # 兜底：任何有颜色的区域
    ],
}


def gray_world_awb(bgr: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    灰世界白平衡校正。

    K1 的 ISP 没有调参文件，自动白平衡会漂移（整张画面偏粉/偏蓝），
    导致基于 HSV 的颜色检测完全不可靠。

    原理：假设画面整体平均颜色应该是灰的，据此把三个通道拉到同一水平。
    strength=0 不校正，1.0 完全校正。
    """
    if strength <= 0:
        return bgr

    f = bgr.astype(np.float32)
    means = f.reshape(-1, 3).mean(axis=0)          # B, G, R
    gray = float(means.mean())
    if gray <= 1.0:
        return bgr

    for i in range(3):
        m = max(float(means[i]), 1.0)
        gain = gray / m
        gain = 1.0 + (gain - 1.0) * strength       # 按强度插值
        f[:, :, i] *= gain

    return np.clip(f, 0, 255).astype(np.uint8)


class Detection:
    """一个检测结果。"""

    def __init__(self, x: float, y: float, radius: float, area: float,
                 circularity: float, contour):
        self.x = x
        self.y = y
        self.radius = radius
        self.area = area
        self.circularity = circularity
        self.contour = contour

    def __repr__(self) -> str:
        return (f"Detection(center=({self.x:.0f},{self.y:.0f}), "
                f"r={self.radius:.0f}, area={self.area:.0f}, "
                f"circ={self.circularity:.2f})")


class GrapeDetector:
    """基于 HSV + 轮廓的葡萄检测器。"""

    def __init__(self,
                 color: str = "purple",
                 min_area: float = 1500,
                 max_area: float = 400000,
                 min_circularity: float = 0.60,
                 min_solidity: float = 0.72,
                 open_kernel: int = 5,
                 close_kernel: int = 11,
                 scale: float = 0.5,
                 awb: float = 1.0,
                 max_contours: int = 20):
        """
        scale: 检测前把图缩小的比例（0.5 = 缩到一半边长，面积降到 1/4）。
               1080p 全分辨率在 K1 上太慢（700+ ms），缩小后约 90 ms。

        判定"像葡萄"用三个指标（都算在**凸包**上，这样物体表面有花纹、
        反光、遮挡时也不会被误杀）：
          - circularity: 凸包圆度，越接近圆形越接近 1.0
          - solidity:    实际面积 / 凸包面积，实心球体接近 1.0
          - area:        面积落在合理区间
        """
        self.color = color
        self.ranges = COLOR_RANGES.get(color, COLOR_RANGES["purple"])
        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity
        self.min_solidity = min_solidity
        self.scale = scale
        self.awb = awb
        self.max_contours = max_contours
        self.k_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        self.k_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))

    def _resize(self, bgr: np.ndarray) -> tuple[np.ndarray, float]:
        if self.scale == 1.0:
            small = bgr
        else:
            small = cv2.resize(bgr, None, fx=self.scale, fy=self.scale,
                               interpolation=cv2.INTER_AREA)
        if self.awb > 0:
            small = gray_world_awb(small, self.awb)
        return small, 1.0 / self.scale

    def _mask_of_small(self, small: np.ndarray) -> np.ndarray:
        """在**已经缩放好**的图上生成颜色掩码。"""
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in self.ranges:
            mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))

        # 先小核开运算去噪点，再用大核闭运算把物体表面的花纹/反光"填平"
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.k_open, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.k_close, iterations=2)
        return mask

    def mask_of(self, bgr: np.ndarray) -> np.ndarray:
        """生成颜色掩码（传入原图，内部会缩放）。"""
        small, _ = self._resize(bgr)
        return self._mask_of_small(small)

    def detect(self, bgr: np.ndarray) -> list[Detection]:
        """返回按面积从大到小排序的检测结果（坐标已换算回原图）。"""
        small, inv = self._resize(bgr)
        mask = self._mask_of_small(small)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # 只处理最大的若干个，避免大面积噪声轮廓拖慢速度
        if self.max_contours and len(contours) > self.max_contours:
            contours = sorted(contours, key=cv2.contourArea,
                              reverse=True)[:self.max_contours]

        results: list[Detection] = []
        for c in contours:
            area_small = cv2.contourArea(c)
            area = area_small * inv * inv          # 换算回原图面积
            if area < self.min_area or area > self.max_area:
                continue

            # 用凸包评估"像不像一个球"
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area <= 0:
                continue
            solidity = area_small / hull_area

            peri = cv2.arcLength(hull, True)
            if peri <= 0:
                continue
            circ = 4.0 * np.pi * hull_area / (peri * peri)

            if circ < self.min_circularity or solidity < self.min_solidity:
                continue

            (x, y), r = cv2.minEnclosingCircle(hull)
            contour_full = (c.astype(np.float32) * inv).astype(np.int32)
            results.append(Detection(x * inv, y * inv, r * inv,
                                     area, circ, contour_full))

        results.sort(key=lambda d: -d.area)
        return results

    def annotate(self, bgr: np.ndarray,
                 detections: list[Detection]) -> np.ndarray:
        """画检测结果。最大的那个用实心圈标出（作为抓取目标）。"""
        vis = bgr.copy()
        for i, d in enumerate(detections):
            is_target = (i == 0)
            color = (0, 255, 0) if is_target else (0, 165, 255)
            thick = 4 if is_target else 2

            cv2.drawContours(vis, [d.contour], -1, color, thick)
            cv2.circle(vis, (int(d.x), int(d.y)), int(d.radius), color, thick)

            # 中心十字
            cx, cy = int(d.x), int(d.y)
            cv2.line(vis, (cx - 15, cy), (cx + 15, cy), (0, 0, 255), 2)
            cv2.line(vis, (cx, cy - 15), (cx, cy + 15), (0, 0, 255), 2)

            label = (f"#{i+1} ({cx},{cy}) r={int(d.radius)}"
                     if not is_target else
                     f"TARGET ({cx},{cy}) r={int(d.radius)}")
            ty = max(30, cy - int(d.radius) - 12)
            cv2.putText(vis, label, (max(10, cx - 160), ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (255, 255, 255), 6, cv2.LINE_AA)
            cv2.putText(vis, label, (max(10, cx - 160), ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 0, 0), 2, cv2.LINE_AA)

        # 左上角信息
        info = f"{self.color}  found={len(detections)}"
        cv2.putText(vis, info, (16, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, (255, 255, 255), 7, cv2.LINE_AA)
        cv2.putText(vis, info, (16, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, (0, 0, 0), 2, cv2.LINE_AA)
        return vis


def grab_one_frame(args) -> np.ndarray:
    """从交互模式摄像头取一帧。"""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from k1_camera_stream import K1CameraStream

    with K1CameraStream(verbose=False) as cam:
        time.sleep(0.3)
        return cam.read_frame()


def main() -> int:
    ap = argparse.ArgumentParser(description="葡萄识别")
    ap.add_argument("--image", default=None, help="直接识别指定图片")
    ap.add_argument("--frames", type=int, default=1,
                    help="从摄像头连续识别多少帧")
    ap.add_argument("--color", default="purple",
                    choices=list(COLOR_RANGES.keys()))
    ap.add_argument("--min-area", type=float, default=1500)
    ap.add_argument("--max-area", type=float, default=400000)
    ap.add_argument("--min-circularity", type=float, default=0.60,
                    help="凸包圆度下限。圆球接近 1.0，墙面/阴影等色块通常 < 0.5")
    ap.add_argument("--min-solidity", type=float, default=0.72,
                    help="实心度下限 = 面积/凸包面积。实心球体接近 1.0")
    ap.add_argument("--close-kernel", type=int, default=11,
                    help="闭运算核大小，越大越能把物体表面的花纹填平")
    ap.add_argument("--scale", type=float, default=0.5,
                    help="检测前缩放比例，0.5 表示缩到一半（更快）")
    ap.add_argument("--awb", type=float, default=1.0,
                    help="灰世界白平衡校正强度，0=关闭，1=完全校正")
    ap.add_argument("--save-raw", default=None)
    ap.add_argument("--save-ann", default=None)
    ap.add_argument("--save-mask", default=None)
    args = ap.parse_args()

    det = GrapeDetector(color=args.color,
                        min_area=args.min_area,
                        max_area=args.max_area,
                        min_circularity=args.min_circularity,
                        min_solidity=args.min_solidity,
                        close_kernel=args.close_kernel,
                        scale=args.scale,
                        awb=args.awb)

    print("=" * 66)
    print(f"葡萄识别  颜色={args.color}")
    print("=" * 66)

    # ---------- 单张图片模式 ----------
    if args.image:
        bgr = cv2.imread(args.image)
        if bgr is None:
            print(f"❌ 读不到图片 {args.image}")
            return 1
        print(f"读入 {args.image}  {bgr.shape[1]}x{bgr.shape[0]}")

        t0 = time.time()
        dets = det.detect(bgr)
        dt = (time.time() - t0) * 1000

        print(f"检测到 {len(dets)} 个候选，耗时 {dt:.1f} ms")
        for i, d in enumerate(dets[:8]):
            tag = " ← 抓取目标" if i == 0 else ""
            print(f"  #{i+1} 中心=({d.x:.0f},{d.y:.0f})  半径={d.radius:.0f}  "
                  f"面积={d.area:.0f}  圆度={d.circularity:.2f}{tag}")

        if args.save_ann:
            cv2.imwrite(args.save_ann, det.annotate(bgr, dets))
            print(f"标注图已保存 {args.save_ann}")
        if args.save_mask:
            cv2.imwrite(args.save_mask, det.mask_of(bgr))
            print(f"掩码已保存 {args.save_mask}")
        return 0

    # ---------- 摄像头模式 ----------
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from k1_camera_stream import K1CameraStream

    try:
        cam = K1CameraStream()
        cam.open()
    except Exception as exc:  # noqa: BLE001
        print(f"❌ 摄像头启动失败：{exc}")
        return 1

    print(f"✅ pipeline 已启动，识别 {args.frames} 帧\n")
    hits = 0
    times = []
    try:
        for i in range(args.frames):
            frame = cam.read_frame()
            t0 = time.time()
            dets = det.detect(frame)
            times.append((time.time() - t0) * 1000)
            if dets:
                hits += 1

            if dets:
                d = dets[0]
                print(f"[{i+1:>3}] 找到 {len(dets)} 个  "
                      f"目标=({d.x:.0f},{d.y:.0f}) r={d.radius:.0f} "
                      f"圆度={d.circularity:.2f}")
            else:
                print(f"[{i+1:>3}] 未找到")

            if args.save_raw and i == 0:
                cv2.imwrite(args.save_raw, frame)
                print(f"       原图已存 {args.save_raw}")
            if args.save_ann and (i == 0 or i == args.frames - 1):
                out = args.save_ann if i == 0 else args.save_ann.replace(
                    ".jpg", "_last.jpg")
                cv2.imwrite(out, det.annotate(frame, dets))
                print(f"       标注图已存 {out}")
            if args.save_mask and i == 0:
                cv2.imwrite(args.save_mask, det.mask_of(frame))
                print(f"       掩码已存 {args.save_mask}")
    except KeyboardInterrupt:
        print("\n中断")
    finally:
        cam.close()

    print()
    print("=" * 66)
    print(f"{hits}/{args.frames} 帧找到葡萄")
    if times:
        print(f"识别耗时 平均 {np.mean(times):.1f} ms/帧")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())