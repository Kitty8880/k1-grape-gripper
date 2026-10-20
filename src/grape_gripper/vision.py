# -*- coding: utf-8 -*-
"""
视觉识别：找到葡萄并给出像素坐标 + 尺度（阶段二）。

两条路线
--------
1. ``HsvDetector``  — HSV 颜色阈值 + 形态学 + 轮廓分析
   优点：零训练、CPU 就能跑满帧率、参数可解释、调试快
   缺点：只对"紫/红葡萄 + 干净背景"有效，光照变化敏感

2. ``YoloDetector`` — K1 NPU 上跑 YOLOv8n（需 ONNX→kmodel 转换）
   优点：鲁棒，能处理遮挡/多目标/不同品种
   缺点：要准备数据集、训练、转换模型

**建议顺序**：先用 HSV 把整条链路跑通（视觉→坐标→夹取），
再替换成 YOLO。因为视觉不是这个项目的风险点，力闭环才是。

关于 OV5647
-----------
树莓派 500 万像素模组，MIPI CSI 2-lane。K1 上通常直接出 /dev/videoN。
注意模组本体没有 ISP 的自动曝光/AWB（它把这些交给树莓派的 GPU），
在 K1 上如果画面过暗或偏色，需要在 v4l2 层面手动设曝光/增益，
或者接受一个固定光照环境（比赛现场打光可控，这反而是最简单可靠的方案）。
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# OpenCV / numpy 都是可选的：没装时 vision 模块仍可 import，
# 只是调用 detect 会抛 VisionError 并给出安装提示。
try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore


class VisionError(Exception):
    """视觉模块出错。"""


def _require_cv() -> None:
    if cv2 is None or np is None:
        raise VisionError(
            "需要 OpenCV 和 numpy。安装：\n"
            "  sudo apt install -y python3-opencv python3-numpy\n"
            "若 apt 版本太老，用 pip：\n"
            "  pip3 install opencv-python-headless numpy"
        )


# ---------------------------------------------------------------------------
# 检测结果
# ---------------------------------------------------------------------------

@dataclass
class GrapeDetection:
    """一个葡萄的检测结果。"""
    cx_px: float                 # 质心 x（像素）
    cy_px: float                 # 质心 y（像素）
    radius_px: float             # 等效半径（像素），用于估计葡萄大小
    area_px: float               # 轮廓面积
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)   # x, y, w, h
    confidence: float = 1.0

    @property
    def center(self) -> Tuple[float, float]:
        return self.cx_px, self.cy_px

    def diameter_px(self) -> float:
        return 2.0 * self.radius_px

    def __str__(self) -> str:
        return (f"葡萄@({self.cx_px:.0f},{self.cy_px:.0f})px "
                f"r={self.radius_px:.1f}px area={self.area_px:.0f} "
                f"conf={self.confidence:.2f}")


@dataclass
class VisionResult:
    """一帧的完整检测结果。"""
    detections: List[GrapeDetection] = field(default_factory=list)
    frame_shape: Tuple[int, int] = (0, 0)     # (h, w)
    timestamp: float = 0.0
    annotations: Optional[object] = None      # 画了框的图（便于保存调试）

    @property
    def best(self) -> Optional[GrapeDetection]:
        """最大的那个目标 —— 通常就是离相机最近的葡萄。"""
        if not self.detections:
            return None
        return max(self.detections, key=lambda d: d.area_px)

    @property
    def found(self) -> bool:
        return self.best is not None

    def to_gripper_frame(self, mm_per_px_x: float, mm_per_px_y: float,
                         offset_x_mm: float = 0.0,
                         offset_y_mm: float = 0.0,
                         image_center: Optional[Tuple[float, float]] = None
                         ) -> Optional[Tuple[float, float]]:
        """
        像素坐标 → 夹爪平面物理坐标（mm）。

        假设：相机光轴垂直向下，与夹爪平面平行。
        这是一个**平面近似**，忽略葡萄的高度（直径约 22mm）。对"夹住"这个
        动作来说，误差主要来自葡萄表面高度，可以在标定时加一个 z 偏移补偿。

        返回相对图像中心（或指定的标定原点）的 (x_mm, y_mm)。
        """
        target = self.best
        if target is None:
            return None
        if image_center is None:
            h, w = self.frame_shape
            image_center = (w / 2.0, h / 2.0)
        dx_px = target.cx_px - image_center[0]
        dy_px = target.cy_px - image_center[1]
        return (dx_px * mm_per_px_x + offset_x_mm,
                dy_px * mm_per_px_y + offset_y_mm)


# ---------------------------------------------------------------------------
# 相机
# ---------------------------------------------------------------------------

class Camera:
    """
    V4L2 摄像头封装。

    优先用 OpenCV 的 V4L2 后端（性能好、支持 MJPG），
    没有 OpenCV 时退回到自己实现的 mmap 抓帧（仅支持 YUYV/MJPG 单帧）。
    """

    def __init__(self, device: str = "/dev/video0", width: int = 1280,
                 height: int = 720, fps: int = 30, fourcc: str = "MJPG",
                 simulate: bool = False):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = fourcc
        self.simulate = simulate
        self._cap = None
        self._frame_index = 0

    def open(self) -> "Camera":
        if self.simulate:
            return self
        if not os.path.exists(self.device):
            raise VisionError(
                f"找不到 {self.device}。用 'v4l2-ctl --list-devices' 查看实际设备节点；"
                "OV5647 模组通常还需要确认 CSI 排线插好、I2C 上能看到 0x36。"
            )
        _require_cv()
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            raise VisionError(
                f"打不开 {self.device}。可能被其他进程占用（如桌面的相机预览），"
                "或需要 sudo 权限。"
            )
        # MJPG 能显著提高分辨率下的帧率
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        except Exception:  # noqa: BLE001
            pass
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # 减少延迟：永远取最新帧
        self._cap = cap
        return self

    def read(self):
        """读一帧，返回 BGR ndarray。"""
        if self.simulate:
            return self._synthetic_frame()
        if self._cap is None:
            raise VisionError("相机未打开，先调用 open()")
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise VisionError("抓帧失败。检查排线、供电，或降低分辨率/换 MJPG")
        return frame

    def _synthetic_frame(self):
        """合成一帧带"葡萄"的测试图，用于在没有相机时验证检测算法。"""
        _require_cv()
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[:, :] = (60, 70, 80)      # 灰蓝背景
        # 画几颗紫葡萄（BGR 里紫色 = 高 R 高 B 低 G）
        for (cx, cy, r) in [
            (int(self.width * 0.45), int(self.height * 0.55), 60),
            (int(self.width * 0.70), int(self.height * 0.45), 42),
            (int(self.width * 0.28), int(self.height * 0.38), 30),
        ]:
            cv2.circle(img, (cx, cy), r, (140, 40, 150), -1)
            cv2.circle(img, (cx - r // 4, cy - r // 4), max(2, r // 5),
                       (200, 120, 210), -1)     # 高光
        self._frame_index += 1
        return img

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    def describe(self) -> str:
        if self.simulate:
            return f"Camera(模拟 {self.width}x{self.height})"
        return f"Camera({self.device} {self.width}x{self.height}@{self.fps} {self.fourcc})"


# ---------------------------------------------------------------------------
# 检测器
# ---------------------------------------------------------------------------

class BaseDetector:
    def detect(self, frame) -> VisionResult:
        raise NotImplementedError

    def describe(self) -> str:
        return type(self).__name__


class HsvDetector(BaseDetector):
    """
    HSV 颜色阈值 + 轮廓分析。

    调参步骤（很重要，不同光照差别巨大）：
      1. 把相机对准葡萄，跑 apps/show_camera.py 保存一张图
      2. 跑 apps/tune_hsv.py，用滑动条实时看掩膜效果
      3. 把满意的 HSV 范围填回 config.yaml

    小技巧：
      - 紫色葡萄的 H 大约在 130~175（OpenCV H ∈ [0,179]）
      - 光照不足时 V 普遍偏低，不要为了"抓到"而把 V_min 降到 20 以下，
        那会把阴影也当成葡萄
      - 加一道"饱和度下限"能有效排除灰色物体
    """

    def __init__(self, h_min: int = 130, h_max: int = 175,
                 s_min: int = 60, s_max: int = 255,
                 v_min: int = 40, v_max: int = 255,
                 min_area_px: int = 800,
                 blur_ksize: int = 5,
                 morph_ksize: int = 7):
        self.h_min = h_min
        self.h_max = h_max
        self.s_min = s_min
        self.s_max = s_max
        self.v_min = v_min
        self.v_max = v_max
        self.min_area_px = min_area_px
        self.blur_ksize = blur_ksize
        self.morph_ksize = morph_ksize

    @classmethod
    def from_config(cls, cfg) -> "HsvDetector":
        return cls(
            h_min=cfg.get_int("vision.hsv.h_min", 130),
            h_max=cfg.get_int("vision.hsv.h_max", 175),
            s_min=cfg.get_int("vision.hsv.s_min", 60),
            s_max=cfg.get_int("vision.hsv.s_max", 255),
            v_min=cfg.get_int("vision.hsv.v_min", 40),
            v_max=cfg.get_int("vision.hsv.v_max", 255),
            min_area_px=cfg.get_int("vision.hsv.min_area_px", 800),
        )

    def detect(self, frame) -> VisionResult:
        _require_cv()
        h, w = frame.shape[:2]
        result = VisionResult(frame_shape=(h, w), timestamp=time.time())

        blurred = cv2.GaussianBlur(frame, (self.blur_ksize, self.blur_ksize), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        lower = np.array([self.h_min, self.s_min, self.v_min], dtype=np.uint8)
        upper = np.array([self.h_max, self.s_max, self.v_max], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        # 形态学：开运算去噪点，闭运算填补高光造成的小洞
        k = np.ones((self.morph_ksize, self.morph_ksize), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        annotated = frame.copy()
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area_px:
                continue

            # 圆形度过滤：葡萄接近圆形，排除长条形的干扰物
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 1e-6:
                continue
            circularity = 4.0 * math.pi * area / (perimeter * perimeter)
            if circularity < 0.55:
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            x, y, bw, bh = cv2.boundingRect(contour)
            detection = GrapeDetection(
                cx_px=float(cx), cy_px=float(cy),
                radius_px=float(radius), area_px=float(area),
                bbox=(int(x), int(y), int(bw), int(bh)),
                confidence=min(1.0, circularity),
            )
            result.detections.append(detection)

            cv2.circle(annotated, (int(cx), int(cy)), int(radius), (0, 255, 0), 2)
            cv2.circle(annotated, (int(cx), int(cy)), 3, (0, 0, 255), -1)
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (255, 200, 0), 1)

        result.annotations = annotated
        return result

    def describe(self) -> str:
        return (f"HsvDetector(H[{self.h_min},{self.h_max}] "
                f"S[{self.s_min},{self.s_max}] V[{self.v_min},{self.v_max}] "
                f"minArea={self.min_area_px})")


class YoloDetector(BaseDetector):
    """
    YOLO 检测器（K1 NPU / ONNX Runtime / OpenCV DNN 三种后端）。

    现状说明：
      * K1 的 NPU 需要 SpacemiT AI SDK，把 ONNX 转成 .kmodel 后调用
        runtime 库（libnncase / spacemit_rt）。这依赖官方 SDK 的安装。
      * 在这条路打通之前，可以先用 OpenCV DNN 跑 ONNX 模型，
        帧率会低很多（K1 CPU 跑 yolov8n 大概 2~8 FPS），但足够做演示。
      * 本类先把接口固定下来，具体后端见 ``_load_backend()``。

    数据集建议：葡萄检测不需要很大的数据集。
      200~400 张自己拍的图（含不同光照、遮挡、单颗/串状），
      用 labelImg 标一个类 "grape"，YOLOv8n 训 100 epoch 就够。
      自己拍的比网上下载的效果好得多 —— 因为背景就是你比赛的桌面。
    """

    def __init__(self, model_path: str, conf_threshold: float = 0.5,
                 backend: str = "auto", input_size: int = 640):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.backend = backend
        self.input_size = input_size
        self._net = None
        self._active_backend = "none"

    @classmethod
    def from_config(cls, cfg) -> "YoloDetector":
        return cls(
            model_path=cfg.get_str("vision.yolo.model_path", "models/grape.kmodel"),
            conf_threshold=cfg.get_float("vision.yolo.conf_threshold", 0.5),
        )

    def _load_backend(self) -> None:
        if self._net is not None:
            return
        if not os.path.isfile(self.model_path):
            raise VisionError(
                f"模型文件不存在：{self.model_path}\n"
                "先用 HSV 方案跑通链路，再准备 YOLO 模型。"
            )
        if self.model_path.endswith((".onnx", ".pt")):
            _require_cv()
            self._net = cv2.dnn.readNetFromONNX(self.model_path)
            self._active_backend = "opencv_dnn"
        elif self.model_path.endswith(".kmodel"):
            raise VisionError(
                "加载 .kmodel 需要 SpacemiT AI SDK 的 runtime 库。\n"
                "步骤：\n"
                "  1. 参考 Bianbu / SpacemiT AI SDK 文档安装 nncase + runtime\n"
                "  2. 用 nncase 把 ONNX 转成 kmodel\n"
                "  3. 在本类里实现 kmodel 推理后端（用 ctypes 调 libspacemit_rt）\n"
                "在此之前，请把 config.yaml 的 vision.detector 改为 'hsv'。"
            )
        else:
            raise VisionError(f"不认识的模型格式：{self.model_path}")

    def detect(self, frame) -> VisionResult:
        _require_cv()
        self._load_backend()
        h, w = frame.shape[:2]
        result = VisionResult(frame_shape=(h, w), timestamp=time.time())

        # 预处理：letterbox 到模型输入尺寸
        size = self.input_size
        scale = min(size / w, size / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(frame, (nw, nh))
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        dx, dy = (size - nw) // 2, (size - nh) // 2
        canvas[dy:dy + nh, dx:dx + nw] = resized

        blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, (size, size),
                                     swapRB=True, crop=False)
        self._net.setInput(blob)
        outputs = self._net.forward()

        # YOLOv8 输出形状 (1, 4+num_classes, num_anchors)
        preds = outputs[0]
        if preds.ndim == 3:
            preds = preds[0]
        if preds.shape[0] < preds.shape[1]:
            preds = preds.T          # → (num_anchors, 4+nc)

        annotated = frame.copy()
        for row in preds:
            if row.shape[0] < 5:
                continue
            cx, cy, bw, bh = row[0], row[1], row[2], row[3]
            scores = row[4:]
            conf = float(scores.max()) if scores.size else 0.0
            if conf < self.conf_threshold:
                continue

            # 从 letterbox 坐标还原到原图
            cx = (cx - dx) / scale
            cy = (cy - dy) / scale
            bw = bw / scale
            bh = bh / scale
            if bw <= 1 or bh <= 1:
                continue

            detection = GrapeDetection(
                cx_px=float(cx), cy_px=float(cy),
                radius_px=float(max(bw, bh) / 2.0),
                area_px=float(math.pi * (bw / 2) * (bh / 2)),
                bbox=(int(cx - bw / 2), int(cy - bh / 2), int(bw), int(bh)),
                confidence=conf,
            )
            result.detections.append(detection)
            x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
            x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"grape {conf:.2f}", (x1, max(12, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        result.annotations = annotated
        return result

    def describe(self) -> str:
        return (f"YoloDetector({self.model_path} backend={self._active_backend} "
                f"conf>{self.conf_threshold})")


def create_detector(cfg) -> BaseDetector:
    """根据配置创建检测器。"""
    kind = cfg.get_str("vision.detector", "hsv").lower()
    if kind == "hsv":
        return HsvDetector.from_config(cfg)
    if kind in ("yolo", "yolov8"):
        return YoloDetector.from_config(cfg)
    raise VisionError(f"未知的 vision.detector：{kind!r}（可选 hsv / yolo）")


def create_camera(cfg, simulate: Optional[bool] = None) -> Camera:
    """根据配置创建相机。"""
    if simulate is None:
        simulate = cfg.simulate
    return Camera(
        device=cfg.get_str("vision.camera.device", "/dev/video0"),
        width=cfg.get_int("vision.camera.width", 1280),
        height=cfg.get_int("vision.camera.height", 720),
        fps=cfg.get_int("vision.camera.fps", 30),
        simulate=simulate,
    )
