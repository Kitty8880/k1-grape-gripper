#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
物体分类推理 —— 加载 ONNX 分类模型，输出类别和置信度。

为什么要有这个（和 HSV 检测的区别）
-----------------------------------
HSV 只认"颜色 + 形状"，分不清"葡萄"和"手指甲"—— 2026-11-02 实测就出现过
把手指甲认成葡萄（置信度 0.81），因为手指甲又小又圆。

分类器学的是"这个东西整体长什么样"，能利用纹理、明暗、上下文，
对"深色圆形但明显不是葡萄"的物体有区分能力。

两种检测的分工
--------------
    models/classifier.onnx 存在  → 用分类器（更准）
    不存在                      → 退回 HSV 检测（能用）

用法::

    # 单张图片测试
    python3 apps/object_classify.py --image debug/last_raw.jpg

    # 拍照 + 分类
    sudo python3 apps/object_classify.py --camera

    # 只看模型信息
    python3 apps/object_classify.py --info
"""
from __future__ import annotations

import argparse
import json
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

MODEL_PATH = os.path.join(_ROOT, "models", "classifier.onnx")
LABELS_PATH = os.path.join(_ROOT, "models", "classifier_labels.json")

_CN = {"grape": "葡萄", "egg": "鸡蛋", "pingpong": "乒乓球",
       "other": "其他", "unknown": "未知物体"}


def model_available() -> bool:
    return os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH)


class Classifier:
    """ONNX 分类器（CPU 推理，一次几十毫秒）。"""

    def __init__(self, model_path: str = MODEL_PATH,
                 labels_path: str = LABELS_PATH, threads: int = 4):
        import onnxruntime as ort

        if not (os.path.exists(model_path) and os.path.exists(labels_path)):
            raise FileNotFoundError(
                f"找不到模型：{model_path}\n"
                f"  请先在电脑上训练：python tools/train_classifier.py\n"
                f"  然后把 models/classifier.onnx 和 classifier_labels.json 拷过来")

        with open(labels_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        self.classes = meta["classes"]
        self.size = int(meta.get("input_size", 224))
        self.mean = meta.get("normalize_mean", [0.485, 0.456, 0.406])
        self.std = meta.get("normalize_std", [0.229, 0.224, 0.225])
        self.val_acc = meta.get("val_acc")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(
            model_path, sess_options=opts,
            providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.last_ms = 0.0

    def describe(self) -> str:
        acc = f"，验证准确率 {self.val_acc*100:.1f}%" if self.val_acc else ""
        return (f"Classifier({len(self.classes)} 类："
                f"{', '.join(self.classes)}，输入 {self.size}x{self.size}{acc})")

    def predict(self, bgr) -> tuple[str, float, list]:
        """返回 (类别, 置信度, 各类概率列表)。"""
        import numpy as np

        t0 = time.time()
        # BGR -> RGB，缩放，归一化
        from PIL import Image
        rgb = bgr[:, :, ::-1] if bgr.ndim == 3 else bgr
        im = Image.fromarray(np.ascontiguousarray(rgb)).resize(
            (self.size, self.size), Image.BILINEAR)
        x = np.asarray(im, dtype=np.float32) / 255.0
        x = (x - np.array(self.mean, dtype=np.float32)) / \
            np.array(self.std, dtype=np.float32)
        x = np.transpose(x, (2, 0, 1))[None, ...].astype(np.float32)

        logits = self.sess.run(None, {self.in_name: x})[0][0]
        # softmax
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        idx = int(probs.argmax())
        self.last_ms = (time.time() - t0) * 1000
        return self.classes[idx], float(probs[idx]), probs.tolist()


# ---------------------------------------------------------------------------

def grab_frame():
    from k1_camera_capture import K1Camera
    need_sudo = hasattr(os, "geteuid") and os.geteuid() != 0
    return K1Camera(use_sudo=need_sudo).capture()


def main() -> int:
    ap = argparse.ArgumentParser(description="ONNX 分类推理")
    ap.add_argument("--image", default=None, help="单张图片路径")
    ap.add_argument("--camera", action="store_true", help="拍一张再分类")
    ap.add_argument("--info", action="store_true", help="只打印模型信息")
    ap.add_argument("--unknown-threshold", type=float, default=0.55,
                    help="最高置信度低于它就判为 unknown")
    args = ap.parse_args()

    if args.info or not (args.image or args.camera):
        print(f"模型路径：{MODEL_PATH}")
        print(f"类别表　：{LABELS_PATH}")
        if not model_available():
            print("❌ 模型不存在")
            print("   训练：在电脑上 python tools/train_classifier.py")
            print("   部署：scp models/classifier.onnx models/classifier_labels.json \\")
            print("             daiweikai@192.168.137.125:~/grape-gripper/models/")
            return 1
        clf = Classifier()
        print(f"✅ {clf.describe()}")
        return 0

    try:
        clf = Classifier()
    except Exception as exc:  # noqa: BLE001
        print(f"❌ 加载模型失败：{exc}")
        return 1

    print(f"模型：{clf.describe()}")
    print()

    if args.image:
        import cv2
        if not os.path.exists(args.image):
            print(f"❌ 找不到 {args.image}")
            return 1
        bgr = cv2.imread(args.image)
        if bgr is None:
            print(f"❌ 读不出图片 {args.image}")
            return 1
        src = args.image
    else:
        bgr = grab_frame()
        src = "camera"

    cls, conf, probs = clf.predict(bgr)
    print(f"来源：{src}")
    print()
    if conf < args.unknown_threshold:
        print(f"⚠️ 判为未知物体（最高只有 {conf:.2f} < "
              f"{args.unknown_threshold:.2f}）")
    print(f"类别：{_CN.get(cls, cls)}  ({cls})   置信度 {conf:.3f}   "
          f"推理 {clf.last_ms:.0f}ms")
    print()
    print("各类概率：")
    for name, p in sorted(zip(clf.classes, probs), key=lambda t: -t[1]):
        bar = "█" * int(p * 40)
        print(f"  {name:<12} {p*100:5.1f}%  {bar}")
    return 0


if __name__ == "__main__":
    sys.exit(main())