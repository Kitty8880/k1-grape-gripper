#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分类模型训练脚本 —— 在你自己的电脑（Windows/Linux x86）上运行。

为什么训练放在电脑上而不是 K1
------------------------------
· K1 是 RISC-V，PyTorch 官方没有 riscv64 轮子，装不了
· 训练只要几分钟，没必要在板子上跑
· 训完导出一个几 MB 的 ONNX 文件，拷到板子上用 onnxruntime 推理即可

⚠️ 重要：图片必须用 K1 摄像头在部署环境拍
   用手机拍的图片会因为领域差异（domain gap）在部署时识别率大跌。
   见 apps/capture_dataset.py。

依赖安装（一次性）
------------------
    pip install torch torchvision onnx onnxruntime pillow numpy

    # 如果下载太慢，用国内镜像：
    pip install torch torchvision onnx onnxruntime pillow numpy ^
        -i https://pypi.tuna.tsinghua.edu.cn/simple

用法
----
    # 数据放在 datasets/ 下，每个类别一个子目录：
    #   datasets/grape/*.jpg
    #   datasets/egg/*.jpg
    #   datasets/pingpong/*.jpg
    #   datasets/other/*.jpg

    python tools/train_classifier.py
    python tools/train_classifier.py --epochs 30 --batch 16
    python tools/train_classifier.py --arch mobilenet_v3_small

输出
----
    models/classifier.onnx      分类模型（拷到板子）
    models/classifier_labels.json  类别列表（顺序必须和模型输出对应）

为什么用迁移学习
----------------
只有几百张图时，从零训练几乎不可能收敛。用 ImageNet 预训练的骨干网络
+ 只训练最后的分类头，几百张图就能到 90%+ 准确率。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# ⚠️ Windows 控制台默认是 GBK 编码，打不出 ⚠️/★ 这类字符会抛
#    UnicodeEncodeError。必须先把 stdout/stderr 改成 UTF-8。
#    （冒烟测试时踩到的坑）
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

DEFAULT_DATA = os.path.join(ROOT, "datasets")
DEFAULT_OUT = os.path.join(ROOT, "models")


def main() -> int:
    ap = argparse.ArgumentParser(description="分类模型训练")
    ap.add_argument("--data", default=DEFAULT_DATA, help="数据集根目录")
    ap.add_argument("--out", default=DEFAULT_OUT, help="输出目录")
    ap.add_argument("--arch", default="mobilenet_v3_small",
                    choices=["mobilenet_v3_small", "mobilenet_v3_large",
                             "efficientnet_b0", "resnet18"],
                    help="骨干网络（小数据集推荐 mobilenet_v3_small）")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--size", type=int, default=224, help="输入边长")
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None, help="cpu / cuda，默认自动")
    args = ap.parse_args()

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, random_split
        from torchvision import datasets, models, transforms
    except ImportError as exc:
        print(f"❌ 缺少依赖：{exc}")
        print("   pip install torch torchvision onnx onnxruntime pillow numpy")
        return 1

    if not os.path.isdir(args.data):
        print(f"❌ 数据集目录不存在：{args.data}")
        return 1

    classes = sorted(d for d in os.listdir(args.data)
                     if os.path.isdir(os.path.join(args.data, d)))
    if len(classes) < 2:
        print(f"❌ 至少需要 2 个类别，现在只有 {classes}")
        print(f"   目录结构应为： {args.data}/<类别名>/*.jpg")
        return 1

    # ---- 统计每个类别的图片数 ----
    print("=" * 70)
    print("数据集概览")
    print("=" * 70)
    counts = {}
    for c in classes:
        cd = os.path.join(args.data, c)
        n = len([f for f in os.listdir(cd)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        counts[c] = n
        flag = "⚠️ 偏少" if n < 80 else ("🔶 一般" if n < 150 else "✅")
        print(f"  {c:<14} {n:>4} 张   {flag}")
    total = sum(counts.values())
    print(f"  {'合计':<14} {total:>4} 张")
    if min(counts.values()) < 50:
        print()
        print("  ⚠️ 有类别少于 50 张，准确率可能不理想。建议每类至少 100 张。")
    print()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  设备 {device}   骨干 {args.arch}   输入 {args.size}x{args.size}")
    print(f"  轮数 {args.epochs}   批大小 {args.batch}   学习率 {args.lr}")
    print()

    torch.manual_seed(args.seed)

    # ---- 数据增强 ----
    # 只做"拍摄时可能发生的变化"：位置、尺度、亮度、轻微旋转
    # 不做翻转（左右翻转会改变物体的朝向语义，对鸡蛋之类的对称物体无害，
    #           但对"手"这类非对称目标有害 —— 这里保守一点）
    train_tf = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.RandomCrop(args.size, padding=args.size // 8),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomRotation(12),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    full = datasets.ImageFolder(args.data, transform=train_tf)
    if full.classes != classes:
        print(f"⚠️ 类别顺序不一致：{full.classes} vs {classes}")
        classes = full.classes

    n_val = max(1, int(len(full) * args.val_split))
    n_train = len(full) - n_val
    train_ds, val_ds = random_split(
        full, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed))
    # 验证集不做增强
    val_ds.dataset = datasets.ImageFolder(args.data, transform=val_tf)
    val_ds = torch.utils.data.Subset(val_ds.dataset, val_ds.indices)

    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=0)
    val_ld = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=0)
    print(f"  训练集 {n_train} 张   验证集 {n_val} 张")
    print()

    # ---- 建模型（迁移学习）----
    def build(arch: str, n_cls: int):
        if arch == "mobilenet_v3_small":
            m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
            m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_cls)
        elif arch == "mobilenet_v3_large":
            m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
            m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_cls)
        elif arch == "efficientnet_b0":
            m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_cls)
        else:
            m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            m.fc = nn.Linear(m.fc.in_features, n_cls)
        return m

    model = build(args.arch, len(classes)).to(device)

    # 只训练分类头（骨干冻结）—— 小数据集这样最稳
    for p in model.parameters():
        p.requires_grad = False
    head_params = []
    for name, p in model.named_parameters():
        if any(k in name for k in ("classifier", "fc", "head")):
            p.requires_grad = True
            head_params.append(p)
    print(f"  可训练参数：{sum(p.numel() for p in head_params):,}")
    print()

    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(head_params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_acc = 0.0
    best_state = None
    t_start = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        tot, correct, loss_sum = 0, 0, 0.0
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            tot += y.size(0)
        sched.step()
        tr_acc = correct / max(1, tot)
        tr_loss = loss_sum / max(1, tot)

        model.eval()
        vtot, vcorrect = 0, 0
        with torch.no_grad():
            for x, y in val_ld:
                x, y = x.to(device), y.to(device)
                out = model(x)
                vcorrect += (out.argmax(1) == y).sum().item()
                vtot += y.size(0)
        v_acc = vcorrect / max(1, vtot)

        mark = ""
        if v_acc > best_acc:
            best_acc = v_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            mark = "  ★ 最佳"
        print(f"  第 {ep:3d}/{args.epochs} 轮   "
              f"训练 loss {tr_loss:.3f} acc {tr_acc:.3f}   "
              f"验证 acc {v_acc:.3f}{mark}")

    print()
    print(f"训练完成，用时 {time.time()-t_start:.0f}s，最佳验证准确率 {best_acc:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # ---- 逐类统计（混淆矩阵）----
    print()
    print("逐类准确率：")
    per_cls = {c: [0, 0] for c in classes}
    with torch.no_grad():
        for x, y in val_ld:
            x = x.to(device)
            out = model(x).argmax(1).cpu()
            for t, p in zip(y, out):
                per_cls[classes[t]][1] += 1
                if t == p:
                    per_cls[classes[t]][0] += 1
    for c in classes:
        ok, tot_c = per_cls[c]
        acc = ok / tot_c if tot_c else 0.0
        print(f"  {c:<14} {ok:>3}/{tot_c:<3}  {acc:.3f}"
              f"{'   ⚠️ 偏低' if acc < 0.85 else ''}")

    # ---- 导出 ONNX ----
    os.makedirs(args.out, exist_ok=True)
    onnx_path = os.path.join(args.out, "classifier.onnx")
    dummy = torch.randn(1, 3, args.size, args.size).to(device)

    print()
    print(f"导出 ONNX → {onnx_path}")
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"], output_names=["logits"],
        opset_version=11, dynamic_axes={"input": {0: "batch"},
                                        "logits": {0: "batch"}},
        do_constant_folding=True,
    )

    labels_path = os.path.join(args.out, "classifier_labels.json")
    meta = {
        "classes": classes,
        "input_size": args.size,
        "arch": args.arch,
        "val_acc": round(best_acc, 4),
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
        "per_class_acc": {c: round(per_cls[c][0] / per_cls[c][1], 4)
                          if per_cls[c][1] else 0.0 for c in classes},
    }
    with open(labels_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    print(f"类别表     → {labels_path}")
    print()
    print("=" * 70)
    print("下一步：把这两个文件拷到 K1")
    print("=" * 70)
    print(f"  scp {onnx_path} daiweikai@192.168.137.125:~/grape-gripper/models/")
    print(f"  scp {labels_path} daiweikai@192.168.137.125:~/grape-gripper/models/")
    print()
    print("然后在板子上试：")
    print("  python3 apps/object_classify.py --image datasets/other/other_0000.jpg")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())