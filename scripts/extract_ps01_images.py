# -*- coding: utf-8 -*-
"""导出 docx 内所有内嵌图片，用于查看接口定义图与上位机界面。"""
import os
import sys
import io
from docx import Document

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

path = r"D:\文档\deepseek-harness\default-workspace\ps01_manual.docx"
outdir = r"D:\文档\deepseek-harness\default-workspace\ps01_images"
os.makedirs(outdir, exist_ok=True)

doc = Document(path)
rels = doc.part.rels

count = 0
for rel_id, rel in rels.items():
    if "image" not in rel.reltype:
        continue
    count += 1
    blob = rel.target_part.blob
    ext = os.path.splitext(rel.target_ref)[1] or ".png"
    out = os.path.join(outdir, f"img{count:02d}{ext}")
    with open(out, "wb") as fh:
        fh.write(blob)
    print(f"{out}  ({len(blob) // 1024} KB)")

print(f"\n共导出 {count} 张图片到 {outdir}")
