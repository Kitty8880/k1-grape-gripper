#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 Markdown 文档转成 Word（.docx）。

支持的 Markdown 子集
--------------------
    # / ## / ### / ####     标题
    - / *                   无序列表（支持 2 空格缩进嵌套）
    1.                      有序列表
    | a | b |               表格（第二行是 |---|---|）
    ``` ... ```             代码块
    > ...                   引用块
    **粗体**  `行内代码`
    ---                     分隔线

中文字体处理
------------
python-docx 只设 run.font.name 对中文**不生效** —— 必须同时设置
w:eastAsia 属性。本脚本统一处理了这一点，否则中文会退化成默认字体。

用法::

    python tools/md_to_docx.py --in docs/课程知识说明.md --out 课程知识说明.docx
"""
from __future__ import annotations

import argparse
import os
import re
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

from docx import Document                                    # noqa: E402
from docx.enum.table import WD_TABLE_ALIGNMENT               # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH                # noqa: E402
from docx.oxml.ns import qn                                  # noqa: E402
from docx.shared import Pt, RGBColor, Inches                 # noqa: E402

CN_FONT = "微软雅黑"
CODE_FONT = "Consolas"


def set_run(run, size=None, bold=None, color=None, font=None,
            mono=False):
    """统一设置字体，中文必须走 eastAsia。"""
    name = font or (CODE_FONT if mono else CN_FONT)
    run.font.name = name
    rpr = run._element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rf)
    rf.set(qn("w:ascii"), name)
    rf.set(qn("w:hAnsi"), name)
    rf.set(qn("w:eastAsia"), name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if color is not None:
        run.font.color.rgb = color


INLINE = re.compile(r"(\*\*.+?\*\*|`[^`]+`)")


def add_inline(par, text, size=10.5, base_bold=False, color=None):
    """处理 **粗体** 和 `行内代码`。"""
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            r = par.add_run(part[2:-2])
            set_run(r, size=size, bold=True, color=color)
        elif part.startswith("`") and part.endswith("`") and len(part) > 2:
            r = par.add_run(part[1:-1])
            set_run(r, size=size - 0.5, mono=True,
                    color=RGBColor(0xC7, 0x25, 0x4E))
        else:
            r = par.add_run(part)
            set_run(r, size=size, bold=base_bold or None, color=color)


def build(doc, md_text: str):
    lines = md_text.split("\n")
    i = 0
    n = len(lines)

    # 文档标题
    if lines and lines[0].startswith("# "):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(lines[0][2:].strip())
        set_run(r, size=22, bold=True)
        p.paragraph_format.space_after = Pt(18)
        i = 1

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # ---- 代码块 ----
        if stripped.startswith("```"):
            i += 1
            buf = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.3)
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(8)
            r = p.add_run("\n".join(buf))
            set_run(r, size=9, mono=True,
                    color=RGBColor(0x1F, 0x3A, 0x5F))
            continue

        # ---- 表格 ----
        if stripped.startswith("|") and i + 1 < n and \
                re.match(r"^\|[\s:|-]+\|$", lines[i + 1].strip()):
            header = [c.strip() for c in stripped.strip("|").split("|")]
            i += 2
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c
                             in lines[i].strip().strip("|").split("|")])
                i += 1
            ncol = len(header)
            tbl = doc.add_table(rows=1, cols=ncol)
            tbl.style = "Light Grid Accent 1"
            tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
            hdr = tbl.rows[0].cells
            for ci, h in enumerate(header):
                hdr[ci].text = ""
                p = hdr[ci].paragraphs[0]
                add_inline(p, h, size=10, base_bold=True)
            for row in rows:
                cells = tbl.add_row().cells
                for ci in range(ncol):
                    val = row[ci] if ci < len(row) else ""
                    cells[ci].text = ""
                    p = cells[ci].paragraphs[0]
                    add_inline(p, val, size=9.5)
            doc.add_paragraph()
            continue

        # ---- 分隔线 ----
        if stripped in ("---", "***", "___"):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            r = p.add_run("─" * 46)
            set_run(r, size=8, color=RGBColor(0xC0, 0xC0, 0xC0))
            i += 1        # ⚠️ 少了这行就是死循环（踩过：进程 CPU 100% 卡死）
            continue

        # ---- 标题 ----
        m = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if m:
            level = len(m.group(1))
            text = m.group(2)
            # ★ 用 Word 真正的标题样式（Heading N），而不是手工加粗放大。
            #   只有用了真样式，Word 才有导航窗格、才能一键生成目录 ——
            #   评审打开文档时靠这个跳章节。手工格式做不到。
            style_name = f"Heading {min(level, 4)}"
            try:
                p = doc.add_paragraph(style=style_name)
            except KeyError:
                p = doc.add_paragraph()      # 模板里没有该样式就退回手工格式
            pf = p.paragraph_format
            if level == 1:
                pf.space_before = Pt(0)
                pf.space_after = Pt(18)
                size = 20
            elif level == 2:
                pf.space_before = Pt(16)
                pf.space_after = Pt(6)
                size = 16
            elif level == 3:
                pf.space_before = Pt(12)
                pf.space_after = Pt(4)
                size = 13
            else:
                pf.space_before = Pt(10)
                pf.space_after = Pt(3)
                size = 11.5
            r = p.add_run(text)
            set_run(r, size=size, bold=True,
                    color=RGBColor(0x14, 0x3C, 0x64) if level <= 2
                    else RGBColor(0x1F, 0x1F, 0x1F))
            i += 1
            continue

        # ---- 引用块 ----
        if stripped.startswith(">"):
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip().lstrip(">").strip())
                i += 1
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.35)
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(8)
            add_inline(p, "\n".join(buf), size=10,
                       color=RGBColor(0x50, 0x50, 0x50))
            continue

        # ---- 列表 ----
        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", line)
        if m:
            indent = len(m.group(1)) // 2
            marker = m.group(2)
            text = m.group(3)
            p = doc.add_paragraph()
            pf = p.paragraph_format
            pf.left_indent = Inches(0.28 + indent * 0.28)
            pf.space_after = Pt(2)
            bullet = "•  " if marker in ("-", "*") else f"{marker}  "
            r = p.add_run(bullet)
            set_run(r, size=10.5, color=RGBColor(0x1F, 0x6F, 0xB2))
            add_inline(p, text, size=10.5)
            i += 1
            continue

        # ---- 空行 ----
        if not stripped:
            i += 1
            continue

        # ---- 普通段落 ----
        buf = [stripped]
        i += 1
        while i < n and lines[i].strip() and \
                not re.match(r"^(\s*)([-*]|\d+\.|#|>|\|)", lines[i]) and \
                not lines[i].strip().startswith("```") and \
                lines[i].strip() not in ("---", "***", "___"):
            buf.append(lines[i].strip())
            i += 1
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing = 1.35
        add_inline(p, " ".join(buf), size=10.5)


def normalize_outline(doc) -> None:
    """
    让 Word 的大纲结构正确。

    为什么需要这一步：Markdown 里文档主标题是 `#`、章节是 `##`。
    照搬成 Word 的 Heading 1 / Heading 2 之后，"章节"落在大纲第 2 层，
    导航窗格里会挂在空的第一层下面，也不方便一键生成目录。

    这里统一**上移一层**：
        第一个段落（文档主标题） → Title 样式
        原 Heading 2 → Heading 1（章节）
        原 Heading 3 → Heading 2（小节）
    这样导航窗格和自动目录都正常。
    """
    # 1) 主标题：第一个非空段落设为 Title
    for para in doc.paragraphs:
        if para.text.strip():
            try:
                para.style = doc.styles["Title"]
            except KeyError:
                pass
            break

    # 2) 其余标题上移一层
    #
    # ⚠️ 必须**先记录原始层级再一次性改**。
    #    如果按"从深层到浅层"循环改，会级联：H5→H4→H3→H2→H1，
    #    最后所有标题都塌成 H1（踩过）。
    plan = []
    for para in doc.paragraphs:
        name = para.style.name
        if name.startswith("Heading "):
            try:
                plan.append((para, int(name.split()[-1])))
            except ValueError:
                continue

    for para, lvl in plan:
        new_lvl = max(1, min(4, lvl - 1))
        try:
            para.style = doc.styles[f"Heading {new_lvl}"]
        except KeyError:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description="Markdown 转 Word")
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    args = ap.parse_args()

    if not os.path.exists(args.src):
        print(f"❌ 找不到 {args.src}")
        return 1

    with open(args.src, encoding="utf-8") as fh:
        md = fh.read()

    doc = Document()
    # 页边距
    for sec in doc.sections:
        sec.left_margin = Inches(0.9)
        sec.right_margin = Inches(0.9)
        sec.top_margin = Inches(0.8)
        sec.bottom_margin = Inches(0.8)
    # 默认样式
    style = doc.styles["Normal"]
    style.font.name = CN_FONT
    style.font.size = Pt(10.5)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), CN_FONT)

    build(doc, md)
    normalize_outline(doc)
    doc.save(args.dst)

    # 验证
    d2 = Document(args.dst)
    n_par = len(d2.paragraphs)
    n_tbl = len(d2.tables)
    chars = sum(len(p.text) for p in d2.paragraphs)
    chars += sum(len(c.text) for t in d2.tables for r in t.rows
                 for c in r.cells)
    print(f"✅ {args.dst}")
    print(f"   段落 {n_par} 个 / 表格 {n_tbl} 个 / 约 {chars:,} 字符")
    return 0


if __name__ == "__main__":
    sys.exit(main())