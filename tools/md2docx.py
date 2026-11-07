# -*- coding: utf-8 -*-
"""
把项目的 Markdown 文档转成排版良好的 Word（.docx）。

为什么不用 pandoc：本机没装。用 python-docx 自己写解析器，
可以精确控制中文字体（w:eastAsia）、表格边框、代码块底纹和页边距。

用法：
    python md2docx.py <输入.md> <输出.docx> [--title 标题] [--subtitle 副标题]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

# ---------------------------------------------------------------- 字体
BODY_FONT = "微软雅黑"
HEAD_FONT = "微软雅黑"
CODE_FONT = "Consolas"
BODY_EA = "微软雅黑"

C_HEAD = RGBColor(0x1F, 0x3A, 0x5F)      # 深蓝
C_ACCENT = RGBColor(0xC0, 0x39, 0x2B)    # 强调红
C_MUTED = RGBColor(0x60, 0x60, 0x60)     # 灰
C_CODE = RGBColor(0x24, 0x29, 0x2E)


def set_run_font(run, name=BODY_FONT, ea=BODY_EA, size=None,
                 bold=None, color=None):
    """★ 中文字体必须同时设 w:eastAsia，否则只改 font.name 不生效。"""
    run.font.name = name
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), name)
    rfonts.set(qn("w:hAnsi"), name)
    rfonts.set(qn("w:eastAsia"), ea)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if color is not None:
        run.font.color.rgb = color


def shade(element, fill: str):
    """给段落或单元格加底色。"""
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    element.append(shd)


def cell_shade(cell, fill: str):
    tc_pr = cell._tc.get_or_add_tcPr()
    shade(tc_pr, fill)


def para_shade(paragraph, fill: str):
    p_pr = paragraph._p.get_or_add_pPr()
    shade(p_pr, fill)


def add_page_number_footer(section):
    """页脚：页码。"""
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    for instr, kind in (("begin", "fldChar"), ("PAGE", "instrText"),
                        ("end", "fldChar")):
        el = OxmlElement("w:" + kind)
        if kind == "fldChar":
            el.set(qn("w:fldCharType"), instr)
        else:
            el.set(qn("xml:space"), "preserve")
            el.text = " PAGE "
        run._r.append(el)
    set_run_font(run, size=9, color=C_MUTED)


def add_toc(doc):
    """插入 Word 目录域（打开后按 F9 或右键更新即可生成）。"""
    p = doc.add_paragraph()
    run = p.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = r' TOC \o "1-3" \h \z \u '
    sep = OxmlElement("w:fldChar")
    sep.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "（打开后按 Ctrl+A 再按 F9 更新目录）"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    for el in (begin, instr, sep, text, end):
        run._r.append(el)


# ---------------------------------------------------------------- 预处理
def normalize(lines: list[str]) -> list[str]:
    """
    把 Markdown 里几种"合法但转换器难处理"的写法规范化。

    ① 引用块里的表格：`> | a | b |` → `| a | b |`
       否则表格会被当成普通段落，输出一堆竖线。
    ② 跨行的粗体：`**开头` + 下一行 + `结尾**`
       如果 `**` 在单行里不成对，就把后续行合并成一段，
       否则 `**` 会原样留在文档里。
    """
    # ① 去掉表格行前面的 "> "
    lines = [re.sub(r"^\s*>\s*(\|.*)$", r"\1", ln) for ln in lines]

    # ①b 表格单元格折行：以 | 开头的行后面如果跟了不以 | 开头的续行，
    #     说明那一格内容换行写了，要接回同一行（否则整张表断裂）
    joined: list[str] = []
    in_fence = False
    for ln in lines:
        if ln.lstrip().startswith("```"):
            in_fence = not in_fence
            joined.append(ln)
            continue
        if (not in_fence and joined
                and joined[-1].lstrip().startswith("|")
                and ln.strip()
                and not ln.lstrip().startswith(("|", "```", "#", ">"))
                and not re.match(r"^\s*([-*+]|\d+[.)])\s+", ln)
                and not re.fullmatch(r"\s*(-{3,}|\*{3,}|_{3,})\s*", ln)):
            joined[-1] = joined[-1].rstrip() + " " + ln.strip()
            continue
        joined.append(ln)
    lines = joined

    # ② 合并跨行的粗体（跳过代码块内部）
    out: list[str] = []
    in_fence = False
    i = 0
    n = len(lines)
    while i < n:
        ln = lines[i]
        if ln.lstrip().startswith("```"):
            in_fence = not in_fence
            out.append(ln)
            i += 1
            continue
        if not in_fence and ln.count("**") % 2 == 1:
            buf = [ln]
            j = i + 1
            # ★ 只在本段落内部合并：遇到空行/标题/表格/引用/列表/分隔线就停，
            #   否则会把整节内容吞成一大段（第一版就踩了这个坑：
            #   段落数从 1120 掉到 655）。
            while (j < n and sum(x.count("**") for x in buf) % 2 == 1):
                nxt = lines[j]
                if (not nxt.strip()
                        or nxt.lstrip().startswith(("```", "#", "|", ">"))
                        or re.match(r"^\s*([-*+]|\d+[.)])\s+", nxt)
                        or re.fullmatch(r"\s*(-{3,}|\*{3,}|_{3,})\s*",
                                        nxt)):
                    break
                buf.append(nxt.strip())
                j += 1
            out.append(" ".join(x.strip() for x in buf))
            i = j
            continue
        out.append(ln)
        i += 1
    return out


# ---------------------------------------------------------------- 行内解析
INLINE_RE = re.compile(
    r"(\*\*.+?\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\)|\*[^*]+\*)", re.S)


def add_inline(paragraph, text, size=10.5, base_color=None):
    """解析 **粗体**、`代码`、[链接](url)、*斜体*。"""
    # 安全网：即使预处理没兜住，也不把字面 ** 留在文档里
    if text.count("**") % 2 == 1:
        text = text.replace("**", "")
    for part in INLINE_RE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            r = paragraph.add_run(part[2:-2])
            set_run_font(r, size=size, bold=True, color=base_color)
        elif part.startswith("`") and part.endswith("`") and len(part) > 2:
            r = paragraph.add_run(part[1:-1])
            set_run_font(r, name=CODE_FONT, ea=CODE_FONT, size=size - 0.5,
                         color=C_CODE)
            r.font.highlight_color = None
            shade(r._element.get_or_add_rPr(), "F2F2F2")
        elif part.startswith("[") and "](" in part:
            m = re.match(r"\[([^\]]+)\]\(([^)]+)\)", part)
            label, url = (m.group(1), m.group(2)) if m else (part, "")
            r = paragraph.add_run(label)
            set_run_font(r, size=size, color=RGBColor(0x1F, 0x4E, 0x79))
            if url and url != label:
                r2 = paragraph.add_run(f"（{url}）")
                set_run_font(r2, size=size - 1, color=C_MUTED)
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            r = paragraph.add_run(part[1:-1])
            set_run_font(r, size=size, bold=False, color=base_color)
            r.font.italic = True
        else:
            r = paragraph.add_run(part)
            set_run_font(r, size=size, color=base_color)


# ---------------------------------------------------------------- 主转换
def convert(md_path: Path, docx_path: Path, title: str, subtitle: str):
    lines = md_path.read_text(encoding="utf-8").splitlines()
    lines = normalize(lines)

    doc = Document()

    # 页面：A4 + 适中页边距
    sec = doc.sections[0]
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.left_margin = Cm(2.4)
    sec.right_margin = Cm(2.4)
    sec.top_margin = Cm(2.4)
    sec.bottom_margin = Cm(2.2)
    add_page_number_footer(sec)

    # 正文默认样式
    normal = doc.styles["Normal"]
    normal.font.name = BODY_FONT
    normal.font.size = Pt(10.5)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_EA)
    normal.paragraph_format.line_spacing = 1.45
    normal.paragraph_format.space_after = Pt(5)

    # ---------------- 封面 ----------------
    for _ in range(6):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(title)
    # ★ 封面标题字号按字数自动收缩：太长会换行拆开（"终端"被拆成两行）
    tlen = len(title)
    tsize = 30 if tlen <= 12 else (26 if tlen <= 16 else 22)
    set_run_font(r, name=HEAD_FONT, ea=HEAD_FONT, size=tsize, bold=True,
                 color=C_HEAD)

    if subtitle:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(16)
        r = p.add_run(subtitle)
        set_run_font(r, size=15, color=C_MUTED)

    for _ in range(8):
        doc.add_paragraph()

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("基于进迭时空 K1（MUSE Pi Pro）平台")
    set_run_font(r, size=11, color=C_MUTED)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("2026 年 11 月")
    set_run_font(r, size=11, color=C_MUTED)

    doc.add_page_break()

    # ---------------- 目录 ----------------
    p = doc.add_paragraph()
    r = p.add_run("目  录")
    set_run_font(r, name=HEAD_FONT, ea=HEAD_FONT, size=18, bold=True,
                 color=C_HEAD)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_toc(doc)
    doc.add_page_break()

    # ---------------- 正文 ----------------
    i = 0
    n = len(lines)
    first_h1_seen = False
    in_code = False
    code_buf: list[str] = []
    code_lang = ""

    def flush_code():
        nonlocal code_buf, code_lang
        if not code_buf:
            return
        # 代码块：每行一个段落，浅灰底、等宽字体
        for ln in code_buf:
            p = doc.add_paragraph()
            pf = p.paragraph_format
            pf.left_indent = Cm(0.6)
            pf.space_before = Pt(0)
            pf.space_after = Pt(0)
            pf.line_spacing = 1.1
            para_shade(p, "F5F6F7")
            r = p.add_run(ln if ln else " ")
            set_run_font(r, name=CODE_FONT, ea=CODE_FONT, size=9,
                         color=C_CODE)
        doc.add_paragraph().paragraph_format.space_after = Pt(2)
        code_buf = []
        code_lang = ""

    while i < n:
        raw = lines[i]
        line = raw.rstrip()

        # 代码围栏
        if line.lstrip().startswith("```"):
            if in_code:
                flush_code()
                in_code = False
            else:
                in_code = True
                code_lang = line.strip().strip("`").strip()
            i += 1
            continue
        if in_code:
            code_buf.append(raw.rstrip("\n"))
            i += 1
            continue

        # 空行
        if not line.strip():
            i += 1
            continue

        # 分隔线
        if re.fullmatch(r"\s*(-{3,}|\*{3,}|_{3,})\s*", line):
            i += 1
            continue

        # 标题
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            level = len(m.group(1))
            text = m.group(2).strip()
            # 去掉标题里的 emoji 星标但保留 ★ 作为强调
            if level == 1:
                if first_h1_seen:
                    doc.add_page_break()
                first_h1_seen = True
                h = doc.add_heading(level=1)
                r = h.add_run(text)
                set_run_font(r, name=HEAD_FONT, ea=HEAD_FONT, size=20,
                             bold=True, color=C_HEAD)
                h.alignment = WD_ALIGN_PARAGRAPH.CENTER
                h.paragraph_format.space_before = Pt(6)
                h.paragraph_format.space_after = Pt(14)
            elif level == 2:
                h = doc.add_heading(level=2)
                r = h.add_run(text)
                set_run_font(r, name=HEAD_FONT, ea=HEAD_FONT, size=15.5,
                             bold=True, color=C_HEAD)
                h.paragraph_format.space_before = Pt(16)
                h.paragraph_format.space_after = Pt(8)
                # 下边框
                p_pr = h._p.get_or_add_pPr()
                bdr = OxmlElement("w:pBdr")
                bottom = OxmlElement("w:bottom")
                bottom.set(qn("w:val"), "single")
                bottom.set(qn("w:sz"), "8")
                bottom.set(qn("w:color"), "1F3A5F")
                bottom.set(qn("w:space"), "3")
                bdr.append(bottom)
                p_pr.append(bdr)
            elif level == 3:
                h = doc.add_heading(level=3)
                r = h.add_run(text)
                set_run_font(r, name=HEAD_FONT, ea=HEAD_FONT, size=13,
                             bold=True, color=C_HEAD)
                h.paragraph_format.space_before = Pt(11)
                h.paragraph_format.space_after = Pt(5)
            else:
                h = doc.add_heading(level=4)
                r = h.add_run(text)
                set_run_font(r, name=HEAD_FONT, ea=HEAD_FONT, size=11.5,
                             bold=True, color=C_ACCENT)
                h.paragraph_format.space_before = Pt(8)
                h.paragraph_format.space_after = Pt(4)
            i += 1
            continue

        # 表格
        if line.lstrip().startswith("|") and i + 1 < n and \
                re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            header = [c.strip() for c in line.strip().strip("|").split("|")]
            i += 2
            rows = []
            while i < n and lines[i].lstrip().startswith("|"):
                rows.append([c.strip()
                             for c in lines[i].strip().strip("|").split("|")])
                i += 1
            ncol = max(len(header), max((len(r) for r in rows), default=0))
            table = doc.add_table(rows=1, cols=ncol)
            table.style = "Table Grid"
            table.alignment = WD_TABLE_ALIGNMENT.CENTER
            hdr = table.rows[0].cells
            for j in range(ncol):
                cell = hdr[j]
                cell.text = ""
                para = cell.paragraphs[0]
                para.paragraph_format.space_after = Pt(2)
                add_inline(para, header[j] if j < len(header) else "",
                           size=9.5)
                for r_ in para.runs:
                    r_.font.bold = True
                    r_.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                cell_shade(cell, "1F3A5F")
            for row in rows:
                cells = table.add_row().cells
                for j in range(ncol):
                    cell = cells[j]
                    cell.text = ""
                    para = cell.paragraphs[0]
                    para.paragraph_format.space_after = Pt(2)
                    add_inline(para, row[j] if j < len(row) else "", size=9.5)
            doc.add_paragraph().paragraph_format.space_after = Pt(4)
            continue

        # 引用块
        if line.lstrip().startswith(">"):
            text = re.sub(r"^\s*>\s?", "", line)
            p = doc.add_paragraph()
            pf = p.paragraph_format
            pf.left_indent = Cm(0.8)
            pf.space_before = Pt(4)
            pf.space_after = Pt(4)
            p_pr = p._p.get_or_add_pPr()
            bdr = OxmlElement("w:pBdr")
            left = OxmlElement("w:left")
            left.set(qn("w:val"), "single")
            left.set(qn("w:sz"), "18")
            left.set(qn("w:color"), "C0392B")
            left.set(qn("w:space"), "8")
            bdr.append(left)
            p_pr.append(bdr)
            add_inline(p, text, size=10, base_color=RGBColor(0x44, 0x44, 0x44))
            i += 1
            continue

        # 列表
        m = re.match(r"^(\s*)([-*+])\s+(.*)$", line)
        if m:
            indent = len(m.group(1)) // 2
            p = doc.add_paragraph(style="List Bullet")
            p.paragraph_format.left_indent = Cm(0.75 + 0.6 * indent)
            p.paragraph_format.space_after = Pt(3)
            add_inline(p, m.group(3), size=10.5)
            i += 1
            continue

        m = re.match(r"^(\s*)(\d+)[.)]\s+(.*)$", line)
        if m:
            indent = len(m.group(1)) // 2
            p = doc.add_paragraph(style="List Number")
            p.paragraph_format.left_indent = Cm(0.75 + 0.6 * indent)
            p.paragraph_format.space_after = Pt(3)
            add_inline(p, m.group(3), size=10.5)
            i += 1
            continue

        # 普通段落（合并连续行）
        buf = [line.strip()]
        i += 1
        while i < n:
            nxt = lines[i]
            if not nxt.strip():
                break
            if nxt.lstrip().startswith(("```", "|", ">", "#")):
                break
            if re.match(r"^\s*([-*+]|\d+[.)])\s+", nxt):
                break
            if re.fullmatch(r"\s*(-{3,}|\*{3,}|_{3,})\s*", nxt):
                break
            buf.append(nxt.strip())
            i += 1
        text = " ".join(buf)
        p = doc.add_paragraph()
        p.paragraph_format.first_line_indent = Cm(0)
        add_inline(p, text, size=10.5)

    doc.save(docx_path)
    return docx_path


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    md = Path(sys.argv[1])
    out = Path(sys.argv[2])
    title = ""
    subtitle = ""
    args = sys.argv[3:]
    for k in range(0, len(args) - 1, 2):
        if args[k] == "--title":
            title = args[k + 1]
        elif args[k] == "--subtitle":
            subtitle = args[k + 1]
    if not title:
        title = md.stem
    convert(md, out, title, subtitle)
    print(f"OK  {out}  {out.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
