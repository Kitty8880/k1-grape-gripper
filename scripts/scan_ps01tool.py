# -*- coding: utf-8 -*-
"""
从 PS01Tool.exe 里提取可打印字符串，搜索串口命令格式。

107MB 的 exe，直接全量正则可能很慢，所以用分块扫描。
优先找 JSON 命令相关的关键词。
"""
import re
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PATH = r"D:\文档\deepseek-harness\default-workspace\ps01tool.exe"

# 关注的关键词（命令行协议相关）
KEYWORDS = [
    b'"cmd"', b'"command"', b'cmd', b'start', b'stop', b'stream',
    b'acq', b'acquire', b'json', b'read', b'config', b'sample',
    b'freq', b'rate', b'set_', b'get_', b'{"', b'"status"',
    b'pd1', b'pd2', b'pd3', b'PD1', b'PD2', b'PD3',
    b'sensor_', b'help',
]

# ASCII 字符串：4 个以上可打印字符
ASCII_RE = re.compile(rb"[\x20-\x7e]{5,}")
# UTF-16LE 字符串（Windows 程序常见）
UTF16_RE = re.compile(rb"(?:[\x20-\x7e]\x00){5,}")

CHUNK = 8 * 1024 * 1024   # 8MB 一块


def scan(path: str) -> None:
    hits = {k: set() for k in KEYWORDS}
    total_ascii = 0
    total_utf16 = 0

    with open(path, "rb") as fh:
        offset = 0
        while True:
            chunk = fh.read(CHUNK)
            if not chunk:
                break
            # 为跨块的字符串保留尾部重叠
            for m in ASCII_RE.finditer(chunk):
                s = m.group()
                total_ascii += 1
                low = s.lower()
                for kw in KEYWORDS:
                    if kw.lower() in low and len(s) < 300:
                        hits[kw].add(s)
            for m in UTF16_RE.finditer(chunk):
                s = m.group().decode("utf-16-le", errors="ignore").encode("ascii", "ignore")
                if len(s) >= 5:
                    total_utf16 += 1
                    low = s.lower()
                    for kw in KEYWORDS:
                        if kw.lower() in low and len(s) < 300:
                            hits[kw].add(s)
            offset += len(chunk)
            print(f"  已扫描 {offset // (1024*1024)} MB …", end="\r", flush=True)

    print(f"\n\n扫描完成：ASCII 字符串 {total_ascii} 个，UTF-16 字符串 {total_utf16} 个\n")

    for kw in KEYWORDS:
        found = hits[kw]
        if not found:
            continue
        label = kw.decode("ascii", "replace")
        print(f"===== 关键词 {label!r}：{len(found)} 条 =====")
        # 只展示最像命令的（短的、含 cmd/json 的）
        interesting = sorted(found, key=lambda x: (len(x), x))[:25]
        for s in interesting:
            try:
                text = s.decode("utf-8")
            except UnicodeDecodeError:
                text = s.decode("latin-1")
            print(f"  {text}")
        print()


if __name__ == "__main__":
    import os
    if not os.path.isfile(PATH):
        print(f"文件不存在：{PATH}")
        sys.exit(1)
    scan(PATH)
