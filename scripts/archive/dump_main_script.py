# -*- coding: utf-8 -*-
"""只提取主脚本 PS01Tool 的字符串，搜索串口命令协议。"""
import struct
import sys
import io
import re
import zlib

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PATH = r"D:\文档\deepseek-harness\default-workspace\ps01tool.exe"
data = open(PATH, "rb").read()

pkg_len = 107504659
toc_off = 107460427
toc_len = 44144
archive_start = len(data) - pkg_len
toc_start = archive_start + toc_off

# 解析 TOC，找主脚本
entries = []
p = toc_start
while p < toc_start + toc_len:
    entry_len = struct.unpack(">i", data[p:p + 4])[0]
    if entry_len < 18 or p + entry_len > len(data):
        break
    data_pos = struct.unpack(">i", data[p + 4:p + 8])[0]
    ulen = struct.unpack(">i", data[p + 8:p + 12])[0]
    length = struct.unpack(">i", data[p + 12:p + 16])[0]
    cflag = data[p + 16]
    typ = chr(data[p + 17])
    name = data[p + 18:p + entry_len].split(b"\x00")[0].decode("utf-8", "replace")
    entries.append((name, typ, data_pos, length, ulen, cflag))
    p += entry_len

main = [e for e in entries if e[0] == "PS01Tool"]
print(f"找到主脚本条目: {main}")
if not main:
    print("没找到 PS01Tool 主脚本")
    sys.exit(1)

name, typ, data_pos, length, ulen, cflag = main[0]
raw = data[archive_start + data_pos: archive_start + data_pos + length]
if cflag:
    raw = zlib.decompress(raw)
print(f"主脚本原始大小: {len(raw)} 字节 (ulen={ulen})")
print(f"头部: {raw[:20].hex(' ')}")

# 字符串扫描
strings = [m.group().decode("ascii", "replace")
           for m in re.finditer(rb"[\x20-\x7e]{3,}", raw)]
print(f"共 {len(strings)} 条字符串\n")

# 保存全部
out = r"D:\文档\deepseek-harness\default-workspace\ps01_unpacked\main_strings.txt"
import os
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as fh:
    fh.write("\n".join(strings))
print(f"全部字符串已保存: {out}\n")

# 重点搜索
keywords = ["cmd", "command", "start", "stop", "acq", "json", "status",
            "pd1", "pd2", "pd3", "sensor", "stream", "sample", "rate",
            "{\"", "set_", "get_", "help", "version", "reset", "tare",
            "zero", "calib", "config"]
print("=" * 70)
print("关键词命中")
print("=" * 70)
seen = set()
for kw in keywords:
    hits = [s for s in strings if kw.lower() in s.lower() and len(s) < 200]
    hits = [h for h in hits if h not in seen]
    if not hits:
        continue
    for h in hits:
        seen.add(h)
    print(f"\n--- {kw!r}: {len(hits)} 条 ---")
    for h in sorted(hits, key=len)[:40]:
        print(f"  {h!r}")
