# -*- coding: utf-8 -*-
"""
扫描主脚本 PS01Tool 的所有字符串常量：
- ASCII（含短的）
- UTF-8 中文
- bytes 常量
专门找串口命令。
"""
import struct
import sys
import io
import re
import zlib
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PATH = r"D:\文档\deepseek-harness\default-workspace\ps01tool.exe"
data = open(PATH, "rb").read()

pkg_len = 107504659
toc_off = 107460427
toc_len = 44144
archive_start = len(data) - pkg_len
toc_start = archive_start + toc_off

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

main = [e for e in entries if e[0] == "PS01Tool"][0]
name, typ, data_pos, length, ulen, cflag = main
raw = data[archive_start + data_pos: archive_start + data_pos + length]
if cflag:
    raw = zlib.decompress(raw)

print(f"主脚本: {len(raw)} 字节\n")

# ---- 1) UTF-8 中文串 ----
print("=" * 70)
print("① UTF-8 中文串")
print("=" * 70)
zh = re.findall(rb'(?:[\xe4-\xe9][\x80-\xbf]{2}){2,}', raw)
seen = set()
for z in zh:
    try:
        s = z.decode("utf-8")
    except UnicodeDecodeError:
        continue
    if s not in seen and len(s) <= 60:
        seen.add(s)
        print(f"  {s!r}")

# ---- 2) 短 ASCII 串（长度 2-25，可能是命令）----
print()
print("=" * 70)
print("② 短 ASCII 串（长度 2~25）")
print("=" * 70)
short = set()
for m in re.finditer(rb"[\x20-\x7e]{2,25}", raw):
    s = m.group().decode("ascii")
    # 过滤明显是字节码碎片/变量名的
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,24}", s):
        short.add(s)
for s in sorted(short):
    print(f"  {s!r}")

# ---- 3) 疑似命令：全大写、或含下划线的短语 ----
print()
print("=" * 70)
print("③ 疑似命令（全大写 / 含特殊字符）")
print("=" * 70)
cand = [s for s in short
        if s.isupper() or "_" in s or s.startswith(("!", ":", "#"))
        or s.lower() in ("start", "stop", "on", "off", "run", "go",
                         "begin", "end", "acq", "read", "set", "get",
                         "help", "reset", "init", "connect", "start_acq")]
for s in sorted(cand):
    print(f"  {s!r}")

# ---- 4) 把 raw 里的 "send_command" 附近字节 dump 出来 ----
print()
print("=" * 70)
print("④ send_command / send_text_command 附近字节")
print("=" * 70)
for kw in (b"send_text_command", b"send_command", b"cmd_input"):
    idx = raw.find(kw)
    while idx != -1:
        lo = max(0, idx - 120)
        hi = min(len(raw), idx + len(kw) + 120)
        snippet = raw[lo:hi]
        asc = "".join(chr(b) if 32 <= b < 127 else "." for b in snippet)
        print(f"\n  位置 {idx}: {asc}")
        idx = raw.find(kw, idx + 1)
        if idx > 30000:
            break
