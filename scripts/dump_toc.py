# -*- coding: utf-8 -*-
"""dump PyInstaller TOC 区域的原始字节，用于核对结构。"""
import struct
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PATH = r"D:\文档\deepseek-harness\default-workspace\ps01tool.exe"
data = open(PATH, "rb").read()

cookie_pos = 107784123
pkg_len = 107504659
toc_off = 107460427
toc_len = 44144

archive_start = len(data) - pkg_len
toc_start = archive_start + toc_off

print(f"archive_start = {archive_start}")
print(f"toc_start     = {toc_start}")
print(f"toc_end       = {toc_start + toc_len}")
print(f"cookie_pos    = {cookie_pos}")
print()

blob = data[toc_start:toc_start + 160]
print("TOC 前 160 字节：")
for i in range(0, len(blob), 16):
    chunk = blob[i:i + 16]
    hx = " ".join(f"{b:02X}" for b in chunk)
    asc = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
    print(f"  +{i:04X}  {hx:<48}  |{asc}|")

print()
# 尝试按 (len, name_pos, data_pos, length, ulen, cflag, type) 解析
print("尝试解析前 5 条：")
p = toc_start
for n in range(5):
    if p + 4 > len(data):
        break
    entry_len = struct.unpack(">i", data[p:p + 4])[0]
    print(f"\n[{n}] @+{p - toc_start:04X} entry_len = {entry_len}")
    if entry_len <= 0 or entry_len > 1000:
        print("    entry_len 不合理，结构假设错误")
        # 试试小端
        entry_len_le = struct.unpack("<i", data[p:p + 4])[0]
        print(f"    (小端解读: {entry_len_le})")
        break
    raw = data[p + 4:p + 4 + entry_len]
    print(f"    原始: {raw.hex(' ')}")
    if entry_len >= 18:
        name_pos, data_pos, length, ulen = struct.unpack(">iiii", raw[:16])
        cflag = raw[16]
        typ = raw[17]
        print(f"    name_pos={name_pos} data_pos={data_pos} len={length} "
              f"ulen={ulen} cflag={cflag} type={chr(typ) if typ else '?'}")
        nm = data[archive_start + name_pos:archive_start + name_pos + 64]
        print(f"    name(相对archive) = {nm.split(b(chr(0)))[0][:60]!r}")
    p += 4 + entry_len
